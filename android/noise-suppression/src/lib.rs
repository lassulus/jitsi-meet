//! DeepFilterNet3 noise suppression engine exposed over JNI.
//!
//! Java-side contract (`org.jitsi.meet.sdk.DeepFilterNet`, loaded via
//! `System.loadLibrary("dfjni")`):
//!
//! - `long nativeNew(float attenLimDb)` — create a single-channel engine with
//!   the embedded DeepFilterNet3 model. `attenLimDb == 0` means no attenuation
//!   limit. Returns a handle, or 0 on any failure. Never throws.
//! - `int nativeFrameLength(long handle)` — model hop size in samples
//!   (480 at 48 kHz). 0 for an invalid handle.
//! - `int nativeSampleRate(long handle)` — model sample rate (48000).
//!   0 for an invalid handle.
//! - `float nativeProcessInPlace(long handle, ByteBuffer buffer, int numFrames, float scale)`
//!   — `buffer` must be a *direct* ByteBuffer holding `numFrames` native-endian
//!   f32 samples. `numFrames` must equal `nativeFrameLength`. Samples are
//!   divided by `scale` before inference and multiplied by `scale` afterwards
//!   (WebRTC APM passes int16-range floats with scale 32768). Returns the LSNR
//!   estimate, or NaN on any error, in which case the buffer is left untouched.
//! - `void nativeFree(long handle)` — destroy the engine. Ignores 0.
//!
//! Threading: a handle must only be used by one thread at a time (the WebRTC
//! capture thread). The engine holds no thread-affine state, so the calling
//! thread may change between invocations.

use std::panic::{catch_unwind, AssertUnwindSafe};

use anyhow::{ensure, Context, Result};
use df::tract::{DfParams, DfTract, RuntimeParams};
use jni::objects::{JByteBuffer, JClass};
use jni::sys::{jfloat, jint, jlong};
use jni::JNIEnv;
use ndarray::Array2;

/// Single-channel DeepFilterNet3 engine plus preallocated frame buffers.
pub struct Engine {
    df: DfTract,
    buf_in: Array2<f32>,
    buf_out: Array2<f32>,
}

impl Engine {
    /// Create an engine using the embedded DeepFilterNet3 model.
    ///
    /// `atten_lim_db == 0` disables the attenuation limit (full suppression).
    pub fn new(atten_lim_db: f32) -> Result<Self> {
        let mut rp = RuntimeParams::default_with_ch(1);
        if atten_lim_db > 0.0 {
            rp = rp.with_atten_lim(atten_lim_db);
        }
        let df = DfTract::new(DfParams::default(), &rp).context("failed to init DfTract")?;
        let hop = df.hop_size;
        Ok(Self {
            df,
            buf_in: Array2::zeros((1, hop)),
            buf_out: Array2::zeros((1, hop)),
        })
    }

    /// Number of samples per frame (model hop size).
    pub fn frame_length(&self) -> usize {
        self.df.hop_size
    }

    /// Model sample rate in Hz.
    pub fn sample_rate(&self) -> usize {
        self.df.sr
    }

    /// Denoise one frame in place and return the LSNR estimate.
    ///
    /// `frame.len()` must equal [`Self::frame_length`]. Samples are divided by
    /// `scale` before inference and multiplied by `scale` on the way out. On
    /// error the frame is left untouched.
    pub fn process(&mut self, frame: &mut [f32], scale: f32) -> Result<f32> {
        ensure!(
            frame.len() == self.df.hop_size,
            "expected {} samples per frame, got {}",
            self.df.hop_size,
            frame.len()
        );
        ensure!(
            scale.is_finite() && scale != 0.0,
            "invalid scale {scale}"
        );
        let inv = 1.0 / scale;
        for (dst, &src) in self.buf_in.iter_mut().zip(frame.iter()) {
            *dst = src * inv;
        }
        let lsnr = self
            .df
            .process(self.buf_in.view(), self.buf_out.view_mut())
            .context("DfTract::process failed")?;
        for (dst, &src) in frame.iter_mut().zip(self.buf_out.iter()) {
            *dst = src * scale;
        }
        Ok(lsnr)
    }
}

/// Reborrow a JNI handle as an engine reference.
///
/// # Safety
/// `handle` must be 0 or a pointer obtained from `nativeNew` that has not been
/// passed to `nativeFree`, with no concurrent use of the same handle.
unsafe fn engine_mut<'a>(handle: jlong) -> Option<&'a mut Engine> {
    (handle as *mut Engine).as_mut()
}

#[no_mangle]
pub extern "system" fn Java_org_jitsi_meet_sdk_DeepFilterNet_nativeNew(
    _env: JNIEnv,
    _class: JClass,
    atten_lim_db: jfloat,
) -> jlong {
    catch_unwind(|| match Engine::new(atten_lim_db) {
        Ok(engine) => Box::into_raw(Box::new(engine)) as jlong,
        Err(_) => 0,
    })
    .unwrap_or(0)
}

#[no_mangle]
pub extern "system" fn Java_org_jitsi_meet_sdk_DeepFilterNet_nativeFrameLength(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) -> jint {
    catch_unwind(|| {
        unsafe { engine_mut(handle) }.map_or(0, |e| e.frame_length() as jint)
    })
    .unwrap_or(0)
}

#[no_mangle]
pub extern "system" fn Java_org_jitsi_meet_sdk_DeepFilterNet_nativeSampleRate(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) -> jint {
    catch_unwind(|| {
        unsafe { engine_mut(handle) }.map_or(0, |e| e.sample_rate() as jint)
    })
    .unwrap_or(0)
}

#[no_mangle]
pub extern "system" fn Java_org_jitsi_meet_sdk_DeepFilterNet_nativeProcessInPlace(
    env: JNIEnv,
    _class: JClass,
    handle: jlong,
    buffer: JByteBuffer,
    num_frames: jint,
    scale: jfloat,
) -> jfloat {
    catch_unwind(AssertUnwindSafe(|| {
        let engine = match unsafe { engine_mut(handle) } {
            Some(engine) => engine,
            None => return f32::NAN,
        };
        if num_frames < 0 || num_frames as usize != engine.frame_length() {
            return f32::NAN;
        }
        let n = num_frames as usize;
        let addr = match env.get_direct_buffer_address(&buffer) {
            Ok(addr) if !addr.is_null() => addr,
            _ => return f32::NAN,
        };
        let capacity = match env.get_direct_buffer_capacity(&buffer) {
            Ok(capacity) => capacity,
            Err(_) => return f32::NAN,
        };
        let byte_len = n * std::mem::size_of::<f32>();
        if capacity < byte_len || addr.align_offset(std::mem::align_of::<f32>()) != 0 {
            return f32::NAN;
        }
        // SAFETY: direct buffer address is valid for `capacity >= byte_len`
        // bytes, checked f32-aligned, and only this thread touches the handle.
        let samples = unsafe { std::slice::from_raw_parts_mut(addr.cast::<f32>(), n) };
        engine.process(samples, scale).unwrap_or(f32::NAN)
    }))
    .unwrap_or(f32::NAN)
}

#[no_mangle]
pub extern "system" fn Java_org_jitsi_meet_sdk_DeepFilterNet_nativeFree(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    if handle == 0 {
        return;
    }
    let _ = catch_unwind(|| {
        // SAFETY: handle originates from `nativeNew` (Java-side contract).
        drop(unsafe { Box::from_raw(handle as *mut Engine) });
    });
}

#[cfg(test)]
mod tests {
    use super::Engine;

    /// Deterministic white-noise source in [-1, 1) (LCG; avoids a rand dep).
    struct Lcg(u64);

    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32) / (1u64 << 23) as f32 - 1.0
        }
    }

    fn rms(energy: f64, samples: usize) -> f64 {
        (energy / samples as f64).sqrt()
    }

    const WARMUP_FRAMES: usize = 10;

    #[test]
    fn white_noise_is_attenuated() {
        let mut engine = Engine::new(0.0).expect("engine init");
        assert_eq!(engine.sample_rate(), 48000);
        assert_eq!(engine.frame_length(), 480);

        let hop = engine.frame_length();
        let n_frames = 2 * engine.sample_rate() / hop; // 2 s
        let mut rng = Lcg(0x1234_5678_9abc_def0);
        let (mut in_energy, mut out_energy, mut samples) = (0.0f64, 0.0f64, 0usize);

        for i in 0..n_frames {
            let mut frame: Vec<f32> = (0..hop).map(|_| rng.next_f32() * 0.1).collect();
            let in_frame = frame.clone();
            let lsnr = engine.process(&mut frame, 1.0).expect("process");
            assert!(lsnr.is_finite(), "LSNR must be finite, got {lsnr}");
            if i >= WARMUP_FRAMES {
                in_energy += in_frame.iter().map(|&x| (x as f64).powi(2)).sum::<f64>();
                out_energy += frame.iter().map(|&x| (x as f64).powi(2)).sum::<f64>();
                samples += hop;
            }
        }

        let drop_db = 20.0 * (rms(in_energy, samples) / rms(out_energy, samples)).log10();
        println!("white-noise RMS attenuation after warmup: {drop_db:.1} dB");
        assert!(
            drop_db >= 10.0,
            "expected >= 10 dB noise attenuation, got {drop_db:.1} dB"
        );
    }

    #[test]
    fn int16_scale_path_matches_unit_scale() {
        const SCALE: f32 = 32768.0;
        // A 6 dB attenuation limit mixes >= half the noisy signal back in
        // (out = lim*noisy + (1-lim)*enh, lim = 10^(-6/20)), so the output is
        // never silent even when the LSNR gate mutes a frame. This exercises
        // the attenLimDb > 0 path and gives a non-trivial signal for the
        // scaled-vs-unit comparison below.
        let mut engine_scaled = Engine::new(6.0).expect("engine init");
        let mut engine_unit = Engine::new(6.0).expect("engine init");

        let hop = engine_scaled.frame_length();
        let sr = engine_scaled.sample_rate() as f32;
        let n_frames = engine_scaled.sample_rate() / hop; // 1 s
        let mut rng = Lcg(0xdead_beef_cafe_f00d);
        let (mut in_energy, mut out_energy, mut samples) = (0.0f64, 0.0f64, 0usize);

        for i in 0..n_frames {
            // 440 Hz tone + white noise at int16-range amplitudes.
            let mut frame: Vec<f32> = (0..hop)
                .map(|j| {
                    let t = ((i * hop + j) as f32) / sr;
                    12000.0 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                        + 300.0 * rng.next_f32()
                })
                .collect();
            let in_frame = frame.clone();
            let mut frame_unit: Vec<f32> = in_frame.iter().map(|&x| x / SCALE).collect();

            let lsnr_scaled = engine_scaled.process(&mut frame, SCALE).expect("process");
            let lsnr_unit = engine_unit.process(&mut frame_unit, 1.0).expect("process");

            assert!(lsnr_scaled.is_finite());
            assert!(
                (lsnr_scaled - lsnr_unit).abs() <= 1e-3,
                "LSNR mismatch at frame {i}: {lsnr_scaled} vs {lsnr_unit}"
            );
            for (j, (&a, &b)) in frame.iter().zip(frame_unit.iter()).enumerate() {
                assert!(
                    (a - b * SCALE).abs() <= 1e-2,
                    "sample mismatch at frame {i}, sample {j}: {a} vs {}",
                    b * SCALE
                );
                assert!(a.is_finite());
            }

            if i >= WARMUP_FRAMES {
                in_energy += in_frame.iter().map(|&x| (x as f64).powi(2)).sum::<f64>();
                out_energy += frame.iter().map(|&x| (x as f64).powi(2)).sum::<f64>();
                samples += hop;
            }
        }

        // With the 6 dB attenuation limit the output RMS can drop at most
        // ~6 dB below the input, and never rise above it — proving the
        // pipeline emits real (non-muted, non-amplified) audio.
        let rms_in = rms(in_energy, samples);
        let rms_out = rms(out_energy, samples);
        assert!(rms_out > 0.0, "output is silent");
        let change_db = 20.0 * (rms_in / rms_out).log10();
        println!("tone+noise RMS change after warmup: {change_db:.1} dB");
        assert!(
            (-1.0..=7.0).contains(&change_db),
            "expected RMS change within [-1, 7] dB, got {change_db:.1} dB"
        );
    }

    #[test]
    fn wrong_frame_length_is_rejected() {
        let mut engine = Engine::new(0.0).expect("engine init");
        let hop = engine.frame_length();
        let mut short = vec![0.0f32; hop - 1];
        assert!(engine.process(&mut short, 1.0).is_err());
        let mut zero_scale = vec![0.0f32; hop];
        assert!(engine.process(&mut zero_scale, 0.0).is_err());
    }
}
