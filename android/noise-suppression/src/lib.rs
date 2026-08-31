//! DeepFilterNet noise suppression engine exposed over JNI.
//!
//! Java-side contract (`org.jitsi.meet.sdk.DeepFilterNet`, loaded via
//! `System.loadLibrary("dfjni")`):
//!
//! - `long nativeNew(String modelPath, float attenLimDb, float minDbThresh, float maxDbErbThresh, float maxDbDfThresh, boolean postFilter, float postFilterBeta)`
//!   — create a single-channel engine from the DeepFilterNet model archive
//!   (tar.gz, as shipped in the SDK assets) at filesystem path `modelPath`.
//!   Only DeepFilterNet3-family models work: the pinned deep_filter v0.5.6
//!   deprecates DeepFilterNet2 models and fails to load them.
//!   `attenLimDb == 0` or `>= 100` means no attenuation limit
//!   (full suppression), otherwise the enhanced signal is mixed with the
//!   noisy input so noise is reduced by at most that many dB.
//!   `minDbThresh` / `maxDbErbThresh` / `maxDbDfThresh` are the LSNR stage
//!   thresholds (libDF defaults -10 / 30 / 20 dB): below `minDbThresh` the
//!   output is muted, above `maxDbErbThresh` the signal passes unprocessed,
//!   above `maxDbDfThresh` only the ERB stage runs. `postFilter` enables the
//!   DFN post filter for extra attenuation of noisy bins; `postFilterBeta <=
//!   0` keeps the library's default beta. Returns a handle, or 0 on any
//!   failure (unreadable, unparseable or deprecated model). Never throws.
//! - `int nativeFrameLength(long handle)` — model hop size in samples
//!   (480 at 48 kHz). 0 for an invalid handle.
//! - `int nativeSampleRate(long handle)` — model sample rate (48000).
//!   0 for an invalid handle.
//! - `float nativeProcessInPlace(long handle, ByteBuffer buffer, int numFrames, int sampleRateHz, float scale)`
//!   — `buffer` must be a *direct* ByteBuffer holding `numFrames` native-endian
//!   f32 samples. `sampleRateHz` must be 16000, 32000 or 48000 and `numFrames`
//!   must equal `sampleRateHz / 100` (one 10 ms frame). 48 kHz frames are fed
//!   to the model directly; 16/32 kHz frames are upsampled to the model's
//!   48 kHz, denoised, and downsampled back in place (the fixed-ratio FFT
//!   resamplers add a few milliseconds of latency, so the first frames after
//!   a rate switch are partially silent). Samples are divided by `scale`
//!   before inference and multiplied by `scale` afterwards (WebRTC APM passes
//!   int16-range floats with scale 32768). Returns the LSNR estimate, or NaN
//!   on any error, in which case the buffer is left untouched.
//! - `void nativeFree(long handle)` — destroy the engine. Ignores 0.
//!
//! Threading: a handle must only be used by one thread at a time (the WebRTC
//! capture thread). The engine holds no thread-affine state, so the calling
//! thread may change between invocations.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;

use anyhow::{ensure, Context, Result};
use df::tract::{DfParams, DfTract, RuntimeParams};
use jni::objects::{JByteBuffer, JClass, JString};
use jni::sys::{jboolean, jfloat, jint, jlong};
use jni::JNIEnv;
use ndarray::Array2;
use rubato::{FftFixedInOut, Resampler};

/// Runtime tuning for [`Engine::new`], mirroring the JNI `nativeNew`
/// contract one to one (minus the model path).
#[derive(Clone, Copy, Debug)]
pub struct EngineConfig {
    /// Attenuation limit in dB; 0 (or >= 100) means unlimited.
    pub atten_lim_db: f32,
    /// LSNR below which the output is muted entirely (libDF default -10).
    pub min_db_thresh: f32,
    /// LSNR above which the signal passes unprocessed (libDF default 30).
    pub max_db_erb_thresh: f32,
    /// LSNR above which only the ERB stage runs, skipping the deep-filter
    /// stage (libDF default 20).
    pub max_db_df_thresh: f32,
    /// Enable the DFN post filter (extra attenuation of noisy bins).
    pub post_filter: bool,
    /// Post-filter beta; <= 0 keeps the library default.
    pub post_filter_beta: f32,
}

impl Default for EngineConfig {
    /// The libDF runtime defaults with no attenuation limit — the previous
    /// hardcoded behavior of this crate.
    fn default() -> Self {
        Self {
            atten_lim_db: 0.0,
            min_db_thresh: -10.0,
            max_db_erb_thresh: 30.0,
            max_db_df_thresh: 20.0,
            post_filter: false,
            post_filter_beta: 0.0,
        }
    }
}

/// Fixed-ratio FFT resamplers bridging one stream rate to the model rate,
/// plus the scratch frames they read/write. State is kept across calls and
/// only rebuilt when the stream rate changes.
struct Resamplers {
    /// Stream sample rate in Hz (16000 or 32000).
    rate: usize,
    up: FftFixedInOut<f32>,
    down: FftFixedInOut<f32>,
    /// One model-rate frame: upsampler output, downsampler input.
    mid: Vec<f32>,
    /// One stream-rate frame: downsampler output.
    out: Vec<f32>,
}

impl Resamplers {
    fn new(rate: usize, model_sr: usize, hop: usize) -> Result<Self> {
        let n = rate / 100;
        // rubato 0.14 sizes FftFixedInOut chunks in units of
        // `sample_rate_output / gcd`, so the `chunk_size_in` argument must be
        // given in *output* frames to land on exact 10 ms chunks. The
        // assertions below turn any deviation into a hard error instead of
        // corrupt audio.
        let up = FftFixedInOut::<f32>::new(rate, model_sr, hop, 1)
            .context("failed to create upsampler")?;
        ensure!(
            up.input_frames_next() == n && up.output_frames_next() == hop,
            "unexpected upsampler chunk sizes {}->{}, want {n}->{hop}",
            up.input_frames_next(),
            up.output_frames_next(),
        );
        let down = FftFixedInOut::<f32>::new(model_sr, rate, n, 1)
            .context("failed to create downsampler")?;
        ensure!(
            down.input_frames_next() == hop && down.output_frames_next() == n,
            "unexpected downsampler chunk sizes {}->{}, want {hop}->{n}",
            down.input_frames_next(),
            down.output_frames_next(),
        );
        Ok(Self {
            rate,
            up,
            down,
            mid: vec![0.0; hop],
            out: vec![0.0; n],
        })
    }
}

/// Denoise one model-rate frame in place and return the LSNR estimate.
///
/// Free function over `Engine`'s fields so the resampled path can borrow the
/// resampler scratch buffer as `frame` at the same time. On error the frame
/// is left untouched.
fn process_model_frame(
    df: &mut DfTract,
    buf_in: &mut Array2<f32>,
    buf_out: &mut Array2<f32>,
    frame: &mut [f32],
    scale: f32,
) -> Result<f32> {
    let inv = 1.0 / scale;
    for (dst, &src) in buf_in.iter_mut().zip(frame.iter()) {
        *dst = src * inv;
    }
    let lsnr = df
        .process(buf_in.view(), buf_out.view_mut())
        .context("DfTract::process failed")?;
    for (dst, &src) in frame.iter_mut().zip(buf_out.iter()) {
        *dst = src * scale;
    }
    Ok(lsnr)
}

/// Single-channel DeepFilterNet engine plus preallocated frame buffers and
/// lazily created 16/32 kHz resamplers.
pub struct Engine {
    df: DfTract,
    buf_in: Array2<f32>,
    buf_out: Array2<f32>,
    resamplers: Option<Resamplers>,
}

impl Engine {
    /// Create an engine from the DeepFilterNet model archive (tar.gz) at
    /// `model_path`, tuned per `config`.
    pub fn new(model_path: &Path, config: EngineConfig) -> Result<Self> {
        let dfp = DfParams::new(model_path.to_path_buf())
            .with_context(|| format!("failed to load model archive {}", model_path.display()))?;
        let mut rp = RuntimeParams::default_with_ch(1).with_thresholds(
            config.min_db_thresh,
            config.max_db_erb_thresh,
            config.max_db_df_thresh,
        );
        // libDF already treats limits >= 100 dB as "unlimited"; its default
        // (100) encodes exactly that, so only real limits are set explicitly.
        if config.atten_lim_db > 0.0 && config.atten_lim_db < 100.0 {
            rp = rp.with_atten_lim(config.atten_lim_db);
        }
        if config.post_filter {
            if config.post_filter_beta > 0.0 {
                rp = rp.with_post_filter(config.post_filter_beta);
            } else {
                // Only flip the switch, keeping the library's default beta.
                rp.post_filter = true;
            }
        }
        let df = DfTract::new(dfp, &rp).context("failed to init DfTract")?;
        // The 10 ms framing contract below relies on the model hop being
        // exactly 10 ms (480 samples at 48 kHz).
        ensure!(
            df.hop_size * 100 == df.sr,
            "model hop {} is not 10 ms at {} Hz",
            df.hop_size,
            df.sr
        );
        let hop = df.hop_size;
        Ok(Self {
            df,
            buf_in: Array2::zeros((1, hop)),
            buf_out: Array2::zeros((1, hop)),
            resamplers: None,
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

    /// Denoise one 10 ms frame in place and return the LSNR estimate.
    ///
    /// `sample_rate_hz` must be 16000, 32000 or 48000 and `frame.len()` must
    /// equal `sample_rate_hz / 100`. 48 kHz frames go to the model directly;
    /// 16/32 kHz frames are upsampled to 48 kHz, denoised, and downsampled
    /// back. Samples are divided by `scale` before inference and multiplied
    /// by `scale` on the way out. On error the frame is left untouched.
    pub fn process(&mut self, frame: &mut [f32], sample_rate_hz: usize, scale: f32) -> Result<f32> {
        ensure!(
            scale.is_finite() && scale != 0.0,
            "invalid scale {scale}"
        );
        ensure!(
            matches!(sample_rate_hz, 16_000 | 32_000 | 48_000),
            "unsupported sample rate {sample_rate_hz} Hz"
        );
        let n = sample_rate_hz / 100;
        ensure!(
            frame.len() == n,
            "expected {n} samples per 10 ms frame at {sample_rate_hz} Hz, got {}",
            frame.len()
        );
        if sample_rate_hz == self.df.sr {
            return process_model_frame(
                &mut self.df,
                &mut self.buf_in,
                &mut self.buf_out,
                frame,
                scale,
            );
        }
        if self
            .resamplers
            .as_ref()
            .map_or(true, |r| r.rate != sample_rate_hz)
        {
            self.resamplers = Some(Resamplers::new(
                sample_rate_hz,
                self.df.sr,
                self.df.hop_size,
            )?);
        }
        let rs = self.resamplers.as_mut().expect("resamplers just created");
        rs.up
            .process_into_buffer(&[&frame[..]], &mut [&mut rs.mid[..]], None)
            .context("upsampling failed")?;
        let lsnr = process_model_frame(
            &mut self.df,
            &mut self.buf_in,
            &mut self.buf_out,
            &mut rs.mid,
            scale,
        )?;
        rs.down
            .process_into_buffer(&[&rs.mid[..]], &mut [&mut rs.out[..]], None)
            .context("downsampling failed")?;
        frame.copy_from_slice(&rs.out);
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
    mut env: JNIEnv,
    _class: JClass,
    model_path: JString,
    atten_lim_db: jfloat,
    min_db_thresh: jfloat,
    max_db_erb_thresh: jfloat,
    max_db_df_thresh: jfloat,
    post_filter: jboolean,
    post_filter_beta: jfloat,
) -> jlong {
    let config = EngineConfig {
        atten_lim_db,
        min_db_thresh,
        max_db_erb_thresh,
        max_db_df_thresh,
        post_filter: post_filter != 0,
        post_filter_beta,
    };
    catch_unwind(AssertUnwindSafe(|| {
        let path: String = match env.get_string(&model_path) {
            Ok(s) => s.into(),
            Err(_) => return 0,
        };
        match Engine::new(Path::new(&path), config) {
            Ok(engine) => Box::into_raw(Box::new(engine)) as jlong,
            Err(_) => 0,
        }
    }))
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
    sample_rate_hz: jint,
    scale: jfloat,
) -> jfloat {
    catch_unwind(AssertUnwindSafe(|| {
        let engine = match unsafe { engine_mut(handle) } {
            Some(engine) => engine,
            None => return f32::NAN,
        };
        if num_frames <= 0 || sample_rate_hz <= 0 {
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
        engine
            .process(samples, sample_rate_hz as usize, scale)
            .unwrap_or(f32::NAN)
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
    use super::{Engine, EngineConfig};
    use std::path::{Path, PathBuf};

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

    /// Model archives ship as Android SDK assets; tests load them from there.
    const DFN3: &str = "DeepFilterNet3_onnx.tgz";
    const DFN3_LL: &str = "DeepFilterNet3_ll_onnx.tgz";

    fn model_path(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../sdk/src/main/assets/noise-suppression")
            .join(name)
    }

    /// DFN3 defaults with the given attenuation limit (the old constructor).
    fn cfg(atten_lim_db: f32) -> EngineConfig {
        EngineConfig {
            atten_lim_db,
            ..EngineConfig::default()
        }
    }

    /// DeepFilterNet3 engine with the given attenuation limit.
    fn dfn3(atten_lim_db: f32) -> Engine {
        Engine::new(&model_path(DFN3), cfg(atten_lim_db)).expect("engine init")
    }

    /// Feed 2 s of deterministic white noise at 48 kHz and return the
    /// post-warmup RMS attenuation in dB.
    fn white_noise_drop_db(engine: &mut Engine) -> f64 {
        let hop = engine.frame_length();
        let n_frames = 2 * engine.sample_rate() / hop; // 2 s
        let mut rng = Lcg(0x1234_5678_9abc_def0);
        let (mut in_energy, mut out_energy, mut samples) = (0.0f64, 0.0f64, 0usize);

        for i in 0..n_frames {
            let mut frame: Vec<f32> = (0..hop).map(|_| rng.next_f32() * 0.1).collect();
            let in_frame = frame.clone();
            let lsnr = engine.process(&mut frame, 48_000, 1.0).expect("process");
            assert!(lsnr.is_finite(), "LSNR must be finite, got {lsnr}");
            if i >= WARMUP_FRAMES {
                in_energy += in_frame.iter().map(|&x| (x as f64).powi(2)).sum::<f64>();
                out_energy += frame.iter().map(|&x| (x as f64).powi(2)).sum::<f64>();
                samples += hop;
            }
        }

        20.0 * (rms(in_energy, samples) / rms(out_energy, samples)).log10()
    }

    #[test]
    fn white_noise_is_attenuated() {
        let mut engine = dfn3(0.0);
        assert_eq!(engine.sample_rate(), 48000);
        assert_eq!(engine.frame_length(), 480);

        let drop_db = white_noise_drop_db(&mut engine);
        println!("white-noise RMS attenuation after warmup: {drop_db:.1} dB");
        assert!(
            drop_db >= 10.0,
            "expected >= 10 dB noise attenuation, got {drop_db:.1} dB"
        );
    }

    #[test]
    fn all_shipped_models_attenuate_white_noise() {
        for (file, name) in [(DFN3, "DeepFilterNet3"), (DFN3_LL, "DeepFilterNet3-LL")] {
            let mut engine = Engine::new(&model_path(file), EngineConfig::default())
                .unwrap_or_else(|e| panic!("{name} init failed: {e}"));
            assert_eq!(engine.sample_rate(), 48000, "{name} sample rate");
            assert_eq!(engine.frame_length(), 480, "{name} frame length");

            let drop_db = white_noise_drop_db(&mut engine);
            println!("{name} white-noise RMS attenuation after warmup: {drop_db:.1} dB");
            assert!(
                drop_db >= 10.0,
                "{name}: expected >= 10 dB noise attenuation, got {drop_db:.1} dB"
            );
        }
    }

    #[test]
    fn invalid_model_is_rejected() {
        // Nonexistent path.
        assert!(
            Engine::new(Path::new("/nonexistent/model.tar.gz"), EngineConfig::default()).is_err(),
            "nonexistent model path must be rejected"
        );
        // An existing file that is not a model archive.
        let not_a_model = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        assert!(
            Engine::new(&not_a_model, EngineConfig::default()).is_err(),
            "non-model file must be rejected"
        );
    }

    #[test]
    fn post_filter_engine_processes() {
        // postFilterBeta <= 0 keeps the library default; also try an explicit
        // beta. Both must construct and yield finite, non-NaN audio.
        for beta in [0.0f32, 0.05] {
            let mut engine = Engine::new(
                &model_path(DFN3),
                EngineConfig {
                    post_filter: true,
                    post_filter_beta: beta,
                    ..EngineConfig::default()
                },
            )
            .expect("post-filter engine init");

            let hop = engine.frame_length();
            let sr = engine.sample_rate() as f32;
            let mut rng = Lcg(0x5eed_5eed_5eed_5eed);
            for i in 0..50 {
                // 440 Hz tone + light noise, unit scale.
                let mut frame: Vec<f32> = (0..hop)
                    .map(|j| {
                        let t = ((i * hop + j) as f32) / sr;
                        0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                            + 0.01 * rng.next_f32()
                    })
                    .collect();
                let lsnr = engine.process(&mut frame, 48_000, 1.0).expect("process");
                assert!(lsnr.is_finite(), "beta {beta}: LSNR must be finite");
                assert!(
                    frame.iter().all(|x| x.is_finite()),
                    "beta {beta}: non-finite sample in frame {i}"
                );
            }
        }
    }

    #[test]
    fn int16_scale_path_matches_unit_scale() {
        const SCALE: f32 = 32768.0;
        // A 6 dB attenuation limit mixes >= half the noisy signal back in
        // (out = lim*noisy + (1-lim)*enh, lim = 10^(-6/20)), so the output is
        // never silent even when the LSNR gate mutes a frame. This exercises
        // the attenLimDb > 0 path and gives a non-trivial signal for the
        // scaled-vs-unit comparison below.
        let mut engine_scaled = dfn3(6.0);
        let mut engine_unit = dfn3(6.0);

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

            let lsnr_scaled = engine_scaled
                .process(&mut frame, 48_000, SCALE)
                .expect("process");
            let lsnr_unit = engine_unit
                .process(&mut frame_unit, 48_000, 1.0)
                .expect("process");

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
    fn white_noise_16k_is_attenuated() {
        const SCALE: f32 = 32768.0;
        let mut engine = dfn3(0.0);

        let n = 160; // 10 ms at 16 kHz
        let n_frames = 200; // 2 s
        let mut rng = Lcg(0x1234_5678_9abc_def0);
        let (mut in_energy, mut out_energy, mut samples) = (0.0f64, 0.0f64, 0usize);

        for i in 0..n_frames {
            // Int16-range white noise, as the WebRTC APM would hand over.
            let mut frame: Vec<f32> = (0..n).map(|_| rng.next_f32() * 3000.0).collect();
            let in_frame = frame.clone();
            let lsnr = engine.process(&mut frame, 16_000, SCALE).expect("process");
            assert!(lsnr.is_finite(), "LSNR must be finite, got {lsnr}");
            if i >= WARMUP_FRAMES {
                in_energy += in_frame.iter().map(|&x| (x as f64).powi(2)).sum::<f64>();
                out_energy += frame.iter().map(|&x| (x as f64).powi(2)).sum::<f64>();
                samples += n;
            }
        }

        let drop_db = 20.0 * (rms(in_energy, samples) / rms(out_energy, samples)).log10();
        println!("16 kHz white-noise RMS attenuation after warmup: {drop_db:.1} dB");
        assert!(
            drop_db >= 10.0,
            "expected >= 10 dB noise attenuation, got {drop_db:.1} dB"
        );
    }

    #[test]
    fn tone_16k_passes_with_atten_limit() {
        const SCALE: f32 = 32768.0;
        // Same rationale as `int16_scale_path_matches_unit_scale`: a 6 dB
        // attenuation limit guarantees a non-silent output, proving the
        // resampled path carries real audio end to end.
        let mut engine = dfn3(6.0);

        let n = 160; // 10 ms at 16 kHz
        let sr = 16_000.0f32;
        let n_frames = 100; // 1 s
        let mut rng = Lcg(0xdead_beef_cafe_f00d);
        let (mut in_energy, mut out_energy, mut samples) = (0.0f64, 0.0f64, 0usize);

        for i in 0..n_frames {
            // 440 Hz tone + white noise at int16-range amplitudes.
            let mut frame: Vec<f32> = (0..n)
                .map(|j| {
                    let t = ((i * n + j) as f32) / sr;
                    12000.0 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                        + 300.0 * rng.next_f32()
                })
                .collect();
            let in_frame = frame.clone();
            let lsnr = engine.process(&mut frame, 16_000, SCALE).expect("process");
            assert!(lsnr.is_finite());
            assert!(frame.iter().all(|x| x.is_finite()));

            if i >= WARMUP_FRAMES {
                in_energy += in_frame.iter().map(|&x| (x as f64).powi(2)).sum::<f64>();
                out_energy += frame.iter().map(|&x| (x as f64).powi(2)).sum::<f64>();
                samples += n;
            }
        }

        let rms_in = rms(in_energy, samples);
        let rms_out = rms(out_energy, samples);
        assert!(rms_out > 0.0, "output is silent");
        let change_db = 20.0 * (rms_in / rms_out).log10();
        println!("16 kHz tone+noise RMS change after warmup: {change_db:.1} dB");
        assert!(
            (-1.0..=7.0).contains(&change_db),
            "expected RMS change within [-1, 7] dB, got {change_db:.1} dB"
        );
    }

    #[test]
    fn unsupported_input_is_rejected_with_buffer_untouched() {
        let mut engine = dfn3(0.0);

        // Unsupported rate, even with a matching 10 ms frame.
        let mut frame_441 = vec![0.25f32; 441];
        let orig = frame_441.clone();
        assert!(engine.process(&mut frame_441, 44_100, 1.0).is_err());
        assert_eq!(frame_441, orig, "44.1 kHz reject must not touch the buffer");

        // numFrames mismatch for a supported rate.
        let mut wrong_len = vec![0.25f32; 480];
        let orig = wrong_len.clone();
        assert!(engine.process(&mut wrong_len, 16_000, 1.0).is_err());
        assert_eq!(wrong_len, orig, "length reject must not touch the buffer");

        let hop = engine.frame_length();
        let mut short = vec![0.0f32; hop - 1];
        assert!(engine.process(&mut short, 48_000, 1.0).is_err());
        let mut zero_scale = vec![0.0f32; hop];
        assert!(engine.process(&mut zero_scale, 48_000, 0.0).is_err());
    }

    #[test]
    fn rate_switch_mid_stream_works() {
        let mut engine = dfn3(0.0);
        let mut rng = Lcg(0x0bad_5eed_0bad_5eed);

        for &(rate, frames) in &[(48_000usize, 20usize), (16_000, 20), (48_000, 20), (32_000, 20)]
        {
            let n = rate / 100;
            for _ in 0..frames {
                let mut frame: Vec<f32> = (0..n).map(|_| rng.next_f32() * 0.1).collect();
                let lsnr = engine
                    .process(&mut frame, rate, 1.0)
                    .unwrap_or_else(|e| panic!("process at {rate} Hz failed: {e}"));
                assert!(lsnr.is_finite(), "LSNR must be finite at {rate} Hz");
                assert!(frame.iter().all(|x| x.is_finite()));
            }
        }
    }
}
