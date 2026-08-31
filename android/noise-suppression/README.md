# Deep-learning noise cancellation

This directory contains the native noise suppression engine used by the
Android SDK: a thin JNI wrapper (Rust, `cdylib`) around
[DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet), a modern
deep-learning speech enhancement model operating on fullband 48 kHz audio in
real time. The model weights are embedded in the library; inference runs on
the [tract](https://github.com/sonos/tract) runtime. No network access, no
extra assets.

## How it is wired up

- The stock `org.jitsi:webrtc` Maven artifact is substituted with the
  API-compatible [webrtc-sdk](https://github.com/webrtc-sdk/webrtc) build
  (see `android/build.gradle`), which adds
  `org.webrtc.ExternalAudioProcessingFactory` — a Java hook into the WebRTC
  audio processing module (APM).
- `patches/react-native-webrtc+*.patch` extends `WebRTCModuleOptions` with an
  `audioProcessingFactory` field that is passed to the
  `PeerConnectionFactory`.
- `org.jitsi.meet.sdk.NoiseSuppression` installs
  `org.jitsi.meet.sdk.DeepFilterNoiseSuppressor` as APM *capture
  post-processing*: it runs on every 10 ms microphone frame after echo
  cancellation and before encoding — for all conferences, with no
  renegotiation.
- Frames arrive as 10 ms float32 chunks and are denoised in place by
  `libdfjni.so`. 48 kHz (the APM rate on virtually all Android hardware)
  feeds the DFN3 hop size of 480 samples directly; 16 and 32 kHz streams
  (e.g. Bluetooth HFP headsets) are resampled to 48 kHz inside the engine.

Noise cancellation is **enabled by default** and configurable from the app's
Settings screen (Noise suppression section): model choice (DeepFilterNet 3 /
DeepFilterNet 3 LL), strength presets (Gentle / Balanced / Aggressive), a
maximum-attenuation slider and the post filter. Models ship as APK assets
(`android/sdk/src/main/assets/noise-suppression/*.tgz` — the `.tgz` name
avoids AAPT's gunzip-and-strip handling of `.gz` assets) and are extracted
to the cache dir on first use. Settings persist natively (SharedPreferences)
and hot-swap the engine mid-call. Hosting apps embedding the SDK can also
toggle programmatically:

```java
org.jitsi.meet.sdk.NoiseSuppression.setEnabled(false);
```

If the native library is missing for the current ABI or the audio pipeline
runs at an unsupported rate, audio passes through unprocessed.

## Building the native library

```sh
rustup target add aarch64-linux-android armv7-linux-androideabi \
    i686-linux-android x86_64-linux-android
cargo install cargo-ndk   # or use your distro package
export ANDROID_NDK_ROOT=/path/to/ndk/27.x
./build-android.sh
```

This drops `libdfjni.so` for every ABI into
`android/sdk/src/main/jniLibs/`, from where Gradle packages it into the SDK
AAR.

Host-side tests for the engine (model sanity, in-place processing contract):

```sh
cargo test --release
```
