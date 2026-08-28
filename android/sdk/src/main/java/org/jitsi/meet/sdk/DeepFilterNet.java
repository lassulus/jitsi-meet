/*
 * Copyright @ 2026-present 8x8, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.jitsi.meet.sdk;

import java.nio.ByteBuffer;

import org.jitsi.meet.sdk.log.JitsiMeetLogger;

/**
 * JNI binding for the DeepFilterNet3 noise suppression engine
 * (android/noise-suppression, a Rust cdylib embedding the DFN3 model and the
 * tract inference runtime).
 *
 * All methods operate on an opaque native handle. A handle must only be used
 * from one thread at a time; WebRTC guarantees this on the APM capture thread.
 */
final class DeepFilterNet {
    private static final boolean AVAILABLE;

    static {
        boolean available;
        try {
            System.loadLibrary("dfjni");
            available = true;
        } catch (UnsatisfiedLinkError e) {
            JitsiMeetLogger.w(e, "DeepFilterNet native library not available");
            available = false;
        }
        AVAILABLE = available;
    }

    private DeepFilterNet() {
    }

    /**
     * Whether libdfjni.so was loaded successfully.
     */
    static boolean isAvailable() {
        return AVAILABLE;
    }

    /**
     * Creates a single-channel DeepFilterNet3 engine.
     *
     * @param attenLimDb maximum noise attenuation in dB; 0 means unlimited.
     * @return native handle, or 0 on failure.
     */
    static native long nativeNew(float attenLimDb);

    /** Samples per processing frame (480 for the 48 kHz DFN3 model). */
    static native int nativeFrameLength(long handle);

    /** Sample rate the model operates at (48000 for DFN3). */
    static native int nativeSampleRate(long handle);

    /**
     * Denoises one frame in place.
     *
     * @param handle native engine handle.
     * @param buffer direct {@link ByteBuffer} over {@code numFrames}
     * native-endian float32 samples.
     * @param numFrames number of samples; must equal
     * {@link #nativeFrameLength(long)}.
     * @param scale full-scale value of the input samples (the model expects
     * [-1, 1]; WebRTC AudioBuffer floats use the int16 range, so pass 32768).
     * @return the local SNR estimate, or NaN if the frame was left untouched.
     */
    static native float nativeProcessInPlace(long handle, ByteBuffer buffer, int numFrames, float scale);

    /** Releases the engine. Passing 0 is a no-op. */
    static native void nativeFree(long handle);
}
