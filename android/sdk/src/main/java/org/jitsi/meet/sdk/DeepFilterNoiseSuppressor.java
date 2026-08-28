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
import java.util.concurrent.atomic.AtomicLong;

import org.jitsi.meet.sdk.log.JitsiMeetLogger;
import org.webrtc.ExternalAudioProcessingFactory;

/**
 * WebRTC APM capture post-processor running DeepFilterNet3 noise suppression
 * on the microphone signal, after the built-in echo cancellation and before
 * encoding.
 *
 * The webrtc-sdk external processing hook delivers 10 ms fullband frames as
 * float32 samples in the int16 value range. DFN3 operates on 480-sample
 * frames at 48 kHz, which is exactly what the APM produces on virtually all
 * Android devices; any other rate is passed through untouched.
 *
 * The engine is created on a background thread at construction time because
 * model loading takes hundreds of milliseconds, while
 * {@link #initialize(int, int)}, {@link #reset(int)} and
 * {@link #process(int, int, ByteBuffer)} run on the audio capture thread and
 * must not block. The handle is handed over through an {@link AtomicLong},
 * which provides the necessary happens-before edge; after the handover the
 * engine is used exclusively by the capture thread.
 */
class DeepFilterNoiseSuppressor implements ExternalAudioProcessingFactory.AudioProcessing {
    private static final String TAG = DeepFilterNoiseSuppressor.class.getSimpleName();

    /**
     * WebRTC AudioBuffer float samples use the int16 range, the model
     * expects [-1, 1].
     */
    private static final float WEBRTC_FLOAT_SCALE = 32768f;

    /**
     * Maximum noise attenuation in dB. 100 dB is effectively unlimited while
     * keeping the post-filter numerically sane.
     */
    private static final float ATTEN_LIM_DB = 100f;

    /** Engine handle created by the loader thread, not yet claimed. */
    private final AtomicLong pendingHandle = new AtomicLong();

    /** Engine handle owned by the audio capture thread. */
    private long handle;
    private int engineFrameLength;
    private int engineSampleRate;
    private int streamSampleRate;
    private boolean warnedPassthrough;

    DeepFilterNoiseSuppressor() {
        Thread loader = new Thread(() -> {
            long h = DeepFilterNet.nativeNew(ATTEN_LIM_DB);
            if (h == 0) {
                JitsiMeetLogger.e(TAG + " failed to create DeepFilterNet engine");
            } else {
                JitsiMeetLogger.i(
                    TAG + " engine ready: frameLength=" + DeepFilterNet.nativeFrameLength(h)
                        + " sampleRate=" + DeepFilterNet.nativeSampleRate(h));
                pendingHandle.set(h);
            }
        }, "DeepFilterNet-loader");
        loader.setDaemon(true);
        loader.start();
    }

    /**
     * Claims the engine created by the loader thread, if any. Only ever
     * called on the audio capture thread.
     */
    private boolean ensureEngine() {
        if (handle == 0) {
            long h = pendingHandle.getAndSet(0);
            if (h != 0) {
                handle = h;
                engineFrameLength = DeepFilterNet.nativeFrameLength(h);
                engineSampleRate = DeepFilterNet.nativeSampleRate(h);
            }
        }
        return handle != 0;
    }

    @Override
    public void initialize(int sampleRateHz, int numChannels) {
        streamSampleRate = sampleRateHz;
        ensureEngine();
        JitsiMeetLogger.i(TAG + " initialized: sampleRate=" + sampleRateHz + " channels=" + numChannels);
    }

    @Override
    public void reset(int newRate) {
        streamSampleRate = newRate;
    }

    @Override
    public void process(int numBands, int numFrames, ByteBuffer buffer) {
        if (!ensureEngine()) {
            return;
        }

        // The buffer holds `numFrames` fullband float32 samples = 10 ms.
        if (streamSampleRate != engineSampleRate
                || numFrames != engineFrameLength
                || !buffer.isDirect()
                || buffer.capacity() < numFrames * 4) {
            if (!warnedPassthrough) {
                warnedPassthrough = true;
                JitsiMeetLogger.w(
                    TAG + " passthrough: rate=" + streamSampleRate + " frames=" + numFrames
                        + " direct=" + buffer.isDirect()
                        + " (engine wants " + engineSampleRate + "/" + engineFrameLength + ")");
            }
            return;
        }

        DeepFilterNet.nativeProcessInPlace(handle, buffer, numFrames, WEBRTC_FLOAT_SCALE);
    }
}
