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

import android.content.Context;
import android.content.res.AssetFileDescriptor;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.concurrent.atomic.AtomicLong;

import org.jitsi.meet.sdk.log.JitsiMeetLogger;
import org.webrtc.ExternalAudioProcessingFactory;

/**
 * WebRTC APM capture post-processor running DeepFilterNet noise suppression
 * on the microphone signal, after the built-in echo cancellation and before
 * encoding.
 *
 * The webrtc-sdk external processing hook delivers 10 ms fullband frames as
 * float32 samples in the int16 value range. The engine operates at 48 kHz —
 * the APM rate on virtually all Android hardware; 16 and 32 kHz streams
 * (e.g. Bluetooth HFP headsets) are resampled inside the engine. Any other
 * rate is passed through untouched.
 *
 * Engine lifecycle: engines are created on background threads because model
 * loading takes hundreds of milliseconds, while the audio callbacks run on
 * the capture thread and must not block. New engines are handed over through
 * {@code pendingHandle} (also used for config-driven hot swaps mid-call);
 * the capture thread parks replaced engines in {@code retiredHandle}, which
 * the next {@link #applyConfig} reaps. At most one dormant engine exists at
 * a time. The {@link AtomicLong} exchanges provide the happens-before edges;
 * outside them each engine is used by one thread only.
 */
class DeepFilterNoiseSuppressor implements ExternalAudioProcessingFactory.AudioProcessing {
    private static final String TAG = DeepFilterNoiseSuppressor.class.getSimpleName();

    /**
     * WebRTC AudioBuffer float samples use the int16 range, the model
     * expects [-1, 1].
     */
    private static final float WEBRTC_FLOAT_SCALE = 32768f;

    /** Engine created by a loader thread, not yet claimed. */
    private final AtomicLong pendingHandle = new AtomicLong();

    /** Engine replaced by a hot swap, waiting to be freed. */
    private final AtomicLong retiredHandle = new AtomicLong();

    /** Engine handle owned by the audio capture thread. */
    private long handle;
    private int streamSampleRate;
    private boolean warnedPassthrough;
    private boolean loggedEngaged;

    private final Context context;

    DeepFilterNoiseSuppressor(Context context, NoiseSuppressionConfig config) {
        this.context = context.getApplicationContext();
        createEngineAsync(config);
    }

    /**
     * Builds an engine for {@code config} on a background thread and
     * publishes it for the capture thread, replacing the current engine.
     * Also frees any engine retired by a previous swap.
     */
    void applyConfig(NoiseSuppressionConfig config) {
        createEngineAsync(config);
    }

    private void createEngineAsync(NoiseSuppressionConfig config) {
        NoiseSuppressionConfig c = config.copy();
        Thread loader = new Thread(() -> {
            String modelPath;
            try {
                modelPath = ensureModelFile(context, c.model);
            } catch (IOException e) {
                JitsiMeetLogger.e(e, TAG + " failed to extract model asset: " + c.model.assetName);
                return;
            }
            long h = DeepFilterNet.nativeNew(
                modelPath, c.attenLimDb, c.minDbThresh, c.maxDbErbThresh,
                c.maxDbDfThresh, c.postFilter, 0f);
            if (h == 0) {
                JitsiMeetLogger.e(TAG + " failed to create engine: model=" + c.model.key);
                return;
            }
            JitsiMeetLogger.i(
                TAG + " engine ready: model=" + c.model.key
                    + " attenLim=" + c.attenLimDb + " postFilter=" + c.postFilter
                    + " thresholds=" + c.minDbThresh + "/" + c.maxDbErbThresh + "/" + c.maxDbDfThresh);

            // Replace any unclaimed previous engine directly...
            long unclaimed = pendingHandle.getAndSet(h);
            if (unclaimed != 0) {
                DeepFilterNet.nativeFree(unclaimed);
            }
            // ...and reap an engine retired by the capture thread. Give the
            // capture thread a moment to swap if a call is active, so the
            // old engine does not linger.
            for (int i = 0; i < 50; i++) {
                long retired = retiredHandle.getAndSet(0);
                if (retired != 0) {
                    DeepFilterNet.nativeFree(retired);
                    break;
                }
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }, "DeepFilterNet-loader");
        loader.setDaemon(true);
        loader.start();
    }

    /**
     * Extracts the model archive from the APK assets into the cache dir,
     * once. Model archives are stored uncompressed in the APK, so the
     * declared asset length is comparable to the file on disk; a mismatch
     * (e.g. after an app update changed the model) triggers re-extraction.
     */
    private static String ensureModelFile(Context context, NoiseSuppressionConfig.Model model)
            throws IOException {
        File dir = new File(context.getCacheDir(), "noise-suppression");
        File file = new File(dir, model.assetName);
        String assetPath = "noise-suppression/" + model.assetName;

        long assetLength = -1;
        try (AssetFileDescriptor fd = context.getAssets().openFd(assetPath)) {
            assetLength = fd.getLength();
        } catch (IOException e) {
            // Compressed asset; fall back to existence check only.
        }

        if (file.isFile() && file.length() > 0
                && (assetLength == -1 || file.length() == assetLength)) {
            return file.getAbsolutePath();
        }

        if (!dir.isDirectory() && !dir.mkdirs()) {
            throw new IOException("cannot create " + dir);
        }
        File tmp = new File(dir, model.assetName + ".tmp");
        try (InputStream in = context.getAssets().open(assetPath);
                OutputStream out = new FileOutputStream(tmp)) {
            byte[] buf = new byte[64 * 1024];
            int n;
            while ((n = in.read(buf)) > 0) {
                out.write(buf, 0, n);
            }
        }
        if (!tmp.renameTo(file)) {
            tmp.delete();
            throw new IOException("cannot move " + tmp + " to " + file);
        }
        return file.getAbsolutePath();
    }

    /**
     * Claims a pending engine, retiring the current one. Only ever called on
     * the audio capture thread.
     */
    private boolean ensureEngine() {
        long p = pendingHandle.getAndSet(0);
        if (p != 0) {
            if (handle != 0) {
                long prevRetired = retiredHandle.getAndSet(handle);
                if (prevRetired != 0) {
                    // Two swaps before a reap; should not happen, but never
                    // leak: freeing here is a one-off cost on config change.
                    DeepFilterNet.nativeFree(prevRetired);
                }
            }
            handle = p;
            loggedEngaged = false;
            warnedPassthrough = false;
        }
        return handle != 0;
    }

    @Override
    public void initialize(int sampleRateHz, int numChannels) {
        streamSampleRate = sampleRateHz;
        warnedPassthrough = false;
        loggedEngaged = false;
        ensureEngine();
        JitsiMeetLogger.i(TAG + " initialized: sampleRate=" + sampleRateHz + " channels=" + numChannels);
    }

    @Override
    public void reset(int newRate) {
        streamSampleRate = newRate;
        warnedPassthrough = false;
        loggedEngaged = false;
    }

    @Override
    public void process(int numBands, int numFrames, ByteBuffer buffer) {
        if (!ensureEngine()) {
            return;
        }

        // The buffer holds `numFrames` fullband float32 samples = 10 ms.
        // The engine handles 48 kHz natively and resamples 16/32 kHz streams;
        // anything else is left untouched (it returns NaN).
        float lsnr = Float.NaN;
        if (buffer.isDirect() && buffer.capacity() >= numFrames * 4) {
            lsnr = DeepFilterNet.nativeProcessInPlace(
                handle, buffer, numFrames, streamSampleRate, WEBRTC_FLOAT_SCALE);
        }

        if (Float.isNaN(lsnr)) {
            if (!warnedPassthrough) {
                warnedPassthrough = true;
                JitsiMeetLogger.w(
                    TAG + " passthrough: rate=" + streamSampleRate + " frames=" + numFrames
                        + " direct=" + buffer.isDirect());
            }
        } else if (!loggedEngaged) {
            loggedEngaged = true;
            JitsiMeetLogger.i(TAG + " engaged: rate=" + streamSampleRate + " frames=" + numFrames);
        }
    }
}
