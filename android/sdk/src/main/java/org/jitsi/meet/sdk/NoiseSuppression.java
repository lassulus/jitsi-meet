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

import com.oney.WebRTCModule.WebRTCModuleOptions;

import org.jitsi.meet.sdk.log.JitsiMeetLogger;
import org.webrtc.ExternalAudioProcessingFactory;

/**
 * Deep-learning noise cancellation (DeepFilterNet) for the microphone
 * capture path.
 *
 * The suppressor is installed as a WebRTC APM capture post-processor via the
 * webrtc-sdk {@link ExternalAudioProcessingFactory}, so it runs after echo
 * cancellation and before the audio encoder, for every conference. It is
 * enabled by default; model and aggressiveness are configurable at runtime
 * (see {@link NoiseSuppressionConfig}) with no renegotiation — engines are
 * swapped mid-call. Configuration persists across app restarts.
 */
public final class NoiseSuppression {
    private static final String TAG = NoiseSuppression.class.getSimpleName();

    private static Context appContext;
    private static ExternalAudioProcessingFactory factory;
    private static DeepFilterNoiseSuppressor suppressor;
    private static NoiseSuppressionConfig config = new NoiseSuppressionConfig();

    private NoiseSuppression() {
    }

    /**
     * Installs the noise suppressor into the WebRTC module options. Called
     * once before the React Native runtime (and thus WebRTC) is initialized.
     */
    static synchronized void install(Context context, WebRTCModuleOptions options) {
        if (factory != null || options.audioProcessingFactory != null) {
            return;
        }

        if (!DeepFilterNet.isAvailable()) {
            JitsiMeetLogger.w(TAG + " not installing: engine library missing");
            return;
        }

        try {
            // ExternalAudioProcessingFactory needs the WebRTC native library,
            // which is normally loaded later, when the WebRTC module
            // initializes. Loading is idempotent.
            System.loadLibrary("jingle_peerconnection_so");

            appContext = context.getApplicationContext();
            config = NoiseSuppressionConfig.load(appContext);
            suppressor = new DeepFilterNoiseSuppressor(appContext, config);
            factory = new ExternalAudioProcessingFactory();
            factory.setCapturePostProcessing(suppressor);
            factory.setBypassFlagForCapturePost(!config.enabled);
            options.audioProcessingFactory = factory;
            JitsiMeetLogger.i(
                TAG + " installed DeepFilterNet capture post-processor: model=" + config.model.key
                    + " enabled=" + config.enabled);
        } catch (UnsatisfiedLinkError | RuntimeException e) {
            JitsiMeetLogger.e(e, TAG + " failed to install");
            factory = null;
            suppressor = null;
        }
    }

    /** Whether the suppressor is installed and operational. */
    public static synchronized boolean isAvailable() {
        return factory != null;
    }

    /**
     * Enables or disables noise cancellation for the local microphone. Safe
     * to call at any time, including mid-conference. Persisted.
     */
    public static synchronized void setEnabled(boolean on) {
        config.enabled = on;
        if (factory != null) {
            factory.setBypassFlagForCapturePost(!on);
        }
        if (appContext != null) {
            config.save(appContext);
        }
    }

    /**
     * Whether noise cancellation is currently enabled. Note this reflects
     * the requested state; if the native engine failed to load, no
     * suppression takes place regardless.
     */
    public static synchronized boolean isEnabled() {
        return config.enabled && factory != null;
    }

    /** Snapshot of the current configuration. */
    static synchronized NoiseSuppressionConfig getConfig() {
        return config.copy();
    }

    /**
     * Applies and persists a new configuration. If engine parameters
     * changed, a new engine is built in the background and hot-swapped into
     * the running audio pipeline.
     */
    static synchronized void applyConfig(NoiseSuppressionConfig newConfig) {
        boolean rebuild = config.needsNewEngine(newConfig);
        boolean enabledChanged = config.enabled != newConfig.enabled;
        config = newConfig.copy();

        if (appContext != null) {
            config.save(appContext);
        }
        if (factory == null) {
            return;
        }
        if (enabledChanged) {
            factory.setBypassFlagForCapturePost(!config.enabled);
        }
        if (rebuild && suppressor != null) {
            suppressor.applyConfig(config);
        }
    }
}
