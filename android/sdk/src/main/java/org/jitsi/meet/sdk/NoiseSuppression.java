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

import com.oney.WebRTCModule.WebRTCModuleOptions;

import org.jitsi.meet.sdk.log.JitsiMeetLogger;
import org.webrtc.ExternalAudioProcessingFactory;

/**
 * Deep-learning noise cancellation (DeepFilterNet3) for the microphone
 * capture path.
 *
 * The suppressor is installed as a WebRTC APM capture post-processor via the
 * webrtc-sdk {@link ExternalAudioProcessingFactory}, so it runs after echo
 * cancellation and before the audio encoder, for every conference. It is
 * enabled by default; {@link #setEnabled(boolean)} toggles it at runtime with
 * no renegotiation.
 */
public final class NoiseSuppression {
    private static final String TAG = NoiseSuppression.class.getSimpleName();

    private static ExternalAudioProcessingFactory factory;
    private static boolean enabled = true;

    private NoiseSuppression() {
    }

    /**
     * Installs the noise suppressor into the WebRTC module options. Called
     * once before the React Native runtime (and thus WebRTC) is initialized.
     */
    static synchronized void install(WebRTCModuleOptions options) {
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

            factory = new ExternalAudioProcessingFactory();
            factory.setCapturePostProcessing(new DeepFilterNoiseSuppressor());
            factory.setBypassFlagForCapturePost(!enabled);
            options.audioProcessingFactory = factory;
            JitsiMeetLogger.i(TAG + " installed DeepFilterNet3 capture post-processor");
        } catch (UnsatisfiedLinkError | RuntimeException e) {
            JitsiMeetLogger.e(e, TAG + " failed to install");
            factory = null;
        }
    }

    /**
     * Enables or disables noise cancellation for the local microphone. Safe
     * to call at any time, including mid-conference.
     */
    public static synchronized void setEnabled(boolean on) {
        enabled = on;
        if (factory != null) {
            factory.setBypassFlagForCapturePost(!on);
        }
    }

    /**
     * Whether noise cancellation is currently enabled. Note this reflects the
     * requested state; if the native engine failed to load, no suppression
     * takes place regardless.
     */
    public static synchronized boolean isEnabled() {
        return enabled && factory != null;
    }
}
