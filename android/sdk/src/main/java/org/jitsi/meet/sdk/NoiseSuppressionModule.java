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

import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.module.annotations.ReactModule;

/**
 * React Native bridge for the DeepFilterNet noise suppression settings.
 *
 * JS contract:
 * <pre>
 * config = {
 *   available:      boolean (read-only),
 *   enabled:        boolean,
 *   model:          'dfn2' | 'dfn3' | 'dfn3_ll',
 *   attenLimDb:     number (dB, >= 100 means unlimited),
 *   minDbThresh:    number (noise gate, default -10),
 *   maxDbErbThresh: number (bypass threshold, default 30),
 *   maxDbDfThresh:  number (bypass threshold, default 20),
 *   postFilter:     boolean
 * }
 * </pre>
 * {@code setConfig} accepts a partial map and resolves with the resulting
 * full config. Changes apply immediately (engines hot-swap mid-call) and
 * persist across restarts.
 */
@ReactModule(name = NoiseSuppressionModule.NAME)
class NoiseSuppressionModule extends ReactContextBaseJavaModule {

    public static final String NAME = "NoiseSuppression";

    public NoiseSuppressionModule(ReactApplicationContext reactContext) {
        super(reactContext);
    }

    @Override
    public String getName() {
        return NAME;
    }

    @ReactMethod
    public void getConfig(Promise promise) {
        com.facebook.react.bridge.WritableMap map = NoiseSuppression.getConfig().toMap();
        map.putBoolean("available", NoiseSuppression.isAvailable());
        promise.resolve(map);
    }

    @ReactMethod
    public void setConfig(ReadableMap partial, Promise promise) {
        NoiseSuppressionConfig config = NoiseSuppression.getConfig();
        config.mergeFrom(partial);
        NoiseSuppression.applyConfig(config);

        com.facebook.react.bridge.WritableMap map = NoiseSuppression.getConfig().toMap();
        map.putBoolean("available", NoiseSuppression.isAvailable());
        promise.resolve(map);
    }
}
