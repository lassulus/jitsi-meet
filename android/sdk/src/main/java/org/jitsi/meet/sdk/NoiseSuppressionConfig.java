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
import android.content.SharedPreferences;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.ReadableType;
import com.facebook.react.bridge.WritableMap;

/**
 * Configuration of the DeepFilterNet noise suppressor: which model runs and
 * how aggressively it suppresses. Persisted in {@link SharedPreferences} so
 * it applies from app start, before the React Native runtime is up.
 *
 * Threshold semantics (libDF): frames whose local SNR is below
 * {@code minDbThresh} are fully muted (noise gate); above
 * {@code maxDbErbThresh}/{@code maxDbDfThresh} enhancement is bypassed
 * (clean speech). Raising the gate and the bypass thresholds suppresses
 * harder. The post filter further attenuates noisy bins at a slight speech
 * quality cost.
 */
final class NoiseSuppressionConfig {

    /**
     * Available models, shipped as APK assets under
     * {@code assets/noise-suppression/}. (DeepFilterNet2 is intentionally
     * absent: upstream libDF deprecated it and refuses to load it.)
     */
    enum Model {
        DFN3("dfn3", "DeepFilterNet3_onnx.tgz"),
        DFN3_LL("dfn3_ll", "DeepFilterNet3_ll_onnx.tgz");

        final String key;
        final String assetName;

        Model(String key, String assetName) {
            this.key = key;
            this.assetName = assetName;
        }

        static Model fromKey(String key) {
            for (Model m : values()) {
                if (m.key.equals(key)) {
                    return m;
                }
            }
            return DFN3;
        }
    }

    private static final String PREFS_NAME = "jitsi-noise-suppression";

    boolean enabled = true;
    Model model = Model.DFN3;
    /** Maximum noise attenuation in dB; >= 100 means unlimited. */
    float attenLimDb = 100f;
    float minDbThresh = -10f;
    float maxDbErbThresh = 30f;
    float maxDbDfThresh = 20f;
    boolean postFilter = false;

    NoiseSuppressionConfig copy() {
        NoiseSuppressionConfig c = new NoiseSuppressionConfig();
        c.enabled = enabled;
        c.model = model;
        c.attenLimDb = attenLimDb;
        c.minDbThresh = minDbThresh;
        c.maxDbErbThresh = maxDbErbThresh;
        c.maxDbDfThresh = maxDbDfThresh;
        c.postFilter = postFilter;
        return c;
    }

    /** Whether {@code other} differs in any way that requires a new engine. */
    boolean needsNewEngine(NoiseSuppressionConfig other) {
        return model != other.model
            || attenLimDb != other.attenLimDb
            || minDbThresh != other.minDbThresh
            || maxDbErbThresh != other.maxDbErbThresh
            || maxDbDfThresh != other.maxDbDfThresh
            || postFilter != other.postFilter;
    }

    static NoiseSuppressionConfig load(Context context) {
        SharedPreferences p = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
        NoiseSuppressionConfig c = new NoiseSuppressionConfig();
        c.enabled = p.getBoolean("enabled", c.enabled);
        c.model = Model.fromKey(p.getString("model", c.model.key));
        c.attenLimDb = p.getFloat("attenLimDb", c.attenLimDb);
        c.minDbThresh = p.getFloat("minDbThresh", c.minDbThresh);
        c.maxDbErbThresh = p.getFloat("maxDbErbThresh", c.maxDbErbThresh);
        c.maxDbDfThresh = p.getFloat("maxDbDfThresh", c.maxDbDfThresh);
        c.postFilter = p.getBoolean("postFilter", c.postFilter);
        return c;
    }

    void save(Context context) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putBoolean("enabled", enabled)
            .putString("model", model.key)
            .putFloat("attenLimDb", attenLimDb)
            .putFloat("minDbThresh", minDbThresh)
            .putFloat("maxDbErbThresh", maxDbErbThresh)
            .putFloat("maxDbDfThresh", maxDbDfThresh)
            .putBoolean("postFilter", postFilter)
            .apply();
    }

    /** Merges fields present in {@code map} (JS partial update). */
    void mergeFrom(ReadableMap map) {
        if (map.hasKey("enabled") && map.getType("enabled") == ReadableType.Boolean) {
            enabled = map.getBoolean("enabled");
        }
        if (map.hasKey("model") && map.getType("model") == ReadableType.String) {
            model = Model.fromKey(map.getString("model"));
        }
        if (map.hasKey("attenLimDb") && map.getType("attenLimDb") == ReadableType.Number) {
            attenLimDb = (float) map.getDouble("attenLimDb");
        }
        if (map.hasKey("minDbThresh") && map.getType("minDbThresh") == ReadableType.Number) {
            minDbThresh = (float) map.getDouble("minDbThresh");
        }
        if (map.hasKey("maxDbErbThresh") && map.getType("maxDbErbThresh") == ReadableType.Number) {
            maxDbErbThresh = (float) map.getDouble("maxDbErbThresh");
        }
        if (map.hasKey("maxDbDfThresh") && map.getType("maxDbDfThresh") == ReadableType.Number) {
            maxDbDfThresh = (float) map.getDouble("maxDbDfThresh");
        }
        if (map.hasKey("postFilter") && map.getType("postFilter") == ReadableType.Boolean) {
            postFilter = map.getBoolean("postFilter");
        }
    }

    WritableMap toMap() {
        WritableMap map = Arguments.createMap();
        map.putBoolean("enabled", enabled);
        map.putString("model", model.key);
        map.putDouble("attenLimDb", attenLimDb);
        map.putDouble("minDbThresh", minDbThresh);
        map.putDouble("maxDbErbThresh", maxDbErbThresh);
        map.putDouble("maxDbDfThresh", maxDbDfThresh);
        map.putBoolean("postFilter", postFilter);
        return map;
    }
}
