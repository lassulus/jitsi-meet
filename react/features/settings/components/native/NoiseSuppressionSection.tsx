import Slider from '@react-native-community/slider';
import { throttle } from 'lodash-es';
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { NativeModules, Text, TouchableHighlight, View, ViewStyle } from 'react-native';

import Icon from '../../../base/icons/components/Icon';
import { IconCheck } from '../../../base/icons/svg';
import BaseTheme from '../../../base/ui/components/BaseTheme.native';
import Switch from '../../../base/ui/components/native/Switch';

import FormRow from './FormRow';
import FormSection from './FormSection';
import styles from './styles';

const { NoiseSuppression } = NativeModules;

/**
 * The noise suppression configuration, as exposed by the native
 * {@code NoiseSuppression} module. Native {@code SharedPreferences} is the
 * single source of truth; this component only mirrors it locally.
 */
interface IConfig {
    attenLimDb: number;
    available: boolean;
    enabled: boolean;
    maxDbDfThresh: number;
    maxDbErbThresh: number;
    minDbThresh: number;
    model: string;
    postFilter: boolean;
}

/**
 * The selectable DeepFilterNet models.
 */
const MODELS = [
    { label: 'settingsView.noiseSuppressionModelDfn3', model: 'dfn3' },
    { label: 'settingsView.noiseSuppressionModelDfn3Ll', model: 'dfn3_ll' }
];

/**
 * Suppression strength presets, applied as partial config updates.
 */
const STRENGTH_PRESETS = {
    gentle: {
        attenLimDb: 12,
        maxDbDfThresh: 20,
        maxDbErbThresh: 30,
        minDbThresh: -15,
        postFilter: false
    },
    balanced: {
        attenLimDb: 100,
        maxDbDfThresh: 20,
        maxDbErbThresh: 30,
        minDbThresh: -10,
        postFilter: false
    },
    aggressive: {
        attenLimDb: 100,
        maxDbDfThresh: 25,
        maxDbErbThresh: 35,
        minDbThresh: -5,
        postFilter: true
    }
};

const PRESETS: Array<{ label: string; preset: keyof typeof STRENGTH_PRESETS }> = [
    { label: 'settingsView.noiseSuppressionGentle', preset: 'gentle' },
    { label: 'settingsView.noiseSuppressionBalanced', preset: 'balanced' },
    { label: 'settingsView.noiseSuppressionAggressive', preset: 'aggressive' }
];

const DISABLED_STYLE: ViewStyle = {
    opacity: 0.5
};

const SLIDER_CONTAINER_STYLE: ViewStyle = {
    alignItems: 'center',
    flexDirection: 'row',
    marginVertical: BaseTheme.spacing[1]
};

const SLIDER_STYLE: ViewStyle = {
    height: 40,
    width: 220
};

const NoiseSuppressionSection = () => {
    const { t } = useTranslation();
    const [ config, setConfig ] = useState<IConfig | undefined>(undefined);
    const [ attenLimDb, setAttenLimDb ] = useState<number | undefined>(undefined);

    useEffect(() => {
        if (!NoiseSuppression) {
            return;
        }

        NoiseSuppression.getConfig()
            .then((fullConfig: IConfig) => {
                setConfig(fullConfig);
                setAttenLimDb(fullConfig.attenLimDb);
            })
            .catch(() => setConfig(undefined));
    }, []);

    const applyConfig = useCallback((partial: Partial<IConfig>) => {
        NoiseSuppression.setConfig(partial)
            .then((fullConfig: IConfig) => {
                // Always adopt the resolved (full) config: the native side
                // applies updates serially, so the last resolution wins.
                setConfig(fullConfig);
                setAttenLimDb(fullConfig.attenLimDb);
            })
            .catch(() => {
                // Keep the last known config.
            });
    }, []);

    const commitAttenLimDb = useMemo(() => throttle((value: number) => {
        NoiseSuppression.setConfig({ attenLimDb: value })
            .then(setConfig)
            .catch(() => {
                // Keep the last known config.
            });
    }, 250), []);

    const onAttenLimDbChange = useCallback((value: number) => {
        setAttenLimDb(value);
        commitAttenLimDb(value);
    }, [ commitAttenLimDb ]);

    const onEnabledToggled = useCallback((enabled?: boolean) => {
        applyConfig({ enabled: Boolean(enabled) });
    }, [ applyConfig ]);

    const onPostFilterToggled = useCallback((postFilter?: boolean) => {
        applyConfig({ postFilter: Boolean(postFilter) });
    }, [ applyConfig ]);

    const onModelSelected = useCallback((model: string) => () => {
        applyConfig({ model });
    }, [ applyConfig ]);

    const onPresetSelected = useCallback((preset: keyof typeof STRENGTH_PRESETS) => () => {
        applyConfig(STRENGTH_PRESETS[preset]);
    }, [ applyConfig ]);

    const activePreset = useMemo(() => {
        if (!config) {
            return undefined;
        }

        return (Object.keys(STRENGTH_PRESETS) as Array<keyof typeof STRENGTH_PRESETS>)
            .find(name => Object.entries(STRENGTH_PRESETS[name])
                .every(([ key, value ]) => config[key as keyof IConfig] === value));
    }, [ config ]);

    if (!config?.available) {
        return null;
    }

    const { enabled } = config;
    const displayedAttenLimDb = attenLimDb ?? config.attenLimDb;

    const renderSelectableRow = (label: string, selected: boolean, onPress: () => void) => (
        <TouchableHighlight
            disabled = { !enabled }
            key = { label }
            onPress = { onPress }>
            <View style = { styles.fieldContainer as ViewStyle }>
                <View style = { styles.fieldLabelContainer as ViewStyle }>
                    <Text
                        style = { [
                            styles.text,
                            styles.fieldLabelText
                        ] }>
                        { t(label) }
                    </Text>
                </View>
                <View style = { styles.fieldValueContainer as ViewStyle }>
                    { selected && <Icon
                        color = { BaseTheme.palette.icon01 }
                        size = { 20 }
                        src = { IconCheck } /> }
                </View>
            </View>
        </TouchableHighlight>
    );

    return (
        <FormSection
            label = 'settingsView.noiseSuppression'>
            <FormRow
                label = 'settingsView.noiseSuppressionEnabled'>
                <Switch
                    checked = { enabled }
                    onChange = { onEnabledToggled } />
            </FormRow>
            <View
                pointerEvents = { enabled ? 'auto' : 'none' }
                style = { enabled ? undefined : DISABLED_STYLE }>
                <FormSection
                    label = 'settingsView.noiseSuppressionModel'>
                    {
                        MODELS.map(({ label, model }) =>
                            renderSelectableRow(label, config.model === model, onModelSelected(model)))
                    }
                </FormSection>
                <FormSection
                    label = 'settingsView.noiseSuppressionStrength'>
                    {
                        PRESETS.map(({ label, preset }) =>
                            renderSelectableRow(label, activePreset === preset, onPresetSelected(preset)))
                    }
                </FormSection>
                <FormRow
                    label = 'settingsView.noiseSuppressionAttenLim'
                    layout = 'column'>
                    <View style = { SLIDER_CONTAINER_STYLE }>
                        <Slider
                            disabled = { !enabled }
                            maximumTrackTintColor = { BaseTheme.palette.ui10 }
                            maximumValue = { 100 }
                            minimumTrackTintColor = { BaseTheme.palette.action01 }
                            minimumValue = { 6 }
                            onValueChange = { onAttenLimDbChange }
                            step = { 1 }
                            style = { SLIDER_STYLE }
                            thumbTintColor = { BaseTheme.palette.ui10 }
                            value = { config.attenLimDb } />
                        <Text style = { styles.text }>
                            {
                                displayedAttenLimDb >= 100
                                    ? t('settingsView.noiseSuppressionUnlimited')
                                    : `${displayedAttenLimDb} dB`
                            }
                        </Text>
                    </View>
                </FormRow>
                <FormRow
                    label = 'settingsView.noiseSuppressionPostFilter'>
                    <Switch
                        checked = { config.postFilter }
                        disabled = { !enabled }
                        onChange = { onPostFilterToggled } />
                </FormRow>
            </View>
        </FormSection>
    );
};

export default NoiseSuppressionSection;
