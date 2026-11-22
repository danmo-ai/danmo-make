<template>
  <div class="composer-advanced-fields" :class="{ 'composer-advanced-fields--stacked': stacked }">
    <div v-if="paramSchema.steps || showGuidanceSlider" class="composer-advanced-fields__row">
      <div v-if="paramSchema.steps" class="composer-advanced-fields__field">
        <label>{{ $t('create.stepsLabel') }}</label>
        <DqSlider
          v-model="params.steps"
          :min="paramSchema.steps.min ?? 1"
          :max="paramSchema.steps.max ?? 50"
          :step="paramSchema.steps.step ?? 1"
        />
        <span class="composer-advanced-fields__val">{{ params.steps }}</span>
      </div>

      <div v-if="showGuidanceSlider" class="composer-advanced-fields__field">
        <label>{{ $t('create.guidanceLabel') }}</label>
        <DqSlider
          v-model="params.guidance"
          :min="paramSchema.guidance?.min ?? 0"
          :max="paramSchema.guidance?.max ?? 20"
          :step="paramSchema.guidance?.step ?? 0.5"
        />
        <span class="composer-advanced-fields__val">{{ params.guidance }}</span>
      </div>
    </div>

    <div class="composer-advanced-fields__row">
      <div class="composer-advanced-fields__field">
        <label>{{ $t('studio.seed') }}</label>
        <div class="composer-advanced-fields__seed">
          <DqInput
            v-model="params.seed"
            size="sm"
            :placeholder="$t('studio.seedPlaceholder')"
          />
          <DqIconButton
            type="text"
            size="sm"
            :label="$t('create.randomSeed')"
            @click="randomizeSeed"
          >
            <DqIcon :size="12"><Refresh /></DqIcon>
          </DqIconButton>
        </div>
      </div>

      <div v-if="showStrengthSlider" class="composer-advanced-fields__field">
        <label>{{ $t('create.strengthLabel') }}</label>
        <DqSlider
          v-model="params.strength"
          :min="paramSchema.strength?.min ?? 0"
          :max="paramSchema.strength?.max ?? 1"
          :step="paramSchema.strength?.step ?? 0.05"
        />
        <span class="composer-advanced-fields__val">{{ params.strength }}</span>
      </div>
    </div>

    <div v-if="paramSchema.use_pe" class="composer-advanced-fields__row">
      <div class="composer-advanced-fields__field">
        <label>{{ $t('create.promptEnhance') }}</label>
        <DqSwitch v-model="params.use_pe" />
      </div>
    </div>

    <div v-if="schedulerOptions.length" class="composer-advanced-fields__row">
      <div class="composer-advanced-fields__field composer-advanced-fields__field--full">
        <label>{{ schedulerLabel }}</label>
        <DqSelect v-model="params.scheduler" size="sm" class="composer-advanced-fields__select">
          <DqOption
            v-for="opt in schedulerOptions"
            :key="String(opt)"
            :label="String(opt)"
            :value="opt"
          />
        </DqSelect>
      </div>
    </div>

    <div v-if="loraSupported" class="composer-advanced-fields__row">
      <div class="composer-advanced-fields__field composer-advanced-fields__field--full composer-advanced-fields__field--stack">
        <div class="composer-advanced-fields__inline">
          <label>{{ $t('studio.loraLabel') }}</label>
          <DqSelect
            v-model="params.lora"
            size="sm"
            clearable
            :placeholder="$t('studio.noLora')"
            class="composer-advanced-fields__select composer-advanced-fields__select--grow"
          >
            <DqOption
              v-for="l in compatibleLoras"
              :key="String(l.id)"
              :label="loraOptionLabel(l)"
              :value="l.id"
              :disabled="l.ready === false"
            />
          </DqSelect>
        </div>
        <p v-if="!installedLoraCount" class="composer-advanced-fields__hint">{{ $t('studio.loraNoneReady') }}</p>
        <div v-if="params.lora" class="composer-advanced-fields__inline">
          <label>{{ $t('create.loraScale') }}</label>
          <DqSlider v-model="params.lora_scale" :min="0" :max="2" :step="0.05" class="composer-advanced-fields__slider-grow" />
          <span class="composer-advanced-fields__val">{{ params.lora_scale }}</span>
        </div>
        <p v-if="params.lora && selectedLoraHintKey" class="composer-advanced-fields__hint">
          {{ $t(selectedLoraHintKey) }}
        </p>
        <p v-if="params.lora && stackedAdapterNames" class="composer-advanced-fields__hint">
          {{ $t('studio.loraStack', { names: stackedAdapterNames }) }}
        </p>
        <p
          v-if="params.lora && selectedLoraScaleRange"
          class="composer-advanced-fields__hint composer-advanced-fields__hint--sub"
          :class="{ 'composer-advanced-fields__hint--warn': loraScaleOutsideRange }"
        >
          <template v-if="loraScaleOutsideRange">
            {{ $t('studio.loraScaleOutsideRange', {
              value: params.lora_scale,
              min: selectedLoraScaleRange[0],
              max: selectedLoraScaleRange[1],
            }) }}
          </template>
          <template v-else>
            {{ $t('studio.loraRecommendedScale', { min: selectedLoraScaleRange[0], max: selectedLoraScaleRange[1] }) }}
          </template>
        </p>
      </div>
    </div>

    <div v-if="compatibleControlNets?.length" class="composer-advanced-fields__row">
      <div class="composer-advanced-fields__field composer-advanced-fields__field--full composer-advanced-fields__field--stack">
        <p v-if="!controlNetRuntimeAvailable" class="composer-advanced-fields__hint">
          {{ $t('studio.controlnetMlxOnly') }}
        </p>
        <div class="composer-advanced-fields__inline">
          <label>{{ $t('studio.controlNet') }}</label>
          <DqSelect
            v-model="params.controlnet"
            size="sm"
            clearable
            :disabled="!controlNetRuntimeAvailable"
            :placeholder="$t('studio.noControlNet')"
            class="composer-advanced-fields__select composer-advanced-fields__select--grow"
          >
            <DqOption
              v-for="n in compatibleControlNets"
              :key="String(n.key)"
              :label="controlNetOptionLabel(n)"
              :value="String(n.key)"
              :disabled="!controlNetReady(n)"
            />
          </DqSelect>
        </div>
        <p v-if="params.controlnet" class="composer-advanced-fields__hint">{{ controlNetGuideHint }}</p>
        <div v-if="params.controlnet" class="composer-advanced-fields__inline">
          <label>{{ $t('create.controlNetStrengthLabel') }}</label>
          <DqSlider v-model="params.controlnet_strength" :min="0" :max="2" :step="0.05" class="composer-advanced-fields__slider-grow" />
          <span class="composer-advanced-fields__val">{{ params.controlnet_strength }}</span>
        </div>
        <div v-if="params.controlnet" class="composer-advanced-fields__inline">
          <label>{{ $t('canvas.controlImage') }}</label>
          <div class="composer-advanced-fields__control-asset">
            <div v-if="controlImage" class="composer-advanced-fields__thumb">
              <img :src="controlImage.previewUrl" alt="control" />
              <DqIconButton type="text" size="sm" :label="$t('common.delete')" @click="$emit('remove-control')">
                <DqIcon :size="10"><Close /></DqIcon>
              </DqIconButton>
            </div>
            <DqButton v-else size="sm" type="secondary" @click="$emit('pick-control')">
              {{ $t('canvas.pickControlImage') }}
            </DqButton>
          </div>
        </div>
      </div>
    </div>

    <div
      v-for="paramKey in inferenceEnumKeys"
      :key="paramKey"
      class="composer-advanced-fields__row"
    >
      <div class="composer-advanced-fields__field composer-advanced-fields__field--full composer-advanced-fields__field--stack">
        <div class="composer-advanced-fields__inline">
          <label>{{ paramFieldLabel(paramKey) }}</label>
          <DqSelect
            v-model="params[paramKey]"
            size="sm"
            class="composer-advanced-fields__select composer-advanced-fields__select--grow"
          >
            <DqOption
              v-for="opt in enumParamOptions(paramSchema[paramKey])"
              :key="`${paramKey}-${opt}`"
              :value="opt"
              :label="enumOptionLabel(paramKey, opt)"
            />
          </DqSelect>
        </div>
        <p v-if="paramFieldHint(paramKey)" class="composer-advanced-fields__hint">
          {{ paramFieldHint(paramKey) }}
        </p>
      </div>
    </div>

    <div v-if="paramSchema.latent_refine_scale" class="composer-advanced-fields__row">
      <div class="composer-advanced-fields__field composer-advanced-fields__field--full composer-advanced-fields__field--stack">
        <div class="composer-advanced-fields__inline">
          <label>{{ paramFieldLabel('latent_refine_scale') }}</label>
          <DqSlider
            v-model="params.latent_refine_scale"
            :min="paramSchema.latent_refine_scale.min ?? 1"
            :max="paramSchema.latent_refine_scale.max ?? 2"
            :step="paramSchema.latent_refine_scale.step ?? 0.25"
            class="composer-advanced-fields__slider-grow"
          />
          <span class="composer-advanced-fields__val">{{ params.latent_refine_scale }}</span>
        </div>
        <div v-if="Number(params.latent_refine_scale) > 1 && paramSchema.latent_refine_denoise" class="composer-advanced-fields__inline">
          <label>{{ paramFieldLabel('latent_refine_denoise') }}</label>
          <DqSlider
            v-model="params.latent_refine_denoise"
            :min="paramSchema.latent_refine_denoise.min ?? 0"
            :max="paramSchema.latent_refine_denoise.max ?? 1"
            :step="paramSchema.latent_refine_denoise.step ?? 0.05"
            class="composer-advanced-fields__slider-grow"
          />
          <span class="composer-advanced-fields__val">{{ params.latent_refine_denoise }}</span>
        </div>
      </div>
    </div>

    <div v-if="showNegativePrompt" class="composer-advanced-fields__row">
      <div class="composer-advanced-fields__field composer-advanced-fields__field--full composer-advanced-fields__field--stack">
        <label>{{ $t('studio.negativePrompt') }}</label>
        <DqInput
          v-model="params.negative_prompt"
          type="textarea"
          :rows="2"
          :placeholder="$t('create.negativePlaceholder')"
          resize="none"
        />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { Close, Refresh } from '@danqing/dq-shell';
import {
  applyControlNetRegistryDefaults,
  controlNetDisplayName,
  controlNetGuideType,
  controlNetReady,
} from '@/composables/useStructuralGuide';
import {
  type CompatibleLoraRow,
  findCompatibleLora,
  loraHintKey,
  loraRecommendedScale,
  stackedAdapterToken,
} from '@/utils/loraAdapterMeta';
import {
  COMPOSER_INFERENCE_ENUM_KEYS,
  composerParamHintI18nKey,
  composerParamLabelI18nKey,
  enumOptionI18nKey,
  enumParamOptions,
  img2imgUsesStrength,
  normalizeParamsDef,
} from '@/utils/registryParamSchema';

export type ComposerAdvancedParams = {
  steps: number;
  guidance: number;
  seed: string;
  strength: number;
  negative_prompt: string;
  use_pe?: boolean;
  scheduler?: string;
  lora?: string;
  lora_scale?: number;
  controlnet?: string;
  controlnet_strength?: number;
  lemica_mode?: string;
  teacache_mode?: string;
  latent_refine_scale?: number;
  latent_refine_denoise?: number;
};

const props = withDefaults(defineProps<{
  params: ComposerAdvancedParams;
  model: string;
  mode?: string;
  referenceImage?: { previewUrl: string; path: string } | null;
  currentModelConfig?: Record<string, unknown> | null;
  compatibleLoras?: Record<string, unknown>[];
  compatibleControlNets?: Record<string, unknown>[];
  controlNetRuntimeAvailable?: boolean;
  controlImage?: { previewUrl: string; path: string } | null;
  showNegativePrompt?: boolean;
  stacked?: boolean;
}>(), {
  mode: 'text2img',
  controlNetRuntimeAvailable: true,
  showNegativePrompt: false,
  stacked: false,
});

defineEmits<{
  (e: 'pick-control'): void;
  (e: 'remove-control'): void;
}>();

const { t: $t } = useI18n();

const paramSchema = computed(() =>
  normalizeParamsDef(props.currentModelConfig?.parameters as Record<string, unknown> | undefined),
);

const isImg2imgMode = computed(() => props.mode === 'img2img');

const showStrengthSlider = computed(
  () => isImg2imgMode.value && img2imgUsesStrength(props.currentModelConfig?.parameters as Record<string, unknown> | undefined),
);

const showGuidanceSlider = computed(() => {
  const g = paramSchema.value.guidance;
  if (!g) return false;
  if (g.fixed === true) return false;
  const min = typeof g.min === 'number' ? g.min : 0;
  const max = typeof g.max === 'number' ? g.max : 20;
  return min !== max;
});

const schedulerOptions = computed(() => {
  const scheduler = (props.currentModelConfig?.parameters as { scheduler?: { options?: unknown[]; label?: string } } | undefined)?.scheduler;
  return Array.isArray(scheduler?.options) ? scheduler.options : [];
});

const schedulerLabel = computed(() => {
  const scheduler = (props.currentModelConfig?.parameters as { scheduler?: { label?: string } } | undefined)?.scheduler;
  return scheduler?.label || $t('create.schedulerLabel');
});

const loraSupported = computed(() =>
  Boolean((props.currentModelConfig?.parameters as { lora_support?: boolean } | undefined)?.lora_support),
);

const installedLoraCount = computed(
  () => ((props.compatibleLoras || []) as CompatibleLoraRow[]).filter((lora) => lora.ready !== false).length,
);

const selectedLoraRow = computed<CompatibleLoraRow | undefined>(() => {
  const id = String(props.params.lora || '');
  if (!id) return undefined;
  return findCompatibleLora((props.compatibleLoras || []) as CompatibleLoraRow[], id);
});

const selectedLoraHintKey = computed(() => loraHintKey(selectedLoraRow.value));

const stackedAdapterNames = computed(() => {
  const extras = selectedLoraRow.value?.extra_adapters || [];
  return extras.map((item) => stackedAdapterToken(item)).filter(Boolean).join(', ');
});

const selectedLoraScaleRange = computed(() => loraRecommendedScale(selectedLoraRow.value));

const loraScaleOutsideRange = computed(() => {
  const rng = selectedLoraScaleRange.value;
  const value = Number(props.params.lora_scale);
  if (!rng || !Number.isFinite(value)) return false;
  // Slider step is 0.05; tolerate float noise at the boundaries.
  return value < rng[0] - 1e-6 || value > rng[1] + 1e-6;
});

const inferenceEnumKeys = computed(() =>
  COMPOSER_INFERENCE_ENUM_KEYS.filter((key) => Boolean(paramSchema.value[key])),
);

const selectedControlNetRow = computed(() => {
  const key = String(props.params.controlnet || '');
  if (!key) return undefined;
  return props.compatibleControlNets?.find((n) => String(n.key) === key);
});

const controlNetGuideHint = computed(() => {
  const guide = controlNetGuideType(selectedControlNetRow.value);
  if (guide === 'auto') return $t('studio.controlnetZImageUnionHint');
  return $t('studio.controlnetBundleHint');
});

function loraOptionLabel(l: Record<string, unknown>): string {
  const base = String(l.name || l.id || '');
  const tagged = l.source === 'user_trained' ? `${base} (${$t('studio.myLoraTag')})` : base;
  if (l.ready === false) return `${tagged} (${$t('studio.loraNotInstalled')})`;
  return tagged;
}

function controlNetOptionLabel(n: Record<string, unknown>): string {
  const name = controlNetDisplayName(n);
  return controlNetReady(n) ? name : `${name} (${$t('studio.controlnetNotInstalled')})`;
}

function randomizeSeed() {
  props.params.seed = String(Math.floor(Math.random() * 1_000_000));
}

function paramFieldLabel(paramKey: string): string {
  const spec = paramSchema.value[paramKey];
  if (spec && typeof spec.label === 'string' && spec.label.trim()) return spec.label;
  const i18nKey = composerParamLabelI18nKey(paramKey);
  return i18nKey ? $t(`create.${i18nKey}`) : paramKey;
}

function paramFieldHint(paramKey: string): string {
  const spec = paramSchema.value[paramKey];
  if (spec && typeof spec.note === 'string' && spec.note.trim()) return spec.note;
  const i18nKey = composerParamHintI18nKey(paramKey);
  return i18nKey ? $t(`create.${i18nKey}`) : '';
}

function enumOptionLabel(paramKey: string, value: string): string {
  const i18nKey = enumOptionI18nKey(paramKey, value);
  return i18nKey ? $t(`create.${i18nKey}`) : value;
}

watch(
  () => props.params.controlnet,
  (key, prev) => {
    if (!key || key === prev) return;
    applyControlNetRegistryDefaults(String(key), props.compatibleControlNets, props.params);
  },
);
</script>

<style scoped>
.composer-advanced-fields {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.composer-advanced-fields__row {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.composer-advanced-fields__field {
  display: flex;
  align-items: center;
  gap: 8px;
  flex: 1;
  min-width: 220px;
}

.composer-advanced-fields--stacked {
  --composer-field-label: 8.75rem;
  --composer-field-value: 2.5rem;
  min-width: 0;
  max-width: 100%;
}

.composer-advanced-fields--stacked .composer-advanced-fields__row {
  flex-direction: column;
  flex-wrap: nowrap;
  min-width: 0;
  width: 100%;
}

.composer-advanced-fields--stacked .composer-advanced-fields__field,
.composer-advanced-fields--stacked .composer-advanced-fields__inline {
  display: grid;
  grid-template-columns: var(--composer-field-label) minmax(0, 1fr) var(--composer-field-value);
  align-items: center;
  column-gap: 10px;
  min-width: 0;
  width: 100%;
}

.composer-advanced-fields--stacked .composer-advanced-fields__field {
  flex: 1 1 100%;
}

.composer-advanced-fields__field--full {
  flex: 1 1 100%;
}

.composer-advanced-fields__field--stack {
  flex-direction: column;
  align-items: stretch;
  gap: 8px;
}

.composer-advanced-fields--stacked .composer-advanced-fields__field--stack {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  grid-template-columns: none;
}

.composer-advanced-fields__field label {
  font-size: var(--dq-font-size-caption);
  font-weight: 500;
  color: var(--dq-label-secondary);
  white-space: nowrap;
  width: 72px;
  flex-shrink: 0;
}

.composer-advanced-fields--stacked .composer-advanced-fields__field label,
.composer-advanced-fields--stacked .composer-advanced-fields__inline label {
  width: auto;
  min-width: 0;
  white-space: normal;
  line-height: 1.35;
}

.composer-advanced-fields__field--stack > label {
  width: auto;
}

.composer-advanced-fields__inline {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
}

.composer-advanced-fields__inline label {
  font-size: var(--dq-font-size-caption);
  font-weight: 500;
  color: var(--dq-label-secondary);
  min-width: 72px;
  flex-shrink: 0;
}

.composer-advanced-fields__select {
  width: 220px;
}

.composer-advanced-fields__select--grow {
  flex: 1;
  width: auto;
  min-width: 0;
}

.composer-advanced-fields--stacked .composer-advanced-fields__select,
.composer-advanced-fields--stacked .composer-advanced-fields__seed,
.composer-advanced-fields--stacked .composer-advanced-fields__control-asset {
  grid-column: 2 / -1;
  width: 100%;
  min-width: 0;
  max-width: 100%;
}

.composer-advanced-fields__control-asset {
  display: flex;
  align-items: center;
  min-width: 0;
}

.composer-advanced-fields--stacked .composer-advanced-fields__field :deep(.dq-slider),
.composer-advanced-fields--stacked .composer-advanced-fields__inline :deep(.dq-slider),
.composer-advanced-fields--stacked .composer-advanced-fields__field :deep(.dq-select),
.composer-advanced-fields--stacked .composer-advanced-fields__inline :deep(.dq-select) {
  width: 100%;
  min-width: 0;
  max-width: 100%;
}

.composer-advanced-fields--stacked .composer-advanced-fields__hint {
  padding-left: calc(var(--composer-field-label) + 10px);
}

.composer-advanced-fields__slider-grow {
  flex: 1;
  min-width: 0;
}

.composer-advanced-fields__val {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
  min-width: 28px;
  text-align: right;
  font-variant-numeric: tabular-nums;
}

.composer-advanced-fields__seed {
  display: flex;
  align-items: center;
  gap: 4px;
  flex: 1;
  min-width: 0;
}

.composer-advanced-fields__seed :deep(.dq-input) {
  flex: 1;
  min-width: 0;
}

.composer-advanced-fields__hint {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
  color: var(--dq-label-tertiary);
}

.composer-advanced-fields__hint--sub {
  color: var(--dq-label-quaternary, var(--dq-label-tertiary));
}

.composer-advanced-fields__hint--warn {
  color: var(--dq-warning);
}

.composer-advanced-fields__asset-row {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
}

.composer-advanced-fields__asset-row label {
  min-width: 72px;
}

.composer-advanced-fields__thumb {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 6px 2px 2px;
  border-radius: 8px;
  border: 0.5px solid var(--dq-glass-border, var(--dq-border-subtle));
  background: color-mix(in srgb, var(--dq-surface-elevated) 80%, transparent);
}

.composer-advanced-fields__thumb img {
  width: 36px;
  height: 36px;
  object-fit: cover;
  border-radius: 6px;
}
</style>
