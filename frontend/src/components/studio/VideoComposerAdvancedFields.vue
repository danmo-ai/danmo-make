<template>
  <div class="video-advanced-fields">
    <div v-if="paramSchema.steps" class="video-advanced-fields__item">
      <div class="video-advanced-fields__head">
        <label>{{ $t('create.steps') }}</label>
        <span class="video-advanced-fields__val">{{ params.steps }}</span>
      </div>
      <DqSlider
        v-if="!isFixedScalar(paramSchema.steps)"
        v-model="stepsModel"
        :min="paramSchema.steps.min ?? 1"
        :max="paramSchema.steps.max ?? 100"
        :step="paramSchema.steps.step ?? 1"
        class="video-advanced-fields__slider"
      />
      <p v-else-if="paramSchema.steps.note" class="video-advanced-fields__note">{{ paramSchema.steps.note }}</p>
    </div>

    <div v-if="showGuideScale" class="video-advanced-fields__item">
      <div class="video-advanced-fields__head">
        <label>{{ $t('create.guidance') }}</label>
        <span class="video-advanced-fields__val">{{ params.guide_scale }}</span>
      </div>
      <DqSlider
        v-model="guideScaleModel"
        :min="paramSchema.guide_scale!.min ?? 1"
        :max="paramSchema.guide_scale!.max ?? 20"
        :step="paramSchema.guide_scale!.step ?? 0.1"
        class="video-advanced-fields__slider"
      />
    </div>

    <div v-if="paramSchema.fps" class="video-advanced-fields__item">
      <div class="video-advanced-fields__head">
        <label>{{ $t('create.fps') }}</label>
        <span class="video-advanced-fields__val">{{ params.fps }}</span>
      </div>
      <DqSlider
        v-if="!isFixedScalar(paramSchema.fps)"
        v-model="fpsModel"
        :min="paramSchema.fps.min ?? 1"
        :max="paramSchema.fps.max ?? 30"
        :step="paramSchema.fps.step ?? 1"
        class="video-advanced-fields__slider"
      />
    </div>

    <div v-if="showNumFrames" class="video-advanced-fields__item video-advanced-fields__item--readonly">
      <div class="video-advanced-fields__head">
        <label>{{ $t('create.numFrames') }}</label>
        <span class="video-advanced-fields__val">{{ numFrames }}</span>
      </div>
      <p class="video-advanced-fields__note">
        {{ $t('video.numFramesFormula', { sec: durationSec, fps: params.fps }) }}
        <template v-if="paramSchema.num_frames?.note"> · {{ paramSchema.num_frames.note }}</template>
      </p>
    </div>

    <div v-if="paramSchema.shift" class="video-advanced-fields__item">
      <div class="video-advanced-fields__head">
        <label>{{ $t('create.shift') }}</label>
        <span class="video-advanced-fields__val">{{ params.shift }}</span>
      </div>
      <DqSlider
        v-if="!isFixedScalar(paramSchema.shift)"
        v-model="shiftModel"
        :min="paramSchema.shift.min ?? 0"
        :max="paramSchema.shift.max ?? 20"
        :step="paramSchema.shift.step ?? 0.1"
        class="video-advanced-fields__slider"
      />
      <p v-else-if="paramSchema.shift.note" class="video-advanced-fields__note">{{ paramSchema.shift.note }}</p>
    </div>

    <div v-if="showSeedField" class="video-advanced-fields__item">
      <label class="video-advanced-fields__label">{{ $t('create.seed') }}</label>
      <div class="video-advanced-fields__seed">
        <DqInput
          v-model="seedModel"
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

    <div v-if="showNegativePrompt" class="video-advanced-fields__item">
      <label class="video-advanced-fields__label">{{ $t('studio.negativePrompt') }}</label>
      <DqInput
        v-model="negativePromptModel"
        type="textarea"
        :rows="2"
        :placeholder="$t('create.negativePlaceholder')"
        resize="none"
      />
    </div>

    <div v-if="showLora && compatibleLoras?.length" class="video-advanced-fields__item video-advanced-fields__item--stack">
      <div class="video-advanced-fields__lora-row">
        <label class="video-advanced-fields__label">{{ $t('studio.loraLabel') }}</label>
        <DqSelect
          v-model="loraModel"
          size="sm"
          clearable
          :placeholder="$t('studio.noLora')"
          class="video-advanced-fields__select"
        >
          <DqOption
            v-for="l in compatibleLoras"
            :key="String(l.id)"
            :label="loraOptionLabelForRow(l)"
            :value="String(l.id)"
          />
        </DqSelect>
      </div>
      <p v-if="selectedLoraHintKey" class="video-advanced-fields__lora-hint">
        {{ $t(selectedLoraHintKey) }}
      </p>
      <div v-if="params.lora" class="video-advanced-fields__item">
        <div class="video-advanced-fields__head">
          <label>{{ $t('create.loraScale') }}</label>
          <span class="video-advanced-fields__val">{{ params.lora_scale }}</span>
        </div>
        <DqSlider
          v-model="loraScaleModel"
          :min="paramSchema.lora_scale?.min ?? 0"
          :max="paramSchema.lora_scale?.max ?? 2"
          :step="paramSchema.lora_scale?.step ?? 0.05"
          class="video-advanced-fields__slider"
        />
      </div>
    </div>

    <div
      v-for="paramKey in inferenceEnumKeys"
      :key="paramKey"
      class="video-advanced-fields__item video-advanced-fields__item--stack"
    >
      <label class="video-advanced-fields__label">{{ paramFieldLabel(paramKey) }}</label>
      <DqSelect
        v-model="params[paramKey]"
        size="sm"
        class="video-advanced-fields__select"
      >
        <DqOption
          v-for="opt in enumParamOptions(paramSchema[paramKey])"
          :key="`${paramKey}-${opt}`"
          :value="opt"
          :label="enumOptionLabel(paramKey, opt)"
        />
      </DqSelect>
      <p v-if="paramFieldHint(paramKey)" class="video-advanced-fields__note">
        {{ paramFieldHint(paramKey) }}
      </p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import { Refresh } from '@danqing/dq-shell';
import type { SegmentComposeParams } from '@/composables/useLongVideoSegmentCompose';
import type { NormalizedParamSpec } from '@/utils/registryParamSchema';
import {
  VIDEO_INFERENCE_ENUM_KEYS,
  composerParamHintI18nKey,
  composerParamLabelI18nKey,
  enumOptionI18nKey,
  enumParamOptions,
} from '@/utils/registryParamSchema';
import {
  findCompatibleLora,
  loraHintKey,
  loraOptionLabel,
  type CompatibleLoraRow,
} from '@/utils/loraAdapterMeta';

const props = defineProps<{
  params: SegmentComposeParams;
  paramSchema: Record<string, NormalizedParamSpec>;
  durationSec: number;
  numFrames: number;
  showNegativePrompt?: boolean;
  showSeedField?: boolean;
  showLora?: boolean;
  compatibleLoras?: Record<string, unknown>[];
}>();

const { t: $t } = useI18n();

function bindField<K extends keyof SegmentComposeParams>(key: K) {
  return computed({
    get: () => props.params[key],
    set: (value: SegmentComposeParams[K]) => {
      props.params[key] = value;
    },
  });
}

const stepsModel = bindField('steps');
const guideScaleModel = bindField('guide_scale');
const fpsModel = bindField('fps');
const shiftModel = bindField('shift');
const seedModel = bindField('seed');
const negativePromptModel = bindField('negative_prompt');
const loraModel = bindField('lora');
const loraScaleModel = bindField('lora_scale');

function isFixedScalar(spec: NormalizedParamSpec | undefined): boolean {
  if (!spec) return false;
  if (spec.fixed === true) return true;
  const min = typeof spec.min === 'number' ? spec.min : null;
  const max = typeof spec.max === 'number' ? spec.max : null;
  return min != null && max != null && min === max;
}

const showGuideScale = computed(() => {
  const g = props.paramSchema.guide_scale;
  if (!g) return false;
  return !isFixedScalar(g);
});

const showNumFrames = computed(() => Boolean(props.paramSchema.num_frames));

const inferenceEnumKeys = computed(() =>
  VIDEO_INFERENCE_ENUM_KEYS.filter((key) => Boolean(props.paramSchema[key])),
);

function paramFieldLabel(paramKey: string): string {
  const spec = props.paramSchema[paramKey];
  if (spec && typeof spec.label === 'string' && spec.label.trim()) return spec.label;
  const i18nKey = composerParamLabelI18nKey(paramKey);
  return i18nKey ? $t(`create.${i18nKey}`) : paramKey;
}

function paramFieldHint(paramKey: string): string {
  const spec = props.paramSchema[paramKey];
  if (spec && typeof spec.note === 'string' && spec.note.trim()) return spec.note;
  const i18nKey = composerParamHintI18nKey(paramKey);
  return i18nKey ? $t(`create.${i18nKey}`) : '';
}

function enumOptionLabel(paramKey: string, value: string): string {
  const i18nKey = enumOptionI18nKey(paramKey, value);
  return i18nKey ? $t(`create.${i18nKey}`) : value;
}

function loraOptionLabelForRow(l: Record<string, unknown>): string {
  if (l.source === 'user_trained') {
    return loraOptionLabel(l as CompatibleLoraRow, $t('studio.myLoraTag'));
  }
  return loraOptionLabel(l as CompatibleLoraRow);
}

const selectedLoraHintKey = computed(() => {
  const id = String(props.params.lora || '');
  if (!id) return null;
  const row = findCompatibleLora((props.compatibleLoras || []) as CompatibleLoraRow[], id);
  return loraHintKey(row);
});

function randomizeSeed() {
  props.params.seed = String(Math.floor(Math.random() * 1_000_000));
}
</script>

<style scoped>
.video-advanced-fields {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.video-advanced-fields__item {
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-width: 0;
}

.video-advanced-fields__item--stack {
  gap: 12px;
}

.video-advanced-fields__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.video-advanced-fields__head label,
.video-advanced-fields__label {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  color: var(--dq-label-secondary);
}

.video-advanced-fields__val {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  color: var(--dq-label-primary);
  font-variant-numeric: tabular-nums;
}

.video-advanced-fields__slider {
  width: 100%;
}

.video-advanced-fields__slider :deep(.dq-slider) {
  width: 100%;
}

.video-advanced-fields__note {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
  color: var(--dq-label-tertiary);
}

.video-advanced-fields__seed {
  display: flex;
  align-items: center;
  gap: 6px;
}

.video-advanced-fields__seed :deep(.dq-input) {
  flex: 1;
  min-width: 0;
}

.video-advanced-fields__lora-row {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.video-advanced-fields__select {
  width: 100%;
}

.video-advanced-fields__lora-hint {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
  color: var(--dq-label-tertiary);
}
</style>
