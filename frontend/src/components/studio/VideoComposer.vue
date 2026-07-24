<template>
  <div
    class="video-composer studio-composer-shell dq-glass--panel"
    :class="{ 'video-composer--expanded-prompt': isLongDuration }"
  >
    <!-- Model not-ready alert -->
    <div v-if="modelNotReady" class="video-composer__model-meta">
      <DqAlert
        :title="$tt('studio.modelNotReady', { name: modelNotReadyName || '' })"
        type="warning"
        :closable="false"
        class="studio-alert-mt"
      >
        <template #default>
          <span>{{ $tt('studio.notDownloadedMsg') }}</span>
          <DqButton type="primary" class="studio-alert-inline-btn" @click="$emit('go-download')">
            {{ $tt('studio.goDownload') }}
          </DqButton>
        </template>
      </DqAlert>
    </div>

    <!-- Title input -->
    <div class="video-composer__title-wrap">
      <DqInput
        v-model="localTitle"
        size="sm"
        :placeholder="$tt('studio.workTitlePlaceholder')"
        class="video-composer__title"
      />
    </div>

    <div class="video-composer__long-link">
      <RouterLink :to="{ name: 'long_video_create' }" class="video-composer__long-link-anchor">
        {{ $tt('video.longVideoOpenStudio') }}
      </RouterLink>
    </div>

    <!-- Prompt -->
    <div class="video-composer__prompt-block">
    <div class="video-composer__prompt-wrap">
      <DqInput
        v-model="localPrompt"
        type="textarea"
        :rows="promptRows"
        :placeholder="promptPlaceholder"
        resize="none"
        class="video-composer__prompt"
        @keydown="onKeydown"
      />
      <!-- Reference media inside textarea: start/tail/source video + Bernini refs -->
      <div v-if="showRefArea" class="video-composer__ref-area">
        <div v-if="needsReferenceInput" class="video-composer__ref-slot">
          <div v-if="referenceMedia" class="video-composer__ref-pill">
            <ComposerIconTip :content="primaryRefHoverTip">
              <img
                v-if="referenceMedia.type === 'image'"
                :src="referenceMedia.previewUrl"
                alt=""
                class="video-composer__ref-thumb"
              />
              <video
                v-else-if="referenceMedia.type === 'video'"
                :src="referenceMedia.previewUrl"
                class="video-composer__ref-thumb"
              />
            </ComposerIconTip>
            <ComposerIconTip
              v-if="workMode === 'animate' && referenceMedia.type === 'image' && referenceAssetId"
              :content="$t('create.composerTip.reversePromptVideo')"
            >
              <DqIconButton
                type="text"
                size="xs"
                :disabled="reversing"
                :aria-label="$t('create.reversePromptVideo')"
                @click="$emit('reverse-prompt')"
              >
                <DqIcon :size="10"><Refresh /></DqIcon>
              </DqIconButton>
            </ComposerIconTip>
            <ComposerIconTip :content="$t('create.composerTip.removeRef')">
              <DqIconButton
                type="text"
                size="xs"
                :aria-label="$tt('common.delete')"
                @click="$emit('remove-reference')"
              >
                <DqIcon :size="10"><Close /></DqIcon>
              </DqIconButton>
            </ComposerIconTip>
          </div>
          <ComposerIconTip v-else :content="primaryRefHoverTip">
            <DqIconButton
              type="text"
              size="xs"
              class="video-composer__ref-add"
              :aria-label="referenceMediaLabel"
              @click="$emit('pick-reference')"
            >
              <DqIcon :size="14"><Picture v-if="workMode === 'animate'" /><VideoCamera v-else /></DqIcon>
            </DqIconButton>
          </ComposerIconTip>
        </div>
        <div v-if="workMode === 'animate'" class="video-composer__ref-slot">
          <div v-if="tailReferenceMedia" class="video-composer__ref-pill">
            <ComposerIconTip :content="$t('video.tailFrameTitle')">
              <img :src="tailReferenceMedia.previewUrl" alt="" class="video-composer__ref-thumb" />
            </ComposerIconTip>
            <ComposerIconTip :content="$t('create.composerTip.removeRef')">
              <DqIconButton
                type="text"
                size="xs"
                :aria-label="$tt('common.delete')"
                @click="$emit('remove-tail-reference')"
              >
                <DqIcon :size="10"><Close /></DqIcon>
              </DqIconButton>
            </ComposerIconTip>
          </div>
          <ComposerIconTip v-else :content="$t('create.composerTip.addTailFrame')">
            <DqIconButton
              type="text"
              size="xs"
              class="video-composer__ref-add"
              :aria-label="$tt('video.addTailFrame')"
              @click="$emit('pick-tail-reference')"
            >
              <DqIcon :size="14"><Picture /></DqIcon>
            </DqIconButton>
          </ComposerIconTip>
        </div>
        <template v-if="showBerniniReferences">
          <div
            v-for="(ref, idx) in extraReferenceMedia"
            :key="ref.path"
            class="video-composer__ref-slot"
          >
            <div class="video-composer__ref-pill">
              <ComposerIconTip :content="$t('create.refImgPreview')">
                <img :src="ref.previewUrl" alt="" class="video-composer__ref-thumb" />
              </ComposerIconTip>
              <ComposerIconTip :content="$t('create.composerTip.removeRef')">
                <DqIconButton
                  type="text"
                  size="xs"
                  :aria-label="$tt('common.delete')"
                  @click="$emit('remove-extra-reference', idx)"
                >
                  <DqIcon :size="10"><Close /></DqIcon>
                </DqIconButton>
              </ComposerIconTip>
            </div>
          </div>
          <ComposerIconTip
            v-if="extraReferenceMedia.length < berniniRefMax"
            :content="$t('create.refImgPreview')"
          >
            <DqIconButton
              type="text"
              size="xs"
              class="video-composer__ref-add"
              :aria-label="$t('create.refImage')"
              @click="$emit('pick-extra-reference')"
            >
              <DqIcon :size="14"><Picture /></DqIcon>
            </DqIconButton>
          </ComposerIconTip>
        </template>
      </div>

      <!-- Preset / enhance (prompt corner, same as ImageComposer) -->
      <div class="video-composer__preset-area">
        <ComposerIconTip
          :content="enhanceTip"
        >
          <DqIconButton
            type="text"
            size="xs"
            :class="{ 'video-composer__enhance-btn--busy': enhancing }"
            :disabled="enhancing || !localPrompt.trim()"
            :aria-label="$t('create.enhance')"
            @click="onEnhanceClick"
          >
            <DqIcon :size="12" :class="{ 'video-composer__enhance-spin': enhancing }"><MagicStick /></DqIcon>
          </DqIconButton>
        </ComposerIconTip>
        <DqDropdown
          v-if="styles && Object.keys(styles).length > 0"
          trigger="click"
          size="sm"
          @command="onStyleCommand"
        >
          <DqIconButton
            type="text"
            size="xs"
            class="video-composer__preset-btn"
            :label="$t('create.composerTip.preset')"
          >
            <DqIcon :size="14"><DocumentCopy /></DqIcon>
          </DqIconButton>
          <template #dropdown>
            <DqDropdownMenu>
              <DqDropdownItem
                v-for="(preset, name) in styles"
                :key="name"
                :command="name"
              >
                {{ presetLabel(name, preset) }}
              </DqDropdownItem>
            </DqDropdownMenu>
          </template>
        </DqDropdown>
      </div>
    </div>
    <ComposerPromptApplyStrip
      v-if="promptApplyPreview"
      :preview="promptApplyPreview"
      @replace="$emit('prompt-apply-replace')"
      @append="$emit('prompt-apply-append')"
      @dismiss="$emit('prompt-apply-dismiss')"
    />
    <div v-if="showLongVideoHandoff" class="video-composer__long-video-handoff">
      <span class="video-composer__long-video-handoff-text">{{ $t('video.longVideoHandoffHint') }}</span>
      <DqButton size="sm" type="text" class="video-composer__long-video-handoff-btn" @click="$emit('open-long-video')">
        {{ $t('video.openLongVideoStudio') }}
      </DqButton>
    </div>
    </div>

    <!-- Toolbar -->
    <ComposerToolbar>
      <template #mode>
        <DqSegmented
          :model-value="localWorkMode"
          size="sm"
          :options="workModeOptions"
          @update:model-value="localWorkMode = $event"
        />
      </template>
      <template #controls>
        <DqSelect
          v-model="localModel"
          size="sm"
          variant="ghost"
          
          class="studio-composer-toolbar__select studio-composer-toolbar__select--model"
          
          @change="(val: string) => emit('model-change', val)"
        >
          <DqOption
            v-for="item in modelOptions"
            :key="item.value"
            :label="item.label"
            :value="item.value"
            :disabled="item.disabled"
          >
            <DqTag
              v-if="item.commercialUseAllowed"
              size="sm"
              type="success"
              class="video-composer__model-badge"
            >
              {{ $tt('download.commercialUseBadge') }}
            </DqTag>
          </DqOption>
        </DqSelect>

        <DqSelect
          v-if="sizeOptions.length > 0"
          v-model="localSize"
          size="sm"
          variant="ghost"
          class="studio-composer-toolbar__select studio-composer-toolbar__select--size"
        variant="ghost"
        >
          <DqOption
            v-for="opt in sizeOptions"
            :key="opt.value"
            :value="opt.value"
            :label="formatResolutionOptionLabel(opt)"
          />
        </DqSelect>

        <DqSelect
          v-model="localDuration"
          size="sm"
          variant="ghost"
          
          class="studio-composer-toolbar__select studio-composer-toolbar__select--duration"
        
        >
          <DqOption v-for="opt in durationOptions" :key="opt.value" :label="opt.label" :value="opt.value" />
        </DqSelect>
      </template>
      <template #actions>
        <DqSelect
          v-if="showBatchCount"
          v-model="localBatchCount"
          size="sm"
          variant="ghost"
          
          class="studio-composer-toolbar__select studio-composer-toolbar__select--batch"
        
        >
          <DqOption
            v-for="n in 4"
            :key="n"
            :label="`x${n}`"
            :value="n"
          />
        </DqSelect>

        <DqButton
          type="primary"
          size="sm"
          class="studio-composer-toolbar__generate"
          :disabled="!canGenerate || submitting"
          @click="$emit('generate')"
        >
          <DqIcon :size="14"><MagicStick /></DqIcon>
          <span>{{ primaryActionLabel }}</span>
        </DqButton>
      </template>
    </ComposerToolbar>

    <ComposerAdvancedCollapsible
      v-model:open="advancedOpen"
      :has-custom-params="hasCustomParams || !!tailReferenceMedia"
      @reset-defaults="$emit('reset-defaults')"
    >
      <div class="video-composer__advanced-inner">
        <p v-if="workMode === 'animate'" class="video-composer__tail-hint">
          {{ $tt('video.tailFrameHint') }}
        </p>

        <div v-if="paramSchema.steps" class="video-composer__advanced-row">
            <div class="video-composer__field">
              <label>{{ $tt('create.steps') }}</label>
              <DqSlider
                v-model="localParams.steps"
                :min="paramSchema.steps.min ?? 1"
                :max="paramSchema.steps.max ?? 100"
                :step="paramSchema.steps.step ?? 1"
              />
              <span class="video-composer__field-val">{{ localParams.steps }}</span>
            </div>
            <div v-if="paramSchema.guide_scale" class="video-composer__field">
              <label>{{ $tt('create.guidance') }}</label>
              <DqSlider
                v-model="localParams.guide_scale"
                :min="paramSchema.guide_scale.min ?? 1"
                :max="paramSchema.guide_scale.max ?? 20"
                :step="paramSchema.guide_scale.step ?? 0.1"
              />
              <span class="video-composer__field-val">{{ localParams.guide_scale }}</span>
            </div>
          </div>
          <div v-if="paramSchema.fps || paramSchema.num_frames" class="video-composer__advanced-row">
            <div v-if="paramSchema.fps" class="video-composer__field">
              <label>{{ $tt('create.fps') }}</label>
              <DqSlider
                v-model="localParams.fps"
                :min="paramSchema.fps.min ?? 1"
                :max="paramSchema.fps.max ?? 30"
                :step="paramSchema.fps.step ?? 1"
              />
              <span class="video-composer__field-val">{{ localParams.fps }}</span>
            </div>
            <div v-if="paramSchema.num_frames" class="video-composer__field">
              <label>{{ $tt('create.numFrames') }}</label>
              <span class="video-composer__field-val video-composer__field-val--auto">
                {{ localParams.num_frames }}
              </span>
              <span class="video-composer__frames-formula">
                {{ $tt('video.numFramesFormula', { sec: duration, fps: localParams.fps }) }}
              </span>
            </div>
          </div>
          <p v-if="paramSchema.num_frames?.note" class="video-composer__param-note">
            {{ paramSchema.num_frames.note }}
          </p>
          <div v-if="paramSchema.shift" class="video-composer__advanced-row">
            <div class="video-composer__field">
              <label>{{ $tt('create.shift') }}</label>
              <DqSlider
                v-model="localParams.shift"
                :min="paramSchema.shift.min ?? 0"
                :max="paramSchema.shift.max ?? 20"
                :step="paramSchema.shift.step ?? 0.1"
              />
              <span class="video-composer__field-val">{{ localParams.shift }}</span>
            </div>
            <div v-if="showSeedField" class="video-composer__field">
              <label>{{ $tt('create.seed') }}</label>
              <div class="video-composer__seed-wrap">
                <DqInput v-model="localParams.seed" size="sm" style="width: 100px" :placeholder="$tt('studio.seedPlaceholder')" />
                <DqIconButton
                  type="text"
                  size="xs"
                  :label="$tt('create.randomSeed')"
                  @click="randomizeSeed"
                >
                  <DqIcon :size="12"><Refresh /></DqIcon>
                </DqIconButton>
              </div>
            </div>
          </div>
          <div v-else-if="showSeedField" class="video-composer__advanced-row">
            <div class="video-composer__field">
              <label>{{ $tt('create.seed') }}</label>
              <div class="video-composer__seed-wrap">
                <DqInput v-model="localParams.seed" size="sm" style="width: 100px" :placeholder="$tt('studio.seedPlaceholder')" />
                <DqIconButton
                  type="text"
                  size="xs"
                  :label="$tt('create.randomSeed')"
                  @click="randomizeSeed"
                >
                  <DqIcon :size="12"><Refresh /></DqIcon>
                </DqIconButton>
              </div>
            </div>
          </div>

        <div v-if="showNegativePrompt" class="video-composer__advanced-row">
          <div class="video-composer__field video-composer__field--full">
            <label>{{ $tt('create.negativePrompt') }}</label>
            <DqInput v-model="localParams.negative_prompt" type="textarea" :rows="2" />
          </div>
        </div>

        <div v-if="showLora && compatibleLoras?.length" class="video-composer__advanced-row">
          <div class="video-composer__field video-composer__field--full video-composer__field--stack">
            <div class="video-composer__lora-row">
              <label>{{ $tt('studio.loraLabel') }}</label>
              <DqSelect
                v-model="localParams.lora"
                size="sm"
                clearable
                :placeholder="$tt('studio.noLora')"
                style="flex: 1; max-width: 300px"
              >
                <DqOption
                  v-for="l in compatibleLoras"
                  :key="String(l.id)"
                  :label="videoLoraLabel(l)"
                  :value="String(l.id)"
                />
              </DqSelect>
            </div>
            <p v-if="selectedLoraHintKey" class="video-composer__lora-hint">
              {{ $tt(selectedLoraHintKey) }}
            </p>
            <div v-if="localParams.lora" class="video-composer__lora-row">
              <label>{{ $tt('create.loraScale') }}</label>
              <DqSlider
                v-model="localParams.lora_scale"
                :min="paramSchema.lora_scale?.min ?? 0"
                :max="paramSchema.lora_scale?.max ?? 2"
                :step="paramSchema.lora_scale?.step ?? 0.05"
                style="flex: 1"
              />
              <span class="video-composer__field-val">{{ localParams.lora_scale }}</span>
            </div>
          </div>
        </div>

        <div
          v-for="paramKey in inferenceEnumKeys"
          :key="paramKey"
          class="video-composer__advanced-row video-composer__advanced-row--stack"
        >
          <div class="video-composer__field video-composer__field--full">
            <label>{{ paramFieldLabel(paramKey) }}</label>
            <DqSelect
              v-model="localParams[paramKey]"
              size="sm"
              class="video-composer__select"
            >
              <DqOption
                v-for="opt in enumParamOptions(paramSchema[paramKey])"
                :key="`${paramKey}-${opt}`"
                :value="opt"
                :label="enumOptionLabel(paramKey, opt)"
              />
            </DqSelect>
            <p v-if="paramFieldHint(paramKey)" class="video-composer__param-note">
              {{ paramFieldHint(paramKey) }}
            </p>
          </div>
        </div>
      </div>
    </ComposerAdvancedCollapsible>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue';
import { RouterLink } from 'vue-router';
import ComposerAdvancedCollapsible from './ComposerAdvancedCollapsible.vue';
import ComposerToolbar from './ComposerToolbar.vue';
import ComposerPromptApplyStrip from './ComposerPromptApplyStrip.vue';
import ComposerIconTip from './ComposerIconTip.vue';
import { useI18n } from 'vue-i18n';
import {
  Close,
  DocumentCopy,
  MagicStick,
  Picture,
  Refresh,
  VideoCamera,
} from '@danqing/dq-shell';
import { assetIdFromGalleryPath } from '@/utils/copilotHandoff';
import { $tt, $pn } from '@/utils/i18n';
import { formatResolutionOptionLabel,
  VIDEO_INFERENCE_ENUM_KEYS,
  composerParamHintI18nKey,
  composerParamLabelI18nKey,
  enumOptionI18nKey,
  enumParamOptions,
  normalizeParamsDef,
} from '@/utils/registryParamSchema';
import { appendStyleBoost } from '@/utils/styleBoost';
import { isLongVideoTargetDuration } from '@/utils/videoStoryboardPrompt';
import {
  findCompatibleLora,
  loraHintKey,
  loraOptionLabel,
  type CompatibleLoraRow,
} from '@/utils/loraAdapterMeta';
import {
  supportsBerniniReferenceImages,
} from '@/utils/videoEditSource';

const props = defineProps<{
  modelValue: string;
  title: string;
  workMode: string;
  model: string;
  size: string;
  duration: number;
  composerBusy?: boolean;
  submitting?: boolean;
  generating: boolean;
  canGenerate: boolean;
  generateLabel: string;
  modelOptions: Array<{ label: string; value: string; disabled?: boolean; commercialUseAllowed?: boolean }>;
  sizeOptions: Array<{ label: string; value: string; pixelLabel?: string }>;
  durationOptions: Array<{ label: string; value: number }>;
  styles: Record<string, { applies_to?: string[]; positive?: string; negative?: string; media_scope?: string }>;
  params: {
    steps: number;
    guide_scale: number;
    seed: string;
    fps: number;
    num_frames: number;
    shift: number;
    negative_prompt: string;
    lora?: string;
    lora_scale?: number;
    teacache_mode?: string;
    upscale_scale?: number;
    upscale_denoise?: number;
    upscale_max_frames?: number;
  };
  hasCustomParams: boolean;
  showNegativePrompt: boolean;
  showLora: boolean;
  showBatchCount?: boolean;
  batchCount?: number;
  referenceMedia: { type: 'image' | 'video'; previewUrl: string; label: string } | null;
  referenceAssetPath?: string | null;
  enhancing?: boolean;
  reversing?: boolean;
  tailReferenceMedia?: { type: 'image'; previewUrl: string; label: string } | null;
  extraReferenceMedia?: Array<{ previewUrl: string; path: string; label?: string }>;
  showBerniniReferences?: boolean;
  berniniRefMax?: number;
  modelNotReady?: boolean;
  modelNotReadyName?: string;
  workModeOptions?: Array<{ label: string; value: string }>;
  currentModelConfig?: Record<string, any> | null;
  compatibleLoras?: Record<string, unknown>[];
  collapsed?: boolean;
  promptApplyPreview?: string | null;
}>();

const emit = defineEmits<{
  (e: 'update:modelValue', value: string): void;
  (e: 'update:title', value: string): void;
  (e: 'update:workMode', value: string): void;
  (e: 'update:model', value: string): void;
  (e: 'update:size', value: string): void;
  (e: 'update:duration', value: number): void;
  (e: 'update:params', value: typeof props.params): void;
  (e: 'update:batchCount', value: number): void;
  (e: 'generate'): void;
  (e: 'pick-reference'): void;
  (e: 'remove-reference'): void;
  (e: 'pick-tail-reference'): void;
  (e: 'remove-tail-reference'): void;
  (e: 'pick-extra-reference'): void;
  (e: 'remove-extra-reference', index: number): void;
  (e: 'model-change', value: string): void;
  (e: 'reset-defaults'): void;
  (e: 'go-download'): void;
  (e: 'enhance', ctx?: { stylePositive?: string }): void;
  (e: 'reverse-prompt'): void;
  (e: 'prompt-apply-replace'): void;
  (e: 'prompt-apply-append'): void;
  (e: 'prompt-apply-dismiss'): void;
  (e: 'open-long-video'): void;
}>();

const { t: $t } = useI18n();

const primaryActionLabel = computed(() => {
  if (props.composerBusy) return $t('create.addToBatch');
  if (props.generating) return $t('create.generating');
  return props.generateLabel;
});

const referenceAssetId = computed(() => {
  if (!props.referenceAssetPath) return null;
  return assetIdFromGalleryPath(props.referenceAssetPath);
});

const advancedOpen = ref(false);
const selectedStyle = ref('');
const lastStylePositive = ref('');

const localPrompt = computed({
  get: () => props.modelValue,
  set: (v) => emit('update:modelValue', v),
});

const localTitle = computed({
  get: () => props.title,
  set: (v) => emit('update:title', v),
});

const localWorkMode = computed({
  get: () => props.workMode,
  set: (v) => emit('update:workMode', v),
});

const workModeOptions = computed(() => props.workModeOptions || [
  { label: $tt('video.modeCreate'), value: 'create' },
  { label: $tt('video.modeAnimate'), value: 'animate' },
  { label: $tt('video.modeEdit'), value: 'edit' },
]);

const localModel = computed({
  get: () => props.model,
  set: (v) => emit('update:model', v),
});

const localSize = computed({
  get: () => props.size,
  set: (v) => emit('update:size', v),
});

const localDuration = computed({
  get: () => props.duration,
  set: (v) => emit('update:duration', v),
});

const localParams = computed({
  get: () => props.params,
  set: (v) => emit('update:params', v),
});

function patchParams(patch: Partial<typeof props.params>) {
  Object.assign(props.params, patch);
}

function randomizeSeed() {
  patchParams({ seed: String(Math.floor(Math.random() * 1_000_000)) });
}

const localBatchCount = computed({
  get: () => props.batchCount ?? 1,
  set: (v) => emit('update:batchCount', v),
});

const needsReferenceInput = computed(
  () => props.workMode === 'animate' || props.workMode === 'edit',
);

const showBerniniReferences = computed(
  () => Boolean(props.showBerniniReferences) && supportsBerniniReferenceImages(paramSchema.value),
);

const showRefArea = computed(
  () => needsReferenceInput.value || showBerniniReferences.value,
);

const extraReferenceMedia = computed(
  () => props.extraReferenceMedia || [],
);

const berniniRefMax = computed(() => props.berniniRefMax ?? 5);

const paramSchema = computed(() =>
  normalizeParamsDef(props.currentModelConfig?.parameters as Record<string, unknown> | undefined),
);

const inferenceEnumKeys = computed(() =>
  VIDEO_INFERENCE_ENUM_KEYS.filter((key) => Boolean(paramSchema.value[key])),
);

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

function videoLoraLabel(l: Record<string, unknown>) {
  return loraOptionLabel(l as CompatibleLoraRow);
}

const selectedLoraHintKey = computed(() => {
  const id = String(localParams.value.lora || '');
  if (!id) return null;
  const row = findCompatibleLora((props.compatibleLoras || []) as CompatibleLoraRow[], id);
  return loraHintKey(row);
});

const longVideoSupport = computed(() => Boolean(paramSchema.value.long_video_support));

const isLongDuration = computed(
  () => props.workMode === 'create' && isLongVideoTargetDuration(localDuration.value, longVideoSupport.value),
);

const showLongVideoHandoff = computed(() => isLongDuration.value);

const promptRows = computed(() => (isLongDuration.value ? 8 : 5));

const promptPlaceholder = computed(() =>
  isLongDuration.value ? $tt('video.promptPlaceholderLong') : $tt('video.promptPlaceholder'),
);

const showSeedField = computed(() => paramSchema.value.seed_support !== undefined);

const referenceMediaLabel = computed(() => {
  if (props.workMode === 'edit') return $tt('video.editSourceTitle');
  if (props.workMode === 'animate') return $tt('action.video.startImage');
  return $tt('create.refImage');
});

const primaryRefHoverTip = computed(() => {
  if (props.workMode === 'edit') return $tt('video.editSourceTitle');
  if (props.workMode === 'animate') return $tt('video.composerStartFrame');
  return $t('create.refImgPreview');
});

const shortcutHint = computed(() => {
  const isMac = navigator.platform.toLowerCase().includes('mac');
  return isMac ? '⌘ + Enter ' + $tt('create.sendShortcutHintMac') : 'Ctrl + Enter ' + $tt('create.sendShortcutHintWin');
});

function onStyleCommand(command: string) {
  onStyleChange(command);
}

function onEnhanceClick() {
  const style = lastStylePositive.value.trim();
  emit('enhance', style ? { stylePositive: style } : undefined);
}

const enhanceTip = computed(() => {
  if (props.enhancing) return $t('create.enhancing');
  if (!localPrompt.value.trim()) return $t('create.composerTip.enhanceEmpty');
  return $t('create.composerTip.enhance');
});

function onStyleChange(name: string) {
  if (!name || !props.styles[name]) return;
  const preset = props.styles[name];
  if (preset.positive) {
    lastStylePositive.value = String(preset.positive);
    localPrompt.value = appendStyleBoost(localPrompt.value || '', String(preset.positive));
  }
  if (preset.negative && props.showNegativePrompt) {
    const current = localParams.value.negative_prompt || '';
    patchParams({
      negative_prompt: current
        ? current + '\n' + preset.negative
        : preset.negative,
    });
  }
  selectedStyle.value = '';
}

function presetLabel(name: string, preset: Record<string, unknown>): string {
  const a = (preset.applies_to as string[]) || [];
  const hasC = a.includes('create');
  const hasA = a.includes('animate');
  const hasU = a.includes('upscale');
  let tag = '';
  if (hasC && hasA) tag = $tt('video.presetTagHybrid');
  else if (hasC && !hasA) tag = $tt('video.presetTagT2V');
  else if (hasA && !hasC) tag = $tt('video.presetTagI2V');
  else if (hasU && !hasC && !hasA) tag = $tt('video.presetTagUpscale');
  const display = $pn(preset as { name?: string | { zh?: string; en?: string } }, name);
  return tag ? `${tag} ${display}` : display;
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
    e.preventDefault();
    if (!props.submitting && props.canGenerate) {
      emit('generate');
    }
  }
}
</script>

<style scoped>
.video-composer {
  border-radius: var(--dq-radius-group);
  padding: 14px 16px 12px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  width: 100%;
  min-width: 0;
  container-type: inline-size;
  box-sizing: border-box;
}

/* Model meta */
.video-composer__model-meta {
  margin-bottom: 8px;
}

/* Title */
.video-composer__title-wrap {
  margin-bottom: 8px;
}

.video-composer__title :deep(.dq-input) {
  border-radius: var(--dq-radius-input);
}

/* Prompt */
.video-composer__prompt-block {
  display: flex;
  flex-direction: column;
  margin-bottom: 8px;
  flex-shrink: 0;
}

.video-composer__long-video-handoff {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px 10px;
  margin-top: 6px;
  padding: 6px 10px;
  border-radius: var(--dq-radius-control-sm);
  background: var(--dq-color-fill-secondary, rgba(128, 128, 128, 0.08));
  font-size: var(--dq-font-size-caption, 12px);
  color: var(--dq-color-text-secondary);
}

.video-composer__long-video-handoff-btn {
  flex-shrink: 0;
}

.video-composer__prompt-wrap {
  position: relative;
}

.video-composer__prompt-head {
  display: flex;
  justify-content: flex-end;
  margin-bottom: 2px;
}

.video-composer__prompt :deep(.dq-input--textarea) {
  border-radius: var(--dq-radius-group);
  font-size: var(--dq-font-size-body);
  line-height: 1.5;
  padding: 10px 12px 32px;
  min-height: 7.5rem;
  max-height: 14rem;
  resize: none;
  overflow-y: auto;
  transition: border-color 0.15s ease, background 0.15s ease;
}

.video-composer--expanded-prompt .video-composer__prompt :deep(.dq-input--textarea) {
  min-height: 8rem;
  max-height: 14rem;
}

/* Reference media area inside textarea */
.video-composer__ref-area {
  position: absolute;
  bottom: 6px;
  left: 8px;
  right: 40px;
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}

.video-composer__ref-slot {
  display: flex;
  align-items: center;
}

.video-composer__ref-pill {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 2px 6px 2px 2px;
  background: var(--dq-fill-secondary);
  border-radius: 6px;
  border: 1px solid var(--dq-border-subtle);
}

.video-composer__ref-thumb {
  display: block;
  width: 20px;
  height: 20px;
  object-fit: cover;
  border-radius: 3px;
}

.video-composer__ref-add {
  opacity: 0.6;
  transition: opacity 0.2s;
}

.video-composer__ref-add:hover {
  opacity: 1;
}

/* Preset picker inside textarea (bottom-right) */
.video-composer__preset-area {
  position: absolute;
  bottom: 6px;
  right: 8px;
  z-index: 1;
  display: flex;
  align-items: center;
}

.video-composer__preset-btn {
  opacity: 0.6;
  transition: opacity 0.2s;
}

.video-composer__preset-btn:hover {
  opacity: 1;
}

.video-composer__enhance-btn--busy {
  opacity: 1;
}

.video-composer__enhance-spin {
  animation: video-composer-enhance-spin 0.9s linear infinite;
}

@keyframes video-composer-enhance-spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.video-composer__size-ratio {
  font-variant-numeric: tabular-nums;
}

.video-composer__size-aspect {
  margin-left: 8px;
  opacity: 0.65;
  font-size: var(--dq-font-size-caption);
}

/* Advanced panel (inside drawer) */
.video-composer__advanced-inner {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.video-composer__advanced-row {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}

.video-composer__field {
  display: flex;
  align-items: center;
  gap: 10px;
  flex: 1;
  min-width: 260px;
  overflow: hidden;
}

.video-composer__field--full {
  flex: 1 1 100%;
  flex-direction: column;
  align-items: flex-start;
  gap: 6px;
}

.video-composer__field label {
  font-size: var(--dq-font-size-caption);
  font-weight: 500;
  color: var(--dq-label-secondary);
  white-space: nowrap;
  width: 90px;
  flex-shrink: 0;
  overflow: hidden;
  text-overflow: ellipsis;
}

.video-composer__field :deep(.dq-slider) {
  flex: 1;
}

.video-composer__field-val {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
  min-width: 32px;
  text-align: right;
  font-variant-numeric: tabular-nums;
}

.video-composer__field-val--auto {
  min-width: 40px;
  font-weight: 600;
  color: var(--dq-label-primary);
}

.video-composer__frames-formula {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
  white-space: nowrap;
}

.video-composer__field--stack {
  flex-direction: column;
  align-items: flex-start;
  gap: 8px;
}

.video-composer__lora-row {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
}

.video-composer__lora-hint {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
  color: var(--dq-label-tertiary);
}

.video-composer__seed-wrap {
  display: flex;
  align-items: center;
  gap: 4px;
}

.video-composer__tail-hint,
.video-composer__param-note {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
  line-height: 1.4;
}

/* Hint - compact inline */
.video-composer__hint {
  text-align: right;
  margin-top: 2px;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
  opacity: 0;
  transition: opacity 0.2s;
  height: 0;
  overflow: hidden;
}

.video-composer__prompt-wrap:focus-within + .video-composer__hint,
.video-composer__prompt-wrap:hover + .video-composer__hint {
  opacity: 1;
  height: auto;
}

.video-composer__long-link {
  padding: 6px 12px 0;
}

.video-composer__long-link-anchor {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-accent);
  text-decoration: none;
}

.video-composer__long-link-anchor:hover {
  text-decoration: underline;
}
</style>
