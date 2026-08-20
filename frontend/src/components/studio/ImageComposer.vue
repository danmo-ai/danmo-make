<template>
  <div
    class="image-composer studio-composer-shell dq-glass--panel"
    :class="{ 'image-composer--embedded': embedded }"
  >
    <!-- Top: Title -->
    <DqInput
      v-if="!embedded"
      v-model="localTitle"
      size="sm"
      :placeholder="$t('studio.workTitlePlaceholder')"
      class="image-composer__title"
    />

    <!-- Middle: Prompt input -->
    <div class="image-composer__prompt-block">
    <div class="image-composer__prompt-wrap">
      <DqInput
        v-model="localPrompt"
        type="textarea"
        :rows="5"
        :placeholder="$t('create.promptPlaceholder')"
        resize="vertical"
        class="image-composer__prompt"
        @keydown.meta.enter.prevent="$emit('generate')"
        @keydown.ctrl.enter.prevent="$emit('generate')"
      />

      <!-- Bottom-left: Reference image -->
      <div class="image-composer__ref-area">
        <div v-if="referenceImage" class="image-composer__ref-pill">
          <img :src="referenceImage.previewUrl" alt="ref" />
          <ComposerIconTip
            v-if="referenceAssetId"
            :content="$t('create.composerTip.reversePrompt')"
          >
            <DqIconButton
              type="text"
              size="sm"
              :disabled="reversing"
              :aria-label="$t('create.reversePrompt')"
              @click="$emit('reverse-prompt')"
            >
              <DqIcon :size="10"><Refresh /></DqIcon>
            </DqIconButton>
          </ComposerIconTip>
          <ComposerIconTip :content="$t('create.composerTip.removeRef')">
            <DqIconButton
              type="text"
              size="sm"
              :aria-label="$t('common.delete')"
              @click="$emit('remove-reference')"
            >
              <DqIcon :size="10"><Close /></DqIcon>
            </DqIconButton>
          </ComposerIconTip>
        </div>
        <ComposerIconTip v-else :content="$t('create.composerTip.refImage')">
          <DqIconButton
            type="text"
            size="sm"
            class="image-composer__ref-add"
            :aria-label="$t('create.refImage')"
            @click="$emit('pick-reference')"
          >
            <DqIcon :size="14"><Picture /></DqIcon>
          </DqIconButton>
        </ComposerIconTip>
      </div>

      <!-- Bottom-right: Preset / Style picker -->
      <div class="image-composer__preset-area">
        <ComposerIconTip
          :content="enhanceTip"
        >
          <DqIconButton
            class="image-composer__preset-area__enhance-btn"
            :class="{ 'image-composer__enhance-btn--busy': enhancing }"
            type="text"
            size="sm"
            :disabled="enhancing || !localPrompt.trim()"
            :aria-label="$t('create.enhance')"
            @click="onEnhanceClick"
          >
            <DqIcon :size="12" :class="{ 'image-composer__enhance-spin': enhancing }"><MagicStick /></DqIcon>
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
            size="sm"
            class="image-composer__preset-btn"
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
    <ComposerSuccessorHintStrip
      v-if="successorHint"
      :successor-name="successorHint.successorName"
      @switch="$emit('successor-switch')"
      @dismiss="$emit('successor-dismiss')"
    />
    </div>

    <!-- Toolbar -->
    <ComposerToolbar>
      <template #mode>
        <DqSegmented
          v-if="modeOptions && modeOptions.length > 0"
          v-model="localMode"
          size="sm"
          :options="modeOptions"
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
              class="image-composer__model-badge"
            >
              {{ $t('download.commercialUseBadge') }}
            </DqTag>
          </DqOption>
        </DqSelect>

        <DqSelect
          v-if="!isImg2imgMode && !lockSize"
          v-model="localSize"
          size="sm"
          variant="ghost"
          
          class="studio-composer-toolbar__select studio-composer-toolbar__select--size"
        
        >
          <DqOption
            v-for="opt in sizeOptions"
            :key="opt.value"
            :value="opt.value"
            :label="formatResolutionOptionLabel(opt)"
          />
        </DqSelect>
      </template>
      <template #actions>
        <DqSelect
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
      :has-custom-params="hasCustomParams"
      @reset-defaults="$emit('reset-defaults')"
    >
      <ImageComposerAdvancedFields
        stacked
        :params="params"
        :model="model"
        :mode="localMode"
        :reference-image="referenceImage"
        :current-model-config="currentModelConfig"
        :compatible-loras="compatibleLoras"
        :compatible-control-nets="compatibleControlNets"
        :control-net-runtime-available="controlNetRuntimeAvailable"
        :control-image="controlImage"
        :inpaint-source-image="inpaintSourceImage"
        :inpaint-mask-image="inpaintMaskImage"
        :show-negative-prompt="showNegativePrompt"
        @pick-control="$emit('pick-control')"
        @remove-control="$emit('remove-control')"
        @pick-inpaint-source="$emit('pick-inpaint-source')"
        @remove-inpaint-source="$emit('remove-inpaint-source')"
        @pick-inpaint-mask="$emit('pick-inpaint-mask')"
        @remove-inpaint-mask="$emit('remove-inpaint-mask')"
      />
    </ComposerAdvancedCollapsible>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, withDefaults } from 'vue';
import { useComposerDefaultModel } from '@/composables/useComposerDefaultModel';
import ComposerAdvancedCollapsible from './ComposerAdvancedCollapsible.vue';
import ComposerToolbar from './ComposerToolbar.vue';
import ImageComposerAdvancedFields from './ImageComposerAdvancedFields.vue';
import ComposerPromptApplyStrip from './ComposerPromptApplyStrip.vue';
import ComposerSuccessorHintStrip from './ComposerSuccessorHintStrip.vue';
import ComposerIconTip from './ComposerIconTip.vue';
import { useI18n } from 'vue-i18n';
import {
  Close,
  DocumentCopy,
  MagicStick,
  Picture,
} from '@danqing/dq-shell';
import { assetIdFromGalleryPath } from '@/utils/copilotHandoff';
import { $tt } from '@/utils/i18n';
import { formatResolutionOptionLabel } from '@/utils/registryParamSchema';
import { appendStyleBoost } from '@/utils/styleBoost';

const props = withDefaults(defineProps<{
  modelValue: string;
  title?: string;
  model: string;
  size: string;
  batchCount: number;
  composerBusy?: boolean;
  submitting?: boolean;
  generating: boolean;
  canGenerate: boolean;
  modelOptions: Array<{ label: string; value: string; disabled?: boolean; commercialUseAllowed?: boolean }>;
  sizeOptions: Array<{ label: string; value: string; pixelLabel?: string }>;
  styles: Record<string, { applies_to?: string[]; positive?: string; negative?: string; trigger_words?: string; media_scope?: string }>;
  params: {
    steps: number;
    guidance: number;
    seed: string;
    strength: number;
    negative_prompt: string;
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
  hasCustomParams: boolean;
  showNegativePrompt: boolean;
  referenceImage: { previewUrl: string; path: string } | null;
  controlImage?: { previewUrl: string; path: string } | null;
  inpaintSourceImage?: { previewUrl: string; path: string } | null;
  inpaintMaskImage?: { previewUrl: string; path: string } | null;
  /**
   * Optional mode selector. If provided, a segmented control is shown in the
   * toolbar for switching between text2img / img2img modes.
   */
  mode?: string;
  modeOptions?: Array<{ label: string; value: string }>;
  currentModelConfig?: Record<string, any> | null;
  compatibleLoras?: Record<string, unknown>[];
  compatibleControlNets?: Record<string, unknown>[];
  controlNetRuntimeAvailable?: boolean;
  enhancing?: boolean;
  reversing?: boolean;
  /** Inspector embed: hide title row. */
  embedded?: boolean;
  /** Hide size selector (output size fixed by parent). */
  lockSize?: boolean;
  /** Override primary generate button label. */
  generateLabel?: string;
  promptApplyPreview?: string | null;
  successorHint?: { successorId: string; successorName: string } | null;
}>(), {
  composerBusy: false,
  submitting: false,
  embedded: false,
  lockSize: false,
});

const emit = defineEmits<{
  (e: 'update:modelValue', value: string): void;
  (e: 'update:title', value: string): void;
  (e: 'update:model', value: string): void;
  (e: 'update:size', value: string): void;
  (e: 'update:batchCount', value: number): void;
  (e: 'update:params', value: typeof props.params): void;
  (e: 'update:mode', value: string): void;
  (e: 'generate'): void;
  (e: 'pick-reference'): void;
  (e: 'remove-reference'): void;
  (e: 'pick-control'): void;
  (e: 'remove-control'): void;
  (e: 'pick-inpaint-source'): void;
  (e: 'remove-inpaint-source'): void;
  (e: 'pick-inpaint-mask'): void;
  (e: 'remove-inpaint-mask'): void;
  (e: 'model-change', value: string): void;
  (e: 'reset-defaults'): void;
  (e: 'enhance', ctx?: { stylePositive?: string }): void;
  (e: 'reverse-prompt'): void;
  (e: 'prompt-apply-replace'): void;
  (e: 'prompt-apply-append'): void;
  (e: 'prompt-apply-dismiss'): void;
  (e: 'successor-switch'): void;
  (e: 'successor-dismiss'): void;
}>();

const { t: $t } = useI18n();

const primaryActionLabel = computed(() => {
  if (props.generateLabel) return props.generateLabel;
  if (props.composerBusy) return $t('create.addToBatch');
  if (props.generating) return $t('create.generating');
  return $t('studio.generate');
});

const referenceAssetId = computed(() => {
  if (!props.referenceImage?.path) return null;
  return assetIdFromGalleryPath(props.referenceImage.path);
});

const localParams = computed({
  get: () => props.params,
  set: (v) => emit('update:params', v),
});

const controlNetRuntimeAvailable = computed(
  () => props.controlNetRuntimeAvailable !== false,
);

const advancedOpen = ref(false);
const selectedStyle = ref('');
const lastStylePositive = ref('');

const isImg2imgMode = computed(
  () => localMode.value === 'img2img' || props.referenceImage != null,
);

const localTitle = computed({
  get: () => props.title || '',
  set: (v) => emit('update:title', v),
});

const localMode = computed({
  get: () => props.mode || 'text2img',
  set: (v) => emit('update:mode', v),
});

const localPrompt = computed({
  get: () => props.modelValue,
  set: (v) => emit('update:modelValue', v),
});

const localModel = computed({
  get: () => props.model,
  set: (v) => emit('update:model', v),
});

useComposerDefaultModel(
  () => props.modelOptions || [],
  () => props.model,
  (value) => {
    emit('update:model', value);
    emit('model-change', value);
  },
);

const localSize = computed({
  get: () => props.size,
  set: (v) => emit('update:size', v),
});

const localBatchCount = computed({
  get: () => props.batchCount,
  set: (v) => emit('update:batchCount', v),
});

function patchParams(patch: Partial<typeof props.params>) {
  Object.assign(props.params, patch);
}

const shortcutHint = computed(() => {
  const isMac = navigator.platform.toLowerCase().includes('mac');
  return isMac ? '⌘ + Enter ' + $t('create.sendShortcutHintMac') : 'Ctrl + Enter ' + $t('create.sendShortcutHintWin');
});

const enhanceTip = computed(() => {
  if (props.enhancing) return $t('create.enhancing');
  if (!localPrompt.value.trim()) return $t('create.composerTip.enhanceEmpty');
  return $t('create.composerTip.enhance');
});

function onEnhanceClick() {
  const style = lastStylePositive.value.trim();
  emit('enhance', style ? { stylePositive: style } : undefined);
}

function onStyleChange(name: string) {
  if (!name || !props.styles[name]) return;
  const preset = props.styles[name];
  if (preset.positive) {
    lastStylePositive.value = String(preset.positive);
    localPrompt.value = appendStyleBoost(localPrompt.value || '', String(preset.positive));
  }
  if (preset.trigger_words) {
    const tw = String(preset.trigger_words).trim();
    if (tw) {
      localPrompt.value = localPrompt.value
        ? localPrompt.value + '\n' + tw
        : tw;
    }
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
  const hasEdit = a.some((x: string) => ['rewrite', 'retouch', 'extend'].includes(x));
  let tag = '';
  if (hasC && !hasEdit) tag = $tt('create.presetTagT2I');
  else if (hasEdit && !hasC) tag = $tt('create.presetTagI2I');
  return tag ? `${tag} ${name}` : name;
}

function onLoadPreset() {
  if (!selectedStyle.value || !props.styles[selectedStyle.value]) return;
  onStyleChange(selectedStyle.value);
  selectedStyle.value = '';
}

function onStyleCommand(command: string) {
  onStyleChange(command);
}
</script>

<style scoped>
.image-composer {
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

/* Title */
.image-composer__title {
  margin: 0;
}

.image-composer__title :deep(.dq-input) {
  border-radius: var(--dq-radius-input);
}

/* Prompt */
.image-composer__prompt-block {
  display: flex;
  flex-direction: column;
}

.image-composer__prompt-wrap {
  position: relative;
  margin: 0;
}

.image-composer__prompt :deep(.dq-input--textarea) {
  border-radius: var(--dq-radius-group);
  font-size: var(--dq-font-size-body);
  line-height: 1.5;
  padding: 10px 12px 32px;
  min-height: 4.5rem;
  max-height: 4.5rem;
  resize: none;
  overflow-y: auto;
  transition: border-color 0.15s ease, background 0.15s ease;
}

/* Reference image area inside textarea */
.image-composer__ref-area {
  position: absolute;
  bottom: 6px;
  left: 8px;
  z-index: 1;
  display: flex;
  align-items: center;
}

.image-composer__ref-pill {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 2px 6px 2px 2px;
  background: var(--dq-fill-secondary);
  border-radius: 6px;
  border: 1px solid var(--dq-border-subtle);
}

.image-composer__ref-pill img {
  width: 20px;
  height: 20px;
  object-fit: cover;
  border-radius: 3px;
}

.image-composer__ref-add {
  opacity: 0.6;
  transition: opacity 0.2s;
}

.image-composer__ref-add:hover {
  opacity: 1;
}

.image-composer__control-image-row {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
}

.image-composer__ref-pill--compact img {
  width: 32px;
  height: 32px;
}

.image-composer__control-hint {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  line-height: 1.4;
  color: var(--dq-label-tertiary);
}

.image-composer__control-hint--sub {
  opacity: 0.85;
}

/* Preset picker inside textarea (bottom-right) */
.image-composer__preset-area {
  position: absolute;
  bottom: 6px;
  right: 8px;
  z-index: 1;
  display: flex;
  align-items: center;
}

.image-composer__preset-btn {
  opacity: 0.6;
  transition: opacity 0.2s;
}

.image-composer__preset-btn:hover {
  opacity: 1;
}

.image-composer__enhance-btn--busy {
  opacity: 1;
}

.image-composer__enhance-spin {
  animation: image-composer-enhance-spin 0.9s linear infinite;
}

@keyframes image-composer-enhance-spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.image-composer__model-badge {
  font-size: var(--dq-font-size-caption);
  padding: 0 4px;
  height: 16px;
  line-height: 16px;
}

/* Size option with ratio + pixel */
.image-composer__size-ratio {
  font-weight: 500;
}

.image-composer__size-pixel {
  margin-left: 6px;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
  opacity: 0.8;
}

/* Advanced panel (inside drawer) */
.image-composer__advanced-inner {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.image-composer__advanced-row {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}

.image-composer__field {
  display: flex;
  align-items: center;
  gap: 10px;
  flex: 1;
  min-width: 260px;
  overflow: hidden;
}

.image-composer__field--full {
  flex: 1 1 100%;
  flex-direction: column;
  align-items: flex-start;
  gap: 6px;
}

 .image-composer__field label {
  font-size: var(--dq-font-size-caption);
  font-weight: 500;
  color: var(--dq-label-secondary);
  white-space: nowrap;
  width: 90px;
  flex-shrink: 0;
  overflow: hidden;
  text-overflow: ellipsis;
}

.image-composer__field :deep(.dq-slider) {
  flex: 1;
}

.image-composer__field-val {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
  min-width: 32px;
  text-align: right;
  font-variant-numeric: tabular-nums;
}

.image-composer__seed-wrap {
  display: flex;
  align-items: center;
  gap: 4px;
}

/* Hint - compact inline */
.image-composer__prompt :deep(.dq-input--textarea) {
  min-height: 7.5rem;
  max-height: 14rem;
}
</style>
