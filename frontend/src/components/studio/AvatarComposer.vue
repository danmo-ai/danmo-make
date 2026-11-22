<template>
  <div class="avatar-composer studio-composer-shell dq-glass--panel">
    <DqInput
      v-model="localTitle"
      size="sm"
      :placeholder="$tt('studio.workTitlePlaceholder')"
      class="avatar-composer__title"
    />

    <DqSegmented
      v-if="modeOptions.length > 0"
      v-model="localMode"
      block
      size="sm"
      :options="modeOptions"
      class="studio-composer-setup__mode"
    />

    <!-- Prompt -->
    <div class="avatar-composer__prompt-block">
      <div class="avatar-composer__prompt-wrap">
        <DqInput
          v-model="localPrompt"
          type="textarea"
          :rows="5"
          :placeholder="$tt('avatar.promptPlaceholder')"
          resize="none"
          class="avatar-composer__prompt"
          @keydown="onPromptKeydown"
        />

        <!-- Reference inputs inside the textarea, same as image / video composers -->
        <div class="avatar-composer__ref-area">
          <div v-if="portraitPreview" class="avatar-composer__ref-pill">
            <ComposerIconTip :content="portraitLabel || $tt('avatar.portrait')">
              <img :src="portraitPreview" alt="" class="avatar-composer__ref-thumb" />
            </ComposerIconTip>
            <ComposerIconTip :content="$tt('create.composerTip.removeRef')">
              <DqIconButton
                type="text"
                size="sm"
                :aria-label="$tt('common.delete')"
                @click="$emit('remove-portrait')"
              >
                <DqIcon :size="10"><Close /></DqIcon>
              </DqIconButton>
            </ComposerIconTip>
          </div>
          <ComposerIconTip v-else :content="$tt('avatar.portrait')">
            <DqIconButton
              type="text"
              size="sm"
              class="avatar-composer__ref-add"
              :aria-label="$tt('avatar.pickPortrait')"
              @click="$emit('pick-portrait')"
            >
              <DqIcon :size="14"><Picture /></DqIcon>
            </DqIconButton>
          </ComposerIconTip>

          <template v-if="!isScriptMode">
            <div v-if="audioLabel" class="avatar-composer__ref-pill">
              <DqIcon :size="12"><Microphone /></DqIcon>
              <ComposerIconTip :content="audioLabel">
                <span class="avatar-composer__ref-label">{{ $tt('avatar.voiceTrack') }}</span>
              </ComposerIconTip>
              <ComposerIconTip :content="$tt('create.composerTip.removeRef')">
                <DqIconButton
                  type="text"
                  size="sm"
                  :aria-label="$tt('common.delete')"
                  @click="$emit('remove-audio')"
                >
                  <DqIcon :size="10"><Close /></DqIcon>
                </DqIconButton>
              </ComposerIconTip>
            </div>
            <ComposerIconTip v-else :content="$tt('avatar.voiceTrack')">
              <DqIconButton
                type="text"
                size="sm"
                class="avatar-composer__ref-add"
                :aria-label="$tt('avatar.pickAudio')"
                @click="$emit('pick-audio')"
              >
                <DqIcon :size="14"><Microphone /></DqIcon>
              </DqIconButton>
            </ComposerIconTip>
          </template>
        </div>
      </div>
    </div>

    <!-- Script (念台词): primary creative input, mirrors lyrics in the audio composer -->
    <div v-if="isScriptMode" class="avatar-composer__script-block">
      <div class="avatar-composer__script-head">
        <span class="avatar-composer__script-label">{{ $tt('avatar.script') }}</span>
      </div>
      <DqInput
        v-model="localScriptText"
        type="textarea"
        :rows="4"
        :placeholder="$tt('avatar.scriptPlaceholder')"
        resize="none"
        class="avatar-composer__script"
      />
      <p class="avatar-composer__script-hint">{{ $tt('avatar.scriptHint') }}</p>
    </div>

    <div class="studio-composer-setup">
      <ModelSelect
        v-model="localModel"
        :options="modelOptions"
        variant="ghost"
        class="studio-composer-toolbar__select studio-composer-toolbar__select--model"
        @change="$emit('model-change')"
      />
      <div v-if="resolutionOptions.length > 0" class="studio-composer-setup__basics">
        <DqSelect
          v-model="localResolution"
          size="sm"
          variant="ghost"
          class="studio-composer-toolbar__select studio-composer-toolbar__select--size"
        >
          <DqOption
            v-for="opt in resolutionOptions"
            :key="opt.value"
            :value="opt.value"
            :label="formatResolutionOptionLabel(opt)"
          />
        </DqSelect>
      </div>
    </div>

    <ComposerToolbar>
      <template #actions>
        <DqButton
          type="primary"
          size="sm"
          class="studio-composer-toolbar__generate"
          :loading="generating"
          :disabled="!canGenerate"
          @click="$emit('generate')"
        >
          <DqIcon :size="14"><MagicStick /></DqIcon>
          <span>{{ generateLabel }}</span>
        </DqButton>
      </template>
    </ComposerToolbar>

    <ComposerAdvancedCollapsible
      v-model:open="advancedOpen"
      :has-custom-params="hasCustomParams"
      @reset-defaults="$emit('reset-defaults')"
    >
      <div class="avatar-composer__advanced-inner">
        <div v-if="paramSchema.steps || paramSchema.fps" class="avatar-composer__advanced-row">
          <div v-if="paramSchema.steps" class="avatar-composer__field">
            <label>{{ $tt('create.steps') }}</label>
            <DqSlider
              :model-value="Number(params.steps)"
              :min="paramSchema.steps.min ?? 1"
              :max="paramSchema.steps.max ?? 50"
              :step="paramSchema.steps.step ?? 1"
              @update:model-value="patchParams({ steps: $event })"
            />
            <span class="avatar-composer__field-val">{{ params.steps }}</span>
          </div>
          <div v-if="paramSchema.fps" class="avatar-composer__field">
            <label>{{ $tt('create.fps') }}</label>
            <DqSlider
              :model-value="Number(params.fps)"
              :min="paramSchema.fps.min ?? 15"
              :max="paramSchema.fps.max ?? 30"
              :step="paramSchema.fps.step ?? 1"
              @update:model-value="patchParams({ fps: $event })"
            />
            <span class="avatar-composer__field-val">{{ params.fps }}</span>
          </div>
        </div>

        <div v-if="paramSchema.num_frames || showSeedField" class="avatar-composer__advanced-row">
          <div v-if="paramSchema.num_frames" class="avatar-composer__field">
            <label>{{ $tt('create.numFrames') }}</label>
            <DqInputNumber
              :model-value="Number(params.num_frames)"
              size="sm"
              :min="paramSchema.num_frames.min ?? 25"
              :max="paramSchema.num_frames.max ?? 121"
              :step="paramSchema.num_frames.step ?? 1"
              @update:model-value="patchParams({ num_frames: $event })"
            />
            <span class="avatar-composer__frames-formula">{{ secondsLabel }}</span>
          </div>
          <div v-if="showSeedField" class="avatar-composer__field">
            <label>{{ $tt('create.seed') }}</label>
            <div class="avatar-composer__seed-wrap">
              <DqInput
                :model-value="params.seed == null ? '' : String(params.seed)"
                size="sm"
                :placeholder="$tt('studio.seedPlaceholder')"
                @update:model-value="patchParams({ seed: $event ? Number($event) : null })"
              />
              <DqIconButton
                type="text"
                size="sm"
                :label="$tt('create.randomSeed')"
                @click="randomizeSeed"
              >
                <DqIcon :size="12"><Refresh /></DqIcon>
              </DqIconButton>
            </div>
          </div>
        </div>

        <div v-if="showNegativePrompt" class="avatar-composer__advanced-row">
          <div class="avatar-composer__field avatar-composer__field--full">
            <label>{{ $tt('create.negativePrompt') }}</label>
            <DqInput
              :model-value="String(params.negative_prompt ?? '')"
              type="textarea"
              :rows="2"
              :placeholder="$tt('create.negativePlaceholder')"
              resize="none"
              @update:model-value="patchParams({ negative_prompt: $event })"
            />
          </div>
        </div>
      </div>
    </ComposerAdvancedCollapsible>
  </div>
</template>

<script setup lang="ts">
// @ts-nocheck — registry param schema is dynamic
import { ref, computed } from 'vue';
import { useComposerDefaultModel } from '@/composables/useComposerDefaultModel';
import { Close, MagicStick, Microphone, Picture, Refresh } from '@danqing/dq-shell';
import ComposerAdvancedCollapsible from './ComposerAdvancedCollapsible.vue';
import ComposerIconTip from './ComposerIconTip.vue';
import ComposerToolbar from './ComposerToolbar.vue';
import { DqSegmented } from '@danqing/dq-ui';
import { $tt } from '@/utils/i18n';
import { formatResolutionOptionLabel } from '@/utils/registryParamSchema';
import ModelSelect from '@/components/model/ModelSelect.vue';
import type { ModelSelectOption } from '@/utils/modelSelectOption';

const props = withDefaults(
  defineProps<{
    modelValue: string;
    title: string;
    model: string;
    resolution: string;
    mode: 'lip_sync' | 'script';
    scriptText: string;
    params: Record<string, unknown>;
    generating?: boolean;
    canGenerate?: boolean;
    generateLabel?: string;
    modelOptions?: ModelSelectOption[];
    resolutionOptions?: Array<{ label: string; value: string; pixelLabel?: string }>;
    currentModelConfig?: Record<string, unknown> | null;
    showNegativePrompt?: boolean;
    portraitPreview?: string;
    portraitLabel?: string;
    audioLabel?: string;
  }>(),
  {
    generating: false,
    canGenerate: false,
    generateLabel: '',
    mode: 'lip_sync',
    scriptText: '',
    modelOptions: () => [],
    resolutionOptions: () => [],
    currentModelConfig: null,
    showNegativePrompt: true,
    portraitPreview: '',
    portraitLabel: '',
    audioLabel: '',
  },
);

const emit = defineEmits<{
  (e: 'update:modelValue', v: string): void;
  (e: 'update:title', v: string): void;
  (e: 'update:model', v: string): void;
  (e: 'update:resolution', v: string): void;
  (e: 'update:mode', v: 'lip_sync' | 'script'): void;
  (e: 'update:scriptText', v: string): void;
  (e: 'update:params', v: Record<string, unknown>): void;
  (e: 'generate'): void;
  (e: 'pick-portrait'): void;
  (e: 'remove-portrait'): void;
  (e: 'pick-audio'): void;
  (e: 'remove-audio'): void;
  (e: 'model-change'): void;
  (e: 'reset-defaults'): void;
}>();

const advancedOpen = ref(false);

const localPrompt = computed({
  get: () => props.modelValue,
  set: (v: string) => emit('update:modelValue', v),
});
const localTitle = computed({
  get: () => props.title,
  set: (v: string) => emit('update:title', v),
});
const localModel = computed({
  get: () => props.model,
  set: (v: string) => emit('update:model', v),
});

useComposerDefaultModel(
  () => props.modelOptions || [],
  () => props.model,
  (value) => {
    emit('update:model', value);
    emit('model-change');
  },
);
const localResolution = computed({
  get: () => props.resolution,
  set: (v: string) => emit('update:resolution', v),
});
const localMode = computed({
  get: () => props.mode,
  set: (v: 'lip_sync' | 'script') => emit('update:mode', v),
});
const localScriptText = computed({
  get: () => props.scriptText,
  set: (v: string) => emit('update:scriptText', v),
});

const isScriptMode = computed(() => props.mode === 'script');

const modeOptions = computed(() => [
  { label: $tt('avatar.modeLipSync'), value: 'lip_sync' },
]);

const paramSchema = computed(() => ((props.currentModelConfig?.parameters as Record<string, unknown>) || {}));
const showSeedField = computed(() => paramSchema.value.seed_support !== false);

const secondsLabel = computed(() => {
  const frames = Number(props.params.num_frames) || 0;
  const fps = Number(props.params.fps) || 0;
  if (!frames || !fps) return '';
  return `${Math.round(((frames - 1) / fps) * 10) / 10}s`;
});

const hasCustomParams = computed(() => {
  const schema = paramSchema.value;
  if (schema.steps && Number(props.params.steps) !== Number(schema.steps.default)) return true;
  if (schema.num_frames && Number(props.params.num_frames) !== Number(schema.num_frames.default)) return true;
  if (schema.fps && Number(props.params.fps) !== Number(schema.fps.default)) return true;
  if (props.params.seed != null && props.params.seed !== '') return true;
  if (String(props.params.negative_prompt ?? '').trim()) return true;
  return false;
});

function patchParams(patch: Record<string, unknown>) {
  emit('update:params', { ...props.params, ...patch });
}

function randomizeSeed() {
  patchParams({ seed: Math.floor(Math.random() * 1_000_000) });
}

function onPromptKeydown(e: KeyboardEvent) {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
    e.preventDefault();
    if (!props.generating && props.canGenerate) {
      emit('generate');
    }
  }
}
</script>

<style scoped>
.avatar-composer {
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

.avatar-composer__title {
  margin: 0;
}

.avatar-composer__title :deep(.dq-input) {
  border-radius: var(--dq-radius-input);
}

/* Prompt */
.avatar-composer__prompt-block {
  display: flex;
  flex-direction: column;
  margin-bottom: 8px;
  flex-shrink: 0;
}

.avatar-composer__prompt-wrap {
  position: relative;
  margin: 0;
}

.avatar-composer__prompt :deep(.dq-input--textarea) {
  border-radius: var(--dq-radius-group);
  font-size: var(--dq-font-size-body);
  line-height: 1.5;
  padding: 10px 12px 40px;
  min-height: 7.5rem;
  max-height: 14rem;
  resize: none;
  overflow-y: auto;
  transition: border-color 0.15s ease, background 0.15s ease;
}

/* Reference inputs inside the textarea */
.avatar-composer__ref-area {
  position: absolute;
  bottom: 6px;
  left: 8px;
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}

.avatar-composer__ref-pill {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 2px 6px 2px 2px;
  background: var(--dq-fill-secondary);
  border-radius: 6px;
  border: 1px solid var(--dq-border-subtle);
}

.avatar-composer__ref-thumb {
  display: block;
  width: 20px;
  height: 20px;
  object-fit: cover;
  border-radius: 3px;
}

.avatar-composer__ref-label {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
}

.avatar-composer__ref-add {
  opacity: 0.6;
  transition: opacity 0.2s;
}

.avatar-composer__ref-add:hover {
  opacity: 1;
}

/* Script (念台词) */
.avatar-composer__script-block {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.avatar-composer__script-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.avatar-composer__script-label {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  color: var(--dq-label-secondary);
}

.avatar-composer__script :deep(.dq-input--textarea) {
  border-radius: var(--dq-radius-group);
  font-size: var(--dq-font-size-body);
  line-height: 1.5;
  padding: 10px 12px;
  min-height: 5.5rem;
  max-height: 9rem;
  resize: none;
  overflow-y: auto;
  transition: border-color 0.15s ease, background 0.15s ease;
}

.avatar-composer__script-hint {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
  color: var(--dq-label-tertiary);
}

/* Advanced panel */
.avatar-composer__advanced-inner {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.avatar-composer__advanced-row {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}

.avatar-composer__field {
  display: flex;
  align-items: center;
  gap: 10px;
  flex: 1;
  min-width: 260px;
  overflow: hidden;
}

.avatar-composer__field--full {
  flex: 1 1 100%;
  flex-direction: column;
  align-items: flex-start;
  gap: 6px;
}

.avatar-composer__field label {
  font-size: var(--dq-font-size-caption);
  font-weight: 500;
  color: var(--dq-label-secondary);
  white-space: nowrap;
  width: 90px;
  flex-shrink: 0;
  overflow: hidden;
  text-overflow: ellipsis;
}

.avatar-composer__field :deep(.dq-slider) {
  flex: 1;
}

.avatar-composer__field :deep(.dq-input-number),
.avatar-composer__field :deep(.dq-input),
.avatar-composer__field :deep(.dq-select) {
  flex: 1;
  min-width: 0;
}

.avatar-composer__field-val {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
  min-width: 32px;
  text-align: right;
  font-variant-numeric: tabular-nums;
}

.avatar-composer__frames-formula {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
  white-space: nowrap;
}

.avatar-composer__seed-wrap {
  display: flex;
  align-items: center;
  gap: 4px;
  flex: 1;
  min-width: 0;
}

.avatar-composer__seed-wrap :deep(.dq-input) {
  flex: 1;
  min-width: 0;
}
</style>
