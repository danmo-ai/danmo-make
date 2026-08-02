<template>
  <div class="avatar-composer studio-composer-shell dq-glass--panel">
    <DqInput
      v-model="localTitle"
      size="sm"
      :placeholder="$tt('studio.workTitlePlaceholder')"
      class="avatar-composer__title"
    />

    <div class="avatar-composer__prompt-wrap">
      <DqInput
        v-model="localPrompt"
        type="textarea"
        :rows="4"
        :placeholder="$tt('avatar.promptPlaceholder')"
        resize="none"
        class="avatar-composer__prompt"
        @keydown="onPromptKeydown"
      />
    </div>

    <div class="avatar-composer__refs">
      <div class="avatar-composer__mode-switch studio-composer-mode-switch">
        <DqSegmented
          v-model="localMode"
          size="sm"
          :options="modeOptions"
        />
      </div>

      <div class="avatar-composer__ref-block">
        <div class="avatar-composer__ref-label">{{ $tt('avatar.portrait') }}</div>
        <div v-if="portraitPreview" class="avatar-composer__ref-pill">
          <img :src="portraitPreview" alt="" class="avatar-composer__ref-thumb" />
          <span class="avatar-composer__ref-name">{{ portraitLabel }}</span>
          <DqIconButton type="text" size="sm" :aria-label="$tt('common.delete')" @click="$emit('remove-portrait')">
            <DqIcon :size="10"><Close /></DqIcon>
          </DqIconButton>
        </div>
        <DqButton v-else size="sm" class="avatar-composer__ref-btn" @click="$emit('pick-portrait')">
          <DqIcon :size="14"><Picture /></DqIcon>
          {{ $tt('avatar.pickPortrait') }}
        </DqButton>
      </div>

      <div v-if="localMode === 'lip_sync'" class="avatar-composer__ref-block">
        <div class="avatar-composer__ref-label">{{ $tt('avatar.voiceTrack') }}</div>
        <div v-if="audioLabel" class="avatar-composer__ref-pill">
          <DqIcon :size="14"><Microphone /></DqIcon>
          <span class="avatar-composer__ref-name">{{ audioLabel }}</span>
          <DqIconButton type="text" size="sm" :aria-label="$tt('common.delete')" @click="$emit('remove-audio')">
            <DqIcon :size="10"><Close /></DqIcon>
          </DqIconButton>
        </div>
        <DqButton v-else size="sm" class="avatar-composer__ref-btn" @click="$emit('pick-audio')">
          <DqIcon :size="14"><Microphone /></DqIcon>
          {{ $tt('avatar.pickAudio') }}
        </DqButton>
      </div>

      <div v-else class="avatar-composer__script-block">
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
    </div>

    <div class="avatar-composer__row">
      <label class="avatar-composer__field-label">{{ $tt('studio.model') }}</label>
      <DqSelect
        v-model="localModel"
        size="sm"
        :options="modelOptions"
        filterable
        class="avatar-composer__model"
        @change="$emit('model-change')"
      />
    </div>

    <div v-if="resolutionOptions.length" class="avatar-composer__row">
      <label class="avatar-composer__field-label">{{ $tt('create.resolution') }}</label>
      <DqSelect
        v-model="localResolution"
        size="sm"
        :options="resolutionOptions"
        class="avatar-composer__resolution"
      />
    </div>

    <div v-if="paramSchema.steps || paramSchema.num_frames || paramSchema.fps" class="avatar-composer__advanced">
      <div v-if="paramSchema.steps" class="avatar-composer__row avatar-composer__row--inline">
        <label class="avatar-composer__field-label">{{ $tt('create.steps') }}</label>
        <DqInputNumber
          :model-value="params.steps as number"
          size="sm"
          :min="paramSchema.steps.min ?? 1"
          :max="paramSchema.steps.max ?? 50"
          @update:model-value="patchParams({ steps: $event })"
        />
      </div>
      <div v-if="paramSchema.num_frames" class="avatar-composer__row avatar-composer__row--inline">
        <label class="avatar-composer__field-label">{{ $tt('create.frames') }}</label>
        <DqInputNumber
          :model-value="params.num_frames as number"
          size="sm"
          :min="paramSchema.num_frames.min ?? 25"
          :max="paramSchema.num_frames.max ?? 121"
          :step="paramSchema.num_frames.step ?? 1"
          @update:model-value="patchParams({ num_frames: $event })"
        />
      </div>
      <div v-if="paramSchema.fps" class="avatar-composer__row avatar-composer__row--inline">
        <label class="avatar-composer__field-label">{{ $tt('create.fps') }}</label>
        <DqInputNumber
          :model-value="params.fps as number"
          size="sm"
          :min="paramSchema.fps.min ?? 15"
          :max="paramSchema.fps.max ?? 30"
          @update:model-value="patchParams({ fps: $event })"
        />
      </div>
      <div v-if="showSeedField" class="avatar-composer__row avatar-composer__row--inline">
        <label class="avatar-composer__field-label">{{ $tt('create.seed') }}</label>
        <DqInput
          :model-value="String(params.seed ?? '')"
          size="sm"
          :placeholder="$tt('studio.seedPlaceholder')"
          @update:model-value="patchParams({ seed: $event ? Number($event) : null })"
        />
      </div>
      <div v-if="showNegativePrompt" class="avatar-composer__row">
        <label class="avatar-composer__field-label">{{ $tt('create.negativePrompt') }}</label>
        <DqInput
          :model-value="String(params.negative_prompt ?? '')"
          type="textarea"
          :rows="2"
          @update:model-value="patchParams({ negative_prompt: $event })"
        />
      </div>
    </div>

    <div class="avatar-composer__footer">
      <DqButton
        type="primary"
        size="lg"
        block
        :loading="generating"
        :disabled="!canGenerate"
        @click="$emit('generate')"
      >
        {{ generateLabel }}
      </DqButton>
    </div>
  </div>
</template>

<script setup lang="ts">
// @ts-nocheck — registry param schema is dynamic
import { computed } from 'vue';
import { Close, Microphone, Picture } from '@danqing/dq-shell';
import { DqSegmented } from '@danqing/dq-ui';
import { $tt } from '@/utils/i18n';

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
    modelOptions?: { label: string; value: string; disabled?: boolean }[];
    resolutionOptions?: { label: string; value: string }[];
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
}>();

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

const modeOptions = computed(() => [
  { label: $tt('avatar.modeLipSync'), value: 'lip_sync' },
  { label: $tt('avatar.modeScript'), value: 'script' },
]);

const paramSchema = computed(() => ((props.currentModelConfig?.parameters as Record<string, unknown>) || {}));
const showSeedField = computed(() => paramSchema.value.seed_support !== false);

function patchParams(patch: Record<string, unknown>) {
  emit('update:params', { ...props.params, ...patch });
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

.avatar-composer__prompt-wrap {
  margin: 0;
}

.avatar-composer__prompt :deep(.dq-input--textarea) {
  border-radius: var(--dq-radius-group);
  line-height: 1.5;
  padding: 10px 12px 32px;
  min-height: 7rem;
  max-height: 14rem;
  resize: none;
  overflow-y: auto;
  transition: border-color 0.15s ease, background 0.15s ease;
}

.avatar-composer__refs {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-bottom: 8px;
}

.avatar-composer__ref-block {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.avatar-composer__ref-label {
  font-size: var(--dq-font-size-caption);
  font-weight: 500;
  color: var(--dq-label-secondary);
  line-height: 1.3;
}

.avatar-composer__ref-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  width: 100%;
  font-size: var(--dq-font-size-caption);
  padding: 6px 10px;
}

.avatar-composer__ref-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 3px 8px 3px 3px;
  background: var(--dq-fill-secondary);
  border-radius: 6px;
  border: 1px solid var(--dq-border-subtle);
  max-width: 100%;
}

.avatar-composer__ref-thumb {
  width: 24px;
  height: 24px;
  object-fit: cover;
  border-radius: 4px;
}

.avatar-composer__ref-name {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
}

.avatar-composer__mode-switch {
  margin: 0;
}

.avatar-composer__script-block {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.avatar-composer__script :deep(.dq-input--textarea) {
  border-radius: var(--dq-radius-group);
  font-size: var(--dq-font-size-body);
  line-height: 1.5;
  padding: 10px 12px;
  min-height: 6rem;
  max-height: 12rem;
  resize: none;
  overflow-y: auto;
}

.avatar-composer__script-hint {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  line-height: 1.4;
  color: var(--dq-label-tertiary);
}

.avatar-composer__row {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-bottom: 8px;
}

.avatar-composer__field-label {
  font-size: var(--dq-font-size-caption);
  font-weight: 500;
  color: var(--dq-label-secondary);
  line-height: 1.3;
}

.avatar-composer__row :deep(.dq-input) {
  font-size: var(--dq-font-size-caption);
}

.avatar-composer__row--inline {
  display: grid;
  grid-template-columns: 80px 1fr;
  align-items: center;
  gap: 10px;
}

.avatar-composer__row--inline .avatar-composer__field-label {
  margin: 0;
}

.avatar-composer__advanced {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 8px;
}

.avatar-composer__advanced .avatar-composer__row {
  margin-bottom: 0;
}

.avatar-composer__footer {
  margin-top: 8px;
}

.avatar-composer__footer :deep(.dq-button) {
  min-height: 36px;
}
</style>
