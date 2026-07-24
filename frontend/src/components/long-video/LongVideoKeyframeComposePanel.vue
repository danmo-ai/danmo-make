<template>
  <section class="lv-kf-compose">
    <div class="lv-kf-compose__toolbar">
      <DqSegmented
        :model-value="mode"
        size="sm"
        :options="modeOptions"
        @update:model-value="$emit('update:mode', $event)"
      />
    </div>

    <div class="lv-kf-compose__prompt-wrap">
      <DqInput
        :model-value="prompt"
        type="textarea"
        :rows="7"
        :placeholder="$tt('create.promptPlaceholder')"
        resize="vertical"
        class="lv-kf-compose__prompt"
        @update:model-value="$emit('update:prompt', $event)"
      />
      <div class="lv-kf-compose__prompt-actions">
        <div class="lv-kf-compose__ref">
          <div v-if="referenceImage" class="lv-kf-compose__ref-pill">
            <img :src="referenceImage.previewUrl" alt="" />
            <DqIconButton type="text" size="sm" :label="$tt('common.delete')" @click="$emit('remove-reference')">
              <DqIcon :size="10"><Close /></DqIcon>
            </DqIconButton>
          </div>
          <DqIconButton
            v-else
            type="text"
            size="sm"
            :label="$tt('create.refImage')"
            @click="$emit('pick-reference')"
          >
            <DqIcon :size="14"><Picture /></DqIcon>
          </DqIconButton>
        </div>
        <div class="lv-kf-compose__prompt-tools">
          <DqIconButton
            type="text"
            size="sm"
            :disabled="enhancing || !prompt.trim()"
            :label="$tt('create.enhance')"
            @click="$emit('enhance')"
          >
            <DqIcon :size="12"><MagicStick /></DqIcon>
          </DqIconButton>
          <DqDropdown
            v-if="Object.keys(styles).length > 0"
            trigger="click"
            size="sm"
            @command="onStyleCommand"
          >
            <DqIconButton type="text" size="sm" :label="$tt('create.composerTip.preset')">
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
    </div>

    <slot name="after-prompt" />

    <div class="lv-kf-compose__advanced-head">
      <button type="button" class="lv-kf-compose__advanced-toggle" @click="advancedOpen = !advancedOpen">
        <DqIcon :size="14"><Tools /></DqIcon>
        <span>{{ $tt('studio.advancedParams') }}</span>
        <span class="lv-kf-compose__chevron" :class="{ 'is-open': advancedOpen }" aria-hidden="true">▾</span>
      </button>
      <DqButton v-if="advancedOpen" type="text" size="sm" @click="$emit('reset-defaults')">
        {{ $tt('create.restoreDefaults') }}
      </DqButton>
    </div>

    <ImageComposerAdvancedFields
      v-if="advancedOpen"
      stacked
      :params="params"
      :model="model"
      :mode="mode"
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

    <DqButton
      type="primary"
      block
      class="lv-kf-compose__generate"
      :loading="generating"
      :disabled="!canGenerate || generating"
      @click="$emit('generate')"
    >
      <DqIcon size="16"><MagicStick /></DqIcon>
      <span>{{ $tt('video.longVideoGenCurrentKeyframe') }}</span>
    </DqButton>
  </section>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { useI18n } from 'vue-i18n';
import { Close, DocumentCopy, MagicStick, Picture, Tools } from '@danqing/dq-shell';
import ImageComposerAdvancedFields from '@/components/studio/ImageComposerAdvancedFields.vue';
import type { KeyframeComposeParams } from '@/composables/useLongVideoKeyframeCompose';
import { appendStyleBoost } from '@/utils/styleBoost';

const props = defineProps<{
  prompt: string;
  model: string;
  mode: string;
  params: KeyframeComposeParams;
  generating?: boolean;
  canGenerate?: boolean;
  styles: Record<string, { applies_to?: string[]; positive?: string; negative?: string; trigger_words?: string; media_scope?: string }>;
  showNegativePrompt?: boolean;
  referenceImage: { previewUrl: string; path: string } | null;
  controlImage?: { previewUrl: string; path: string } | null;
  inpaintSourceImage?: { previewUrl: string; path: string } | null;
  inpaintMaskImage?: { previewUrl: string; path: string } | null;
  currentModelConfig?: Record<string, unknown> | null;
  compatibleLoras?: Record<string, unknown>[];
  compatibleControlNets?: Record<string, unknown>[];
  controlNetRuntimeAvailable?: boolean;
  enhancing?: boolean;
}>();

const emit = defineEmits<{
  (e: 'update:prompt', value: string): void;
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
  (e: 'reset-defaults'): void;
  (e: 'enhance'): void;
}>();

const { t: $tt } = useI18n();

const advancedOpen = ref(false);

const modeOptions = [
  { label: $tt('action.image.text2img'), value: 'text2img' },
  { label: $tt('action.image.img2img'), value: 'img2img' },
];

function presetLabel(name: string, preset: Record<string, unknown>): string {
  const a = (preset.applies_to as string[]) || [];
  const hasC = a.includes('create');
  const hasEdit = a.some((x: string) => ['rewrite', 'retouch', 'extend'].includes(x));
  let tag = '';
  if (hasC && !hasEdit) tag = $tt('create.presetTagT2I');
  else if (hasEdit && !hasC) tag = $tt('create.presetTagI2I');
  return tag ? `${tag} ${name}` : name;
}

function onStyleCommand(name: string) {
  const preset = props.styles[name];
  if (!preset) return;
  let nextPrompt = props.prompt;
  if (preset.positive) {
    const positive = String(preset.positive);
    nextPrompt = appendStyleBoost(nextPrompt || '', positive);
  }
  if (preset.trigger_words) {
    const tw = String(preset.trigger_words).trim();
    if (tw) nextPrompt = nextPrompt ? `${nextPrompt}\n${tw}` : tw;
  }
  if (nextPrompt !== props.prompt) emit('update:prompt', nextPrompt);
  if (preset.negative && props.showNegativePrompt) {
    const current = props.params.negative_prompt || '';
    props.params.negative_prompt = current
      ? `${current}\n${String(preset.negative)}`
      : String(preset.negative);
  }
}
</script>

<style scoped>
.lv-kf-compose {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-width: 0;
}

.lv-kf-compose__toolbar {
  display: flex;
  align-items: center;
}

.lv-kf-compose__prompt-wrap {
  position: relative;
}

.lv-kf-compose__prompt :deep(textarea) {
  font-size: var(--dq-font-size-body);
  line-height: 1.55;
  min-height: 168px;
  max-height: 320px;
  padding-bottom: 34px;
}

.lv-kf-compose__prompt-actions {
  position: absolute;
  left: 8px;
  right: 8px;
  bottom: 8px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  pointer-events: none;
}

.lv-kf-compose__ref,
.lv-kf-compose__prompt-tools {
  display: flex;
  align-items: center;
  gap: 4px;
  pointer-events: auto;
}

.lv-kf-compose__ref-pill {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 6px 2px 2px;
  border-radius: 8px;
  border: 0.5px solid var(--dq-glass-border, var(--dq-border-subtle));
  background: color-mix(in srgb, var(--dq-surface-elevated) 85%, transparent);
}

.lv-kf-compose__ref-pill img {
  width: 28px;
  height: 28px;
  object-fit: cover;
  border-radius: 6px;
}

.lv-kf-compose__advanced-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.lv-kf-compose__advanced-toggle {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border: none;
  background: none;
  padding: 4px 0;
  font-size: var(--dq-font-size-caption);
  font-weight: 500;
  color: var(--dq-label-secondary);
  cursor: pointer;
}

.lv-kf-compose__advanced-toggle:hover {
  color: var(--dq-accent);
}

.lv-kf-compose__chevron {
  font-size: var(--dq-font-size-caption);
  line-height: 1;
  transition: transform 0.2s ease;
  color: var(--dq-label-tertiary);
}

.lv-kf-compose__chevron.is-open {
  transform: rotate(180deg);
}

.lv-kf-compose__generate {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  font-weight: 600;
}
</style>
