<template>
  <section class="lv-seg-compose">
    <div class="lv-seg-compose__meta-row">
      <span class="lv-seg-compose__meta-label">{{ $tt('video.longVideoShotDuration') }}</span>
      <DqSelect
        :model-value="durationSec"
        size="sm"
        variant="ghost"
        class="lv-seg-compose__duration"
        @update:model-value="$emit('update:duration', Number($event))"
      >
        <DqOption
          v-for="sec in durationOptions"
          :key="sec"
          :label="$tt('video.longVideoSegmentDurationSec', { sec })"
          :value="sec"
        />
      </DqSelect>
    </div>

    <div class="lv-seg-compose__meta-row">
      <span class="lv-seg-compose__meta-label">{{ $tt('video.longVideoSegmentChainMode') }}</span>
      <DqSelect
        :model-value="chainMode"
        size="sm"
        variant="ghost"
        class="lv-seg-compose__chain"
        :title="chainModeHint"
        @update:model-value="$emit('update:chainMode', $event)"
      >
        <DqOption :label="$tt('video.longVideoChainKeyframe')" value="keyframe_only" />
        <DqOption
          :label="$tt('video.longVideoChainLastFrame')"
          value="last_frame"
          :disabled="!canUseLastFrame"
        />
        <DqOption
          v-if="canUseReferenceR2v"
          :label="$tt('video.longVideoChainReferenceR2v')"
          value="reference_r2v"
        />
      </DqSelect>
    </div>

    <p v-if="chainMode === 'last_frame' && !canUseLastFrame" class="lv-seg-compose__hint">
      {{ $tt('video.longVideoSegmentChainFirstEdgeHint') }}
    </p>
    <p v-else-if="chainModeHint" class="lv-seg-compose__hint">{{ chainModeHint }}</p>

    <div class="lv-seg-compose__prompt-wrap">
      <DqInput
        :model-value="motionPrompt"
        type="textarea"
        :rows="7"
        resize="vertical"
        class="lv-seg-compose__prompt"
        :placeholder="$tt('video.longVideoTransitionPh')"
        @update:model-value="$emit('update:motion', $event)"
      />
      <div class="lv-seg-compose__prompt-actions">
        <DqIconButton
          type="text"
          size="sm"
          :disabled="polishing || !canPolish"
          :label="$tt('video.longVideoPolishSegment')"
          @click="$emit('polish')"
        >
          <DqIcon :size="12"><MagicStick /></DqIcon>
        </DqIconButton>
      </div>
    </div>

    <div class="lv-seg-compose__advanced-head">
      <button type="button" class="lv-seg-compose__advanced-toggle" @click="advancedOpen = !advancedOpen">
        <DqIcon :size="14"><Tools /></DqIcon>
        <span>{{ $tt('studio.advancedParams') }}</span>
        <span class="lv-seg-compose__chevron" :class="{ 'is-open': advancedOpen }" aria-hidden="true">▾</span>
      </button>
      <DqButton v-if="advancedOpen" type="text" size="sm" @click="$emit('reset-defaults')">
        {{ $tt('create.restoreDefaults') }}
      </DqButton>
    </div>

    <VideoComposerAdvancedFields
      v-if="advancedOpen"
      :params="params"
      :param-schema="paramSchema"
      :duration-sec="durationSec"
      :num-frames="numFrames"
      :show-negative-prompt="showNegativePrompt"
      :show-seed-field="showSeedField"
      :show-lora="showLora"
      :compatible-loras="compatibleLoras"
    />

    <p v-if="!canGenerate && missingKeyframe" class="lv-seg-compose__warn">
      {{ missingAnchor ? $tt('video.longVideoNeedAnchorForSegment') : $tt('video.longVideoNeedKeyframeForSegment') }}
    </p>

    <DqButton
      type="primary"
      block
      class="lv-seg-compose__generate"
      :loading="generating"
      :disabled="!canGenerate || generating"
      @click="$emit('generate')"
    >
      <DqIcon size="16"><VideoPlay /></DqIcon>
      <span>{{ $tt('video.longVideoGenSegment') }}</span>
    </DqButton>
  </section>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import { MagicStick, Tools, VideoPlay } from '@danqing/dq-shell';
import VideoComposerAdvancedFields from '@/components/studio/VideoComposerAdvancedFields.vue';
import type { SegmentComposeParams } from '@/composables/useLongVideoSegmentCompose';
import type { LongVideoChainMode } from '@/types';
import type { NormalizedParamSpec } from '@/utils/registryParamSchema';

const props = defineProps<{
  motionPrompt: string;
  durationSec: number;
  durationOptions: number[];
  chainMode: LongVideoChainMode;
  canUseLastFrame?: boolean;
  canUseReferenceR2v?: boolean;
  params: SegmentComposeParams;
  paramSchema: Record<string, NormalizedParamSpec>;
  numFrames: number;
  generating?: boolean;
  canGenerate?: boolean;
  canPolish?: boolean;
  polishing?: boolean;
  missingKeyframe?: boolean;
  missingAnchor?: boolean;
  showNegativePrompt?: boolean;
  showSeedField?: boolean;
  showLora?: boolean;
  compatibleLoras?: Record<string, unknown>[];
}>();

defineEmits<{
  (e: 'update:motion', value: string): void;
  (e: 'update:duration', value: number): void;
  (e: 'update:chainMode', value: LongVideoChainMode): void;
  (e: 'generate'): void;
  (e: 'polish'): void;
  (e: 'reset-defaults'): void;
}>();

const { t: $tt } = useI18n();
const advancedOpen = ref(false);

const chainModeHint = computed(() => {
  if (props.chainMode === 'last_frame') return $tt('video.longVideoChainModeHintLastFrame');
  if (props.chainMode === 'reference_r2v') return $tt('video.longVideoChainModeHintReferenceR2v');
  return $tt('video.longVideoChainModeHintKeyframeOnly');
});
</script>

<style scoped>
.lv-seg-compose {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-width: 0;
}

.lv-seg-compose__meta-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
}

.lv-seg-compose__meta-label {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  letter-spacing: var(--dq-tracking-wide);
  color: var(--dq-label-secondary);
  white-space: nowrap;
}

.lv-seg-compose__duration,
.lv-seg-compose__chain {
  width: min(100%, 168px);
  flex-shrink: 0;
}

.lv-seg-compose__hint {
  margin: -4px 0 0;
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
  color: var(--dq-label-tertiary);
}

.lv-seg-compose__prompt-wrap {
  position: relative;
}

.lv-seg-compose__prompt :deep(textarea) {
  font-size: var(--dq-font-size-body);
  line-height: 1.55;
  min-height: 168px;
  max-height: 320px;
  padding-bottom: 34px;
}

.lv-seg-compose__prompt-actions {
  position: absolute;
  right: 8px;
  bottom: 8px;
  display: flex;
  align-items: center;
  pointer-events: auto;
}

.lv-seg-compose__advanced-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.lv-seg-compose__advanced-toggle {
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

.lv-seg-compose__advanced-toggle:hover {
  color: var(--dq-accent);
}

.lv-seg-compose__chevron {
  font-size: var(--dq-font-size-caption);
  line-height: 1;
  transition: transform 0.2s ease;
  color: var(--dq-label-tertiary);
}

.lv-seg-compose__chevron.is-open {
  transform: rotate(180deg);
}

.lv-seg-compose__warn {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
  color: color-mix(in srgb, var(--dq-warning) 85%, var(--dq-label-secondary));
}

.lv-seg-compose__generate {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  font-weight: 600;
}
</style>
