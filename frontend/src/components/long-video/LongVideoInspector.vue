<template>
  <div v-if="selection" class="lv-inspector">
    <div class="lv-inspector__tabs">
      <DqSegmented
        v-model="activeTab"
        block
        size="sm"
        class="lv-inspector-segmented"
        :options="inspectorTabOptions"
      />
    </div>

    <!-- 分镜图 -->
    <div v-show="activeTab === 'frame' && selectedShot" class="lv-inspector__body">
      <DqAlert
        v-if="isTailChainSegment"
        type="info"
        :closable="false"
        class="lv-inspector__tail-hint"
        :title="tailChainHint"
      />

      <DqAlert
        v-else-if="isPostAnchorLinkedStart"
        type="info"
        :closable="false"
        class="lv-inspector__tail-hint"
        :title="$tt('video.longVideoPostAnchorStartHint')"
      />

      <section v-if="showFramePreview" class="lv-inspector__canvas lv-inspector__canvas--sticky">
        <div class="lv-inspector__media-head">
          <div v-if="outputSizeLabel" class="lv-inspector__meta">
            <span class="lv-inspector__meta-size">{{ outputSizeLabel }}</span>
            <span v-if="roleLabel" class="lv-inspector__meta-dot" aria-hidden="true">·</span>
            <span v-if="roleLabel" class="lv-inspector__meta-size">{{ roleLabel }}</span>
          </div>
          <div v-if="showFrameCompose" class="lv-inspector__tools">
            <button
              type="button"
              class="lv-inspector__tool"
              :disabled="keyframeGenerating"
              @click="$emit('pick-keyframe-gallery', segmentIndex)"
            >
              <DqIcon :size="14"><PictureFilled /></DqIcon>
              <span>{{ $tt('video.longVideoGalleryShort') }}</span>
            </button>
            <button
              v-if="selectedShot?.keyframe_asset_id"
              type="button"
              class="lv-inspector__tool lv-inspector__tool--danger"
              :disabled="keyframeGenerating"
              @click="$emit('clear-keyframe', segmentIndex)"
            >
              <DqIcon :size="14"><Delete /></DqIcon>
              <span>{{ $tt('video.longVideoClearShort') }}</span>
            </button>
          </div>
          <DqButton
            v-if="isPostAnchorLinkedStart && faceAnchorIndex != null"
            type="text"
            size="xs"
            @click="$emit('select-segment', faceAnchorIndex!)"
          >
            {{ $tt('video.longVideoJumpToAnchor') }}
          </DqButton>
        </div>
        <div
          class="lv-inspector__preview"
          :class="{ 'is-generating': keyframeGenerating && showFrameCompose }"
        >
          <img v-if="framePreviewUrl" class="lv-inspector__img" :src="framePreviewUrl" alt="" />
          <div v-if="keyframeGenerating && showFrameCompose" class="lv-inspector__loading" aria-live="polite">
            <DqIcon :size="24"><Loading /></DqIcon>
          </div>
        </div>
      </section>

      <p v-if="firstFrameRequirementText" class="lv-inspector__requirement">
        <span class="lv-inspector__requirement-label">{{ $tt('video.longVideoFirstFrameRequirement') }}</span>
        {{ firstFrameRequirementText }}
      </p>
      <p v-if="visibilitySummary" class="lv-inspector__vis-meta">{{ visibilitySummary }}</p>

      <LongVideoAnchorComposePanel
        v-if="showFrameCompose && isFaceAnchor"
        :prompt="keyframePromptText"
        :model="composeModel"
        :mode="composeMode ?? 'text2img'"
        :params="composeParams"
        :generating="keyframeGenerating"
        :can-generate="canGenerateKeyframe"
        :styles="composeStyles"
        :show-negative-prompt="composeShowNegativePrompt"
        :reference-image="referenceImage"
        :control-image="controlImage"
        :inpaint-source-image="inpaintSourceImage"
        :inpaint-mask-image="inpaintMaskImage"
        :current-model-config="composeModelConfig"
        :compatible-loras="compatibleLoras"
        :compatible-control-nets="compatibleControlNets"
        :control-net-runtime-available="controlNetRuntimeAvailable"
        :enhancing="visualPolishing"
        @update:prompt="$emit('update-visual', segmentIndex, $event)"
        @update:mode="$emit('update-compose-mode', $event)"
        @generate="$emit('generate-keyframe', segmentIndex)"
        @pick-reference="$emit('pick-reference')"
        @remove-reference="$emit('remove-reference')"
        @pick-control="$emit('pick-control')"
        @remove-control="$emit('remove-control')"
        @pick-inpaint-source="$emit('pick-inpaint-source')"
        @remove-inpaint-source="$emit('remove-inpaint-source')"
        @pick-inpaint-mask="$emit('pick-inpaint-mask')"
        @remove-inpaint-mask="$emit('remove-inpaint-mask')"
        @reset-defaults="$emit('reset-compose-defaults')"
        @enhance="$emit('polish-visual', segmentIndex)"
      >
        <template #after-prompt>
          <LongVideoCastLookPanel
            v-if="characters.length"
            :characters="characters"
            :cast-looks="selectedShot?.cast_looks ?? []"
            :scene-text="shotCastMatchTextValue"
            :select-id-prefix="`lv-cast-${segmentIndex}`"
            @update:cast-looks="$emit('update-cast-looks', segmentIndex, $event)"
          />
          <LongVideoSceneLookPanel
            v-if="(scenes ?? []).length"
            :scenes="scenes ?? []"
            :scene-look="selectedShot?.scene_look"
            :beat-text="shotSceneBeatText"
            :select-id-prefix="`lv-scene-anchor-${segmentIndex}`"
            @update:scene-look="$emit('update-scene-look', segmentIndex, $event)"
          />
          <LongVideoGenerationPromptPreview
            :preview="keyframeT2iPromptPreview"
            :mode-hint="keyframePromptModeHint"
          />
          <LongVideoT2iProvenancePanel
            :provenance="keyframeT2iProvenance"
            :parse-run-id="parseRunId"
          />
        </template>
      </LongVideoAnchorComposePanel>

      <LongVideoKeyframeComposePanel
        v-else-if="showFrameCompose"
        :prompt="keyframePromptText"
        :model="composeModel"
        :mode="composeMode ?? 'text2img'"
        :params="composeParams"
        :generating="keyframeGenerating"
        :can-generate="canGenerateKeyframe"
        :styles="composeStyles"
        :show-negative-prompt="composeShowNegativePrompt"
        :reference-image="referenceImage"
        :control-image="controlImage"
        :inpaint-source-image="inpaintSourceImage"
        :inpaint-mask-image="inpaintMaskImage"
        :current-model-config="composeModelConfig"
        :compatible-loras="compatibleLoras"
        :compatible-control-nets="compatibleControlNets"
        :control-net-runtime-available="controlNetRuntimeAvailable"
        :enhancing="visualPolishing"
        @update:prompt="$emit('update-visual', segmentIndex, $event)"
        @update:mode="$emit('update-compose-mode', $event)"
        @generate="$emit('generate-keyframe', segmentIndex)"
        @pick-reference="$emit('pick-reference')"
        @remove-reference="$emit('remove-reference')"
        @pick-control="$emit('pick-control')"
        @remove-control="$emit('remove-control')"
        @pick-inpaint-source="$emit('pick-inpaint-source')"
        @remove-inpaint-source="$emit('remove-inpaint-source')"
        @pick-inpaint-mask="$emit('pick-inpaint-mask')"
        @remove-inpaint-mask="$emit('remove-inpaint-mask')"
        @reset-defaults="$emit('reset-compose-defaults')"
        @enhance="$emit('polish-visual', segmentIndex)"
      >
        <template #after-prompt>
          <LongVideoSceneLookPanel
            v-if="(scenes ?? []).length"
            :scenes="scenes ?? []"
            :scene-look="selectedShot?.scene_look"
            :beat-text="shotSceneBeatText"
            :select-id-prefix="`lv-scene-${segmentIndex}`"
            @update:scene-look="$emit('update-scene-look', segmentIndex, $event)"
          />
          <LongVideoCastLookPanel
            v-if="characters.length"
            :characters="characters"
            :cast-looks="selectedShot?.cast_looks ?? []"
            :scene-text="shotCastMatchTextValue"
            :select-id-prefix="`lv-cast-${segmentIndex}`"
            @update:cast-looks="$emit('update-cast-looks', segmentIndex, $event)"
          />
          <LongVideoGenerationPromptPreview
            :preview="keyframeT2iPromptPreview"
            :mode-hint="keyframePromptModeHint"
          />
          <LongVideoT2iProvenancePanel
            :provenance="keyframeT2iProvenance"
            :parse-run-id="parseRunId"
          />
        </template>
      </LongVideoKeyframeComposePanel>

      <DqAlert
        v-if="isFaceAnchor && selectedShot && !selectedShot.keyframe_asset_id && showFrameCompose"
        type="warning"
        :closable="false"
        :title="$tt('video.longVideoAnchorMissingWarn')"
      />
    </div>

    <!-- 分镜视频 -->
    <div v-show="activeTab === 'clip'" class="lv-inspector__body">
      <p v-if="!selectedShot" class="lv-inspector__tab-empty">
        {{ $tt('video.longVideoInspectorClipEmpty') }}
      </p>
      <template v-else>
        <section class="lv-inspector__canvas lv-inspector__canvas--sticky">
          <div class="lv-inspector__media-head">
            <div class="lv-inspector__meta">
              <span class="lv-inspector__meta-range">#{{ segmentIndex + 1 }}</span>
              <span class="lv-inspector__meta-dot" aria-hidden="true">·</span>
              <span class="lv-inspector__meta-size">{{ segmentDurationLabel }}</span>
              <template v-if="segmentModelLabel">
                <span class="lv-inspector__meta-dot" aria-hidden="true">·</span>
                <span class="lv-inspector__meta-size">{{ segmentModelLabel }}</span>
              </template>
            </div>
            <div class="lv-inspector__tools">
              <button
                v-if="selectedShot.segment_asset_id"
                type="button"
                class="lv-inspector__tool lv-inspector__tool--danger"
                :disabled="segmentGenerating"
                @click="$emit('clear-segment', segmentIndex)"
              >
                <DqIcon :size="14"><Delete /></DqIcon>
                <span>{{ $tt('video.longVideoClearShort') }}</span>
              </button>
            </div>
          </div>
          <div
            class="lv-inspector__preview"
            :class="{ 'is-generating': segmentGenerating, 'is-empty': !segmentVideoUrl && !segmentGenerating }"
          >
            <video
              v-if="segmentVideoUrl"
              class="lv-inspector__segment-video"
              :src="segmentVideoUrl"
              controls
              playsinline
              preload="metadata"
            />
            <p v-else-if="!segmentGenerating" class="lv-inspector__clip-empty">
              {{ $tt('video.longVideoSegmentPreviewEmpty') }}
            </p>
            <div v-if="segmentGenerating" class="lv-inspector__loading" aria-live="polite">
              <DqIcon :size="24"><Loading /></DqIcon>
            </div>
          </div>
        </section>

        <LongVideoSegmentComposePanel
          :motion-prompt="segmentVideoPrompt"
          :duration-sec="segmentDurationSec"
          :duration-options="segmentDurationOptions"
          :chain-mode="clipChainMode"
          :can-use-last-frame="canUseLastFrame"
          :can-use-reference-r2v="canUseReferenceR2v"
          :params="segmentComposeParams"
          :param-schema="segmentParamSchema"
          :num-frames="segmentNumFrames"
          :generating="segmentGenerating"
          :can-generate="canGenerateSegment"
          :can-polish="canPolishMotion"
          :polishing="motionPolishing"
          :missing-keyframe="segmentMissingKeyframe"
          :missing-anchor="segmentMissingAnchor"
          :show-negative-prompt="segmentShowNegativePrompt"
          :show-seed-field="segmentShowSeedField"
          :show-lora="segmentShowLora"
          :compatible-loras="segmentCompatibleLoras"
          @update:motion="$emit('update-motion', segmentIndex, $event)"
          @update:duration="$emit('update-duration', segmentIndex, $event)"
          @update:chain-mode="$emit('update-chain-mode', segmentIndex, $event)"
          @generate="$emit('generate-segment', segmentIndex)"
          @polish="$emit('polish-motion', segmentIndex)"
          @reset-defaults="$emit('reset-segment-defaults')"
        />
        <LongVideoGenerationPromptPreview
          :preview="segmentI2vPromptPreview"
          :mode-hint="$tt('video.longVideoGenerationPromptI2vHint')"
        />
      </template>
    </div>

    <LongVideoRelatedTasks
      v-if="selectedShot?.id"
      :project-id="projectId"
      :shot-id="selectedShot.id"
      :refresh-token="relatedTasksRefreshToken"
    />
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { Delete, Loading, PictureFilled } from '@danqing/dq-shell';
import { api } from '@/utils/api';
import {
  keyframeGenerationPrompt,
  keyframePromptContextForShot,
  buildKeyframeT2iProvenance,
  keyframeThumbnailUrl,
  numFramesForDurationSec,
  shotDurationSec,
  shotKeyframeText,
  shotNeedsKeyframe,
  shotVideoPrompt,
  segmentVideoSubmitPreview,
  canGenerateSegmentShot,
  effectiveShotChainMode,
  extractKeyframeShotScene,
  findFaceAnchorShot,
  shotCastMatchText,
  segmentRoleLabelKey,
  visibilityShortLabel,
} from '@/utils/longVideoProject';
import LongVideoAnchorComposePanel from './LongVideoAnchorComposePanel.vue';
import LongVideoCastLookPanel from './LongVideoCastLookPanel.vue';
import LongVideoSceneLookPanel from './LongVideoSceneLookPanel.vue';
import LongVideoGenerationPromptPreview from './LongVideoGenerationPromptPreview.vue';
import LongVideoT2iProvenancePanel from './LongVideoT2iProvenancePanel.vue';
import LongVideoKeyframeComposePanel from './LongVideoKeyframeComposePanel.vue';
import LongVideoSegmentComposePanel from './LongVideoSegmentComposePanel.vue';
import LongVideoRelatedTasks from './LongVideoRelatedTasks.vue';
import type { KeyframeComposeParams } from '@/composables/useLongVideoKeyframeCompose';
import type { SegmentComposeParams } from '@/composables/useLongVideoSegmentCompose';
import type {
  LongVideoChainMode,
  LongVideoCharacter,
  LongVideoInspectorTab,
  LongVideoScene,
  LongVideoSelection,
  LongVideoShotCastLook,
  LongVideoShotSceneLook,
  LongVideoShotState,
} from '@/types';
import type { NormalizedParamSpec } from '@/utils/registryParamSchema';

const props = defineProps<{
  shots: LongVideoShotState[];
  selection: LongVideoSelection;
  segmentDurationOptions: number[];
  segmentModelLabel?: string;
  segmentModelSupportsR2v?: boolean;
  segmentComposeParams: SegmentComposeParams;
  segmentParamSchema: Record<string, NormalizedParamSpec>;
  segmentCompatibleLoras?: Record<string, unknown>[];
  segmentShowNegativePrompt?: boolean;
  segmentShowSeedField?: boolean;
  segmentShowLora?: boolean;
  defaultChainMode: LongVideoChainMode;
  keyframeGenerating?: boolean;
  segmentGenerating?: boolean;
  visualPolishing?: boolean;
  motionPolishing?: boolean;
  outputSizeLabel?: string;
  composeModel: string;
  composeParams: KeyframeComposeParams;
  composeStyles: Record<string, { applies_to?: string[]; positive?: string; negative?: string; trigger_words?: string; media_scope?: string }>;
  composeShowNegativePrompt?: boolean;
  composeMode?: string;
  composeModelConfig?: Record<string, unknown> | null;
  compatibleLoras?: Record<string, unknown>[];
  compatibleControlNets?: Record<string, unknown>[];
  controlNetRuntimeAvailable?: boolean;
  referenceImage: { previewUrl: string; path: string } | null;
  controlImage?: { previewUrl: string; path: string } | null;
  inpaintSourceImage?: { previewUrl: string; path: string } | null;
  inpaintMaskImage?: { previewUrl: string; path: string } | null;
  canGenerateKeyframe?: boolean;
  characters?: LongVideoCharacter[];
  scenes?: LongVideoScene[];
  characterAnchor?: string;
  styleAnchor?: string;
  projectId?: string;
  parseRunId?: string;
}>();

defineEmits<{
  (e: 'update-visual', index: number, value: string): void;
  (e: 'update-cast-looks', index: number, value: LongVideoShotCastLook[]): void;
  (e: 'update-scene-look', index: number, value: LongVideoShotSceneLook | undefined): void;
  (e: 'update-motion', index: number, value: string): void;
  (e: 'update-duration', index: number, value: number): void;
  (e: 'update-chain-mode', index: number, value: LongVideoChainMode): void;
  (e: 'update-compose-mode', value: string): void;
  (e: 'reset-compose-defaults'): void;
  (e: 'reset-segment-defaults'): void;
  (e: 'generate-keyframe', index: number): void;
  (e: 'generate-segment', index: number): void;
  (e: 'pick-keyframe-gallery', index: number): void;
  (e: 'clear-keyframe', index: number): void;
  (e: 'clear-segment', index: number): void;
  (e: 'select-segment', index: number): void;
  (e: 'polish-visual', index: number): void;
  (e: 'polish-motion', index: number): void;
  (e: 'pick-reference'): void;
  (e: 'remove-reference'): void;
  (e: 'pick-control'): void;
  (e: 'remove-control'): void;
  (e: 'pick-inpaint-source'): void;
  (e: 'remove-inpaint-source'): void;
  (e: 'pick-inpaint-mask'): void;
  (e: 'remove-inpaint-mask'): void;
}>();

const { t: $tt, locale } = useI18n();

const activeTab = ref<LongVideoInspectorTab>('frame');

const segmentIndex = computed(() => {
  if (!props.selection) return 0;
  if (props.selection.kind === 'segment' || props.selection.kind === 'clip') return props.selection.index;
  return 0;
});

const selectedShot = computed(() => props.shots[segmentIndex.value] ?? null);

const relatedTasksRefreshToken = computed(() =>
  `${props.keyframeGenerating ? 1 : 0}:${props.segmentGenerating ? 1 : 0}:${selectedShot.value?.id ?? ''}`,
);

const roleLabel = computed(() => {
  const role = selectedShot.value?.segment_role;
  if (!role || role === 'keyframe') return '';
  return $tt(segmentRoleLabelKey(role));
});

const inspectorTabOptions = computed(() => [
  {
    label: $tt('video.longVideoInspectorTabFrame'),
    value: 'frame' as LongVideoInspectorTab,
    disabled: false,
  },
  {
    label: $tt('video.longVideoInspectorTabClip'),
    value: 'clip' as LongVideoInspectorTab,
    disabled: !selectedShot.value || !shotVideoPrompt(selectedShot.value),
  },
]);

watch(
  () => props.selection,
  (sel) => {
    if (!sel) return;
    activeTab.value = sel.kind === 'clip' ? 'clip' : 'frame';
  },
  { immediate: true },
);

const characters = computed(() => props.characters ?? []);

const isFaceAnchor = computed(() => selectedShot.value?.segment_role === 'face_anchor');

const isTailChainSegment = computed(() => !shotNeedsKeyframe(selectedShot.value ?? undefined));

const isPostAnchorLinkedStart = computed(
  () =>
    selectedShot.value?.segment_role === 'post_anchor' ||
    selectedShot.value?.start_frame_mode === 'anchor_link',
);

const showFrameCompose = computed(
  () => shotNeedsKeyframe(selectedShot.value ?? undefined) && !isPostAnchorLinkedStart.value,
);

const showFramePreview = computed(
  () => showFrameCompose.value || isPostAnchorLinkedStart.value || Boolean(framePreviewUrl.value),
);

const tailChainHint = computed(() => {
  if (selectedShot.value?.segment_role === 'tail_continuation') {
    return $tt('video.longVideoSegmentTailFrameNoKeyframe');
  }
  return $tt('video.longVideoSegmentTailFrameHint');
});

const firstFrameRequirementText = computed(() =>
  (selectedShot.value?.first_frame_requirement || '').trim(),
);

const visibilitySummary = computed(() => {
  const shot = selectedShot.value;
  if (!shot?.first_frame_visibility) return '';
  const start = visibilityShortLabel(shot.first_frame_visibility);
  const end = shot.end_visibility ? visibilityShortLabel(shot.end_visibility) : '';
  return end && end !== start
    ? $tt('video.longVideoVisibilityRange', { start, end })
    : $tt('video.longVideoVisibilityStart', { vis: start });
});

const faceAnchorShot = computed(() =>
  selectedShot.value ? findFaceAnchorShot(props.shots, selectedShot.value) : undefined,
);

const faceAnchorIndex = computed(() => {
  const anchor = faceAnchorShot.value;
  if (!anchor) return null;
  return props.shots.findIndex((s) => s.id === anchor.id);
});

const shotSceneBeatText = computed(() => {
  const shot = selectedShot.value;
  if (!shot) return '';
  return (shot.scene_prompt || extractKeyframeShotScene(shot.visual_prompt)).trim();
});

const shotCastMatchTextValue = computed(() =>
  selectedShot.value ? shotCastMatchText(selectedShot.value) : '',
);

const keyframePromptText = computed(() => shotKeyframeText(selectedShot.value ?? undefined));

const keyframeT2iPromptPreview = computed(() => {
  if (!selectedShot.value || !showFrameCompose.value) return '';
  const visual = keyframePromptText.value;
  if (!visual && !(props.characters?.length)) return '';
  return keyframeGenerationPrompt(visual, keyframePromptCtx.value);
});

const keyframePromptCtx = computed(() => {
  if (!selectedShot.value) return { characterAnchor: props.characterAnchor ?? '' };
  return keyframePromptContextForShot(selectedShot.value, {
    character_anchor: props.characterAnchor,
    characters: props.characters,
    scenes: props.scenes,
    style_anchor: props.styleAnchor,
  });
});

const keyframeT2iProvenance = computed(() => {
  if (!selectedShot.value || !showFrameCompose.value) return null;
  const visual = keyframePromptText.value;
  if (!visual && !(props.characters?.length)) return null;
  return buildKeyframeT2iProvenance(visual, keyframePromptCtx.value);
});

const keyframePromptModeHint = computed(() => {
  if ((props.composeMode ?? 'text2img') === 'img2img') {
    return $tt('video.longVideoGenerationPromptImg2imgHint');
  }
  return $tt('video.longVideoGenerationPromptT2iHint');
});

const clipChainMode = computed(() => {
  const raw = effectiveShotChainMode(selectedShot.value ?? undefined, props.defaultChainMode);
  return raw === 'first_last' ? 'keyframe_only' : raw;
});

const canUseLastFrame = computed(() => {
  const idx = segmentIndex.value;
  if (idx <= 0) return false;
  return Boolean(props.shots[idx - 1]?.segment_asset_id);
});

const canUseReferenceR2v = computed(() => props.segmentModelSupportsR2v ?? false);

const segmentVideoPrompt = computed(() => shotVideoPrompt(selectedShot.value ?? undefined));

const segmentI2vPromptPreview = computed(() => {
  const shot = selectedShot.value;
  if (!shot || activeTab.value !== 'clip') return '';
  return segmentVideoSubmitPreview(shot, props.shots, {
    chainMode: props.defaultChainMode,
    locale: String(locale.value).startsWith('zh') ? 'zh' : 'en',
    shotIndex: segmentIndex.value,
  });
});

const segmentDurationSec = computed(() => shotDurationSec(selectedShot.value ?? undefined));

const segmentNumFrames = computed(() => {
  const schema = props.segmentParamSchema.num_frames;
  return numFramesForDurationSec(
    segmentDurationSec.value,
    props.segmentComposeParams.fps,
    schema as { min?: number; max?: number; step?: number } | undefined,
  );
});

const segmentDurationLabel = computed(() =>
  $tt('video.longVideoSegmentDurationSec', { sec: segmentDurationSec.value }),
);

const segmentVideoUrl = computed(() => {
  const id = selectedShot.value?.segment_asset_id;
  return id ? `/api/assets/${id}/file` : '';
});

const canPolishMotion = computed(() => Boolean(segmentVideoPrompt.value));

const canGenerateSegment = computed(() =>
  canGenerateSegmentShot(
    { shots: props.shots, chain_mode: props.defaultChainMode, characters: props.characters },
    segmentIndex.value,
  ),
);

const segmentMissingAnchor = computed(() => {
  const shot = selectedShot.value;
  if (!shot) return false;
  if (shot.start_frame_mode === 'anchor_link' || shot.segment_role === 'post_anchor') {
    return !faceAnchorShot.value?.keyframe_asset_id;
  }
  return false;
});

const segmentMissingKeyframe = computed(() => {
  const shot = selectedShot.value;
  if (!shot) return true;
  if (clipChainMode.value === 'last_frame' || !shotNeedsKeyframe(shot)) return false;
  if (shot.start_frame_mode === 'anchor_link') return false;
  return !shot.keyframe_asset_id;
});

function thumbForAsset(assetId: string | undefined) {
  if (!assetId) return '';
  return keyframeThumbnailUrl(assetId, (p) => api.gallery.getImageUrl(p));
}

const framePreviewUrl = computed(() => {
  if (isPostAnchorLinkedStart.value) {
    return thumbForAsset(faceAnchorShot.value?.keyframe_asset_id);
  }
  return thumbForAsset(selectedShot.value?.keyframe_asset_id);
});
</script>

<style scoped>
.lv-inspector {
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
}

.lv-inspector__tab-empty {
  margin: 12px 0;
  font-size: var(--dq-font-size-caption);
  line-height: 1.5;
  color: var(--dq-label-tertiary);
}

.lv-inspector__tail-hint {
  margin: 0 0 8px;
}

.lv-inspector__body {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding: 4px 2px 12px;
}

.lv-inspector__canvas {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.lv-inspector__canvas--sticky {
  position: sticky;
  top: 0;
  z-index: 2;
  padding-bottom: 8px;
  background: var(--dq-surface-base, var(--dq-bg));
}

.lv-inspector__media-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.lv-inspector__preview {
  position: relative;
  aspect-ratio: 16 / 9;
  border-radius: 10px;
  overflow: hidden;
  background: #000;
  border: 0.5px solid var(--dq-glass-border, var(--dq-border-subtle));
  box-shadow: inset 0 1px 0 color-mix(in srgb, white 4%, transparent);
}

.lv-inspector__preview.is-empty {
  background: color-mix(in srgb, var(--dq-surface-elevated) 40%, #000);
}

.lv-inspector__img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.lv-inspector__segment-video {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
}

.lv-inspector__clip-empty {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  margin: 0;
  padding: 16px 20px;
  text-align: center;
  font-size: var(--dq-font-size-caption);
  line-height: 1.5;
  color: var(--dq-label-tertiary);
}

.lv-inspector__tools {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-shrink: 0;
}

.lv-inspector__tool {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  height: 28px;
  padding: 0 10px;
  border: 0.5px solid var(--dq-glass-border, var(--dq-border-subtle));
  border-radius: 7px;
  background: color-mix(in srgb, var(--dq-glass-floating-bar-bg, var(--dq-surface-elevated)) 85%, transparent);
  color: var(--dq-label-secondary);
  font-size: var(--dq-font-size-caption);
  font-weight: 500;
  cursor: pointer;
}

.lv-inspector__tool:hover:not(:disabled) {
  background: color-mix(in srgb, var(--dq-accent) 12%, var(--dq-surface-elevated));
  color: var(--dq-accent);
}

.lv-inspector__tool:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.lv-inspector__tool--danger:hover:not(:disabled) {
  background: color-mix(in srgb, var(--dq-danger) 10%, var(--dq-surface-elevated));
  color: var(--dq-danger);
}

.lv-inspector__loading {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: color-mix(in srgb, var(--dq-surface-base) 62%, transparent);
  color: var(--dq-accent);
}

.lv-inspector__preview.is-generating .lv-inspector__segment-video,
.lv-inspector__preview.is-generating .lv-inspector__clip-empty {
  opacity: 0.35;
}

.lv-inspector__meta {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
  flex-wrap: wrap;
}

.lv-inspector__meta-range {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  color: var(--dq-label-secondary);
}

.lv-inspector__meta-dot {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
}

.lv-inspector__meta-size {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
}
</style>
