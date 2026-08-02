<template>
  <section class="lv-beat-group" :data-group-id="group.groupId">
    <header class="lv-beat-group__head">
      <span class="lv-beat-group__title">{{ groupTitle }}</span>
      <div class="lv-beat-group__actions">
        <DqIconButton
          type="text"
          size="sm"
          :label="$tt('video.longVideoBeatGroupMenu')"
          @click.stop="menuOpen = !menuOpen"
        >
          <DqIcon :size="14"><MoreFilled /></DqIcon>
        </DqIconButton>
        <div v-if="menuOpen" class="lv-beat-group__menu" @click.stop>
          <button
            type="button"
            class="lv-beat-group__menu-item"
            :disabled="hasAnchor"
            @click="onInsertAnchor"
          >
            {{ $tt('video.longVideoInsertFaceAnchor') }}
          </button>
          <button type="button" class="lv-beat-group__menu-item" @click="onResplitBeat">
            {{ $tt('video.longVideoResplitBeat') }}
          </button>
          <button
            type="button"
            class="lv-beat-group__menu-item"
            :disabled="parseRegenerating"
            @click="onRegenerateParse"
          >
            {{ $tt('video.longVideoBeatRegenerate') }}
          </button>
        </div>
      </div>
    </header>
    <div class="lv-beat-group__row">
      <template v-for="(shotIdx, pos) in group.shotIndices" :key="shots[shotIdx].id">
        <div v-if="pos > 0" class="lv-beat-group__link" aria-hidden="true">→</div>
        <button
          type="button"
          class="lv-beat-node"
          :class="nodeClasses(shotIdx)"
          :aria-label="nodeAria(shots[shotIdx], shotIdx)"
          @click="$emit('select-segment', shotIdx)"
        >
          <div class="lv-beat-node__thumb" :class="{ 'is-anchor': isAnchor(shots[shotIdx]) }">
            <img v-if="thumbUrl(shots[shotIdx])" :src="thumbUrl(shots[shotIdx])" alt="" />
            <div v-else class="lv-rail__node-empty">
              <span>{{ $tt('video.longVideoKeyframePending') }}</span>
            </div>
            <span v-if="isLinkedStart(shots[shotIdx])" class="lv-beat-node__link-badge" title="linked">🔗</span>
            <span v-if="visibilityLabel(shots[shotIdx])" class="lv-beat-node__vis-badge">
              {{ visibilityLabel(shots[shotIdx]) }}
            </span>
          </div>
          <span class="lv-beat-node__role">{{ roleLabel(shots[shotIdx]) }}</span>
          <span class="lv-beat-node__index">#{{ shotIdx + 1 }}</span>
        </button>
      </template>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import { MoreFilled } from '@danqing/dq-shell';
import {
  groupHasFaceAnchor,
  segmentRoleLabelKey,
  shotDurationSec,
  extractKeyframeShotScene,
  visibilityShortLabel,
} from '@/utils/longVideoProject';
import type { LongVideoBeatGroup, LongVideoSelection, LongVideoShotState } from '@/types';

const props = defineProps<{
  group: LongVideoBeatGroup;
  shots: LongVideoShotState[];
  selection: LongVideoSelection;
  keyframeGeneratingIndex?: number | null;
  segmentGeneratingIndices?: number[];
  parseRegenerating?: boolean;
}>();

const emit = defineEmits<{
  (e: 'select-segment', index: number): void;
  (e: 'insert-anchor', groupId: string): void;
  (e: 'resplit-beat', groupId: string): void;
  (e: 'regenerate-beat', beatIndex: number): void;
}>();

const { t: $tt } = useI18n();
const menuOpen = ref(false);

const hasAnchor = computed(() => groupHasFaceAnchor(props.shots, props.group.groupId));

const groupTitle = computed(() => {
  const first = props.shots[props.group.shotIndices[0]!];
  const scene = (first?.scene_prompt || extractKeyframeShotScene(first?.visual_prompt ?? '') || '').trim();
  return scene.slice(0, 36) || $tt('video.longVideoBeatGroupFallback', { n: props.group.beatIndex + 1 });
});

function closeMenu() {
  menuOpen.value = false;
}

function onInsertAnchor() {
  closeMenu();
  emit('insert-anchor', props.group.groupId);
}

function onResplitBeat() {
  closeMenu();
  emit('resplit-beat', props.group.groupId);
}

function onRegenerateParse() {
  closeMenu();
  emit('regenerate-beat', props.group.beatIndex);
}

function onDocClick() {
  closeMenu();
}

onMounted(() => document.addEventListener('click', onDocClick));
onUnmounted(() => document.removeEventListener('click', onDocClick));

function isAnchor(shot: LongVideoShotState) {
  return shot.segment_role === 'face_anchor';
}

function isLinkedStart(shot: LongVideoShotState) {
  return shot.start_frame_mode === 'anchor_link';
}

function visibilityLabel(shot: LongVideoShotState) {
  return visibilityShortLabel(shot.first_frame_visibility);
}

function thumbUrl(shot: LongVideoShotState) {
  const id = shot.keyframe_asset_id;
  return id ? `/api/assets/${id}/thumbnail` : '';
}

function isSelected(idx: number) {
  return props.selection?.kind === 'segment' && props.selection.index === idx;
}

function nodeClasses(idx: number) {
  const shot = props.shots[idx];
  return {
    'lv-beat-node--selected': isSelected(idx),
    'lv-beat-node--ready': Boolean(shot?.keyframe_asset_id || shot?.segment_asset_id),
    'lv-beat-node--generating':
      props.keyframeGeneratingIndex === idx || props.segmentGeneratingIndices?.includes(idx),
    'lv-beat-node--anchor': shot?.segment_role === 'face_anchor',
  };
}

function roleLabel(shot: LongVideoShotState) {
  return $tt(segmentRoleLabelKey(shot.segment_role));
}

function nodeAria(shot: LongVideoShotState, idx: number) {
  return `${roleLabel(shot)} #${idx + 1} ${shotDurationSec(shot)}s`;
}
</script>

<style scoped>
.lv-beat-group__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 8px;
}

.lv-beat-group__actions {
  position: relative;
  flex-shrink: 0;
}

.lv-beat-group__menu {
  position: absolute;
  right: 0;
  top: calc(100% + 4px);
  z-index: 20;
  min-width: 168px;
  padding: 4px;
  border-radius: 8px;
  border: 0.5px solid var(--dq-glass-border, var(--dq-border-subtle));
  background: var(--dq-surface-elevated);
  box-shadow: var(--dq-shadow-popover, 0 8px 24px rgba(0, 0, 0, 0.12));
}

.lv-beat-group__menu-item {
  display: block;
  width: 100%;
  text-align: left;
  padding: 8px 10px;
  border: none;
  border-radius: 6px;
  background: transparent;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
  cursor: pointer;
}

.lv-beat-group__menu-item:hover:not(:disabled) {
  background: color-mix(in srgb, var(--dq-accent) 10%, transparent);
  color: var(--dq-accent);
}

.lv-beat-group__menu-item:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}
</style>
