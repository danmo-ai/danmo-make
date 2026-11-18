<template>
  <article
    class="studio-card"
    :class="{
      'studio-card--video': isVideo,
      'studio-card--audio': isAudio,
      'studio-card--selected': selected,
      'studio-card--selection-mode': selectionMode,
    }"
    @mouseenter="isHovered = true"
    @mouseleave="isHovered = false"
  >
    <div class="studio-card__media">
      <button
        type="button"
        class="studio-card__open"
        :aria-label="$t('gallery.preview') + ': ' + item.name"
        @click="$emit('click', $event)"
      />

      <div v-if="selectionMode" class="studio-card__checkbox">
        <button
          type="button"
          class="studio-card__checkbox-button"
          :aria-label="$t('gallery.selectMode') + ': ' + item.name"
          :aria-pressed="selected"
          @click.stop="$emit('toggle-select')"
        >
          <span class="dq-gallery-check" :class="{ 'is-checked': selected }" aria-hidden="true" />
        </button>
      </div>

      <template v-if="isImage">
        <img v-if="!thumbFailed" :src="thumbUrl" :alt="item.name" loading="lazy" @error="thumbFailed = true" />
        <div v-else class="studio-card__fallback"><DqIcon :size="32"><Picture /></DqIcon></div>
      </template>

      <template v-else-if="isVideo">
        <video
          :src="fileUrl"
          muted
          loop
          preload="metadata"
          @mouseenter="handleVideoEnter"
          @mouseleave="handleVideoLeave"
        />
        <div class="studio-card__play-overlay"><DqIcon size="26"><VideoPlay /></DqIcon></div>
      </template>

      <template v-else-if="isAudio">
        <div class="studio-card__audio-tile" :style="audioTileStyle">
          <div class="studio-card__audio-visualizer">
            <span v-for="i in 8" :key="i" class="studio-card__audio-bar" :style="{ '--bar-i': i - 1 }" />
          </div>
          <DqIcon class="studio-card__audio-icon" :size="30"><Headset /></DqIcon>
          <span class="studio-card__audio-label">{{ $t('gallery.audioLabel') }}</span>
          <span v-if="durationLabel" class="studio-card__audio-dur">{{ durationLabel }}</span>
        </div>
      </template>

      <div v-if="!selectionMode" class="studio-card__overlay">
        <div class="studio-card__overlay-actions">
          <ComposerIconTip v-for="action in primaryActions" :key="action.id" :content="action.label">
            <DqIconButton type="text" size="sm" :label="action.label" @click.stop="emitAction(action.id)">
              <DqIcon :size="14"><component :is="action.icon" /></DqIcon>
            </DqIconButton>
          </ComposerIconTip>
          <DqDropdown v-if="menuActions.length" trigger="click" size="sm" @command="emitAction">
            <ComposerIconTip :content="$t('canvas.moreMenu')">
              <DqIconButton type="text" size="sm" :label="$t('canvas.moreMenu')" @click.stop>
                <DqIcon :size="14"><Menu /></DqIcon>
              </DqIconButton>
            </ComposerIconTip>
            <template #dropdown>
              <DqDropdownMenu>
                <DqDropdownItem
                  v-for="action in menuActions"
                  :key="action.id"
                  :command="action.id"
                  :divided="action.id === 'delete'"
                >
                  <span :class="{ 'studio-card__danger-action': action.id === 'delete' }">{{ action.label }}</span>
                </DqDropdownItem>
              </DqDropdownMenu>
            </template>
          </DqDropdown>
        </div>
      </div>
    </div>

    <div class="studio-card__footer">
      <span class="studio-card__prompt" :title="item.prompt || item.name">
        {{ item.prompt || item.name }}
      </span>
      <span class="studio-card__meta">{{ resolutionLabel }}<template v-if="item.model"> · {{ item.model }}</template></span>
    </div>
  </article>
</template>

<script setup lang="ts">
import { ref, computed, type Component } from 'vue';
import { useI18n } from 'vue-i18n';
import {
  Aim,
  Brush,
  Delete,
  Download,
  Grid,
  MagicStick,
  Headset,
  Menu,
  Picture,
  Plus,
  VideoCamera,
  VideoPlay,
  ZoomIn,
} from '@danqing/dq-shell';
import { api } from '@/utils/api';
import ComposerIconTip from '@/components/studio/ComposerIconTip.vue';
import type { GalleryItem } from '@/types';

interface CardAction {
  id: string;
  label: string;
  icon: Component | string;
}

const props = defineProps<{
  item: GalleryItem;
  media: 'image' | 'video' | 'audio';
  selectionMode?: boolean;
  selected?: boolean;
  galleryCanvasMode?: boolean;
}>();

const emit = defineEmits<{
  (e: 'click', event: MouseEvent): void;
  (e: 'toggle-select'): void;
  (e: 'action', payload: { action: string; item: GalleryItem }): void;
}>();

const { t } = useI18n();
const isHovered = ref(false);
const thumbFailed = ref(false);

const isImage = computed(() => {
  if (props.item.metadata?.asset_kind === 'video' || props.item.metadata?.asset_kind === 'audio') return false;
  const ext = props.item.name?.split('.').pop()?.toLowerCase() || '';
  return ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'].includes(ext);
});

const isVideo = computed(() => {
  if (props.item.metadata?.asset_kind === 'audio') return false;
  if (props.item.metadata?.asset_kind === 'video') return true;
  const ext = props.item.name?.split('.').pop()?.toLowerCase() || '';
  return ['mp4', 'mov', 'avi', 'mkv', 'webm'].includes(ext);
});

const isAudio = computed(() => {
  if (props.item.metadata?.asset_kind === 'audio') return true;
  const ext = props.item.name?.split('.').pop()?.toLowerCase() || '';
  return ['wav', 'mp3', 'flac', 'm4a', 'aac', 'opus', 'ogg'].includes(ext);
});

const fileUrl = computed(() => api.gallery.getImageUrl(props.item.path));
const thumbUrl = computed(() => props.item.thumbnail || fileUrl.value);

const durationLabel = computed(() => {
  const raw = props.item.duration_seconds ?? (props.item.metadata?.duration_seconds as number | undefined);
  if (raw == null) return '';
  const n = Number(raw);
  if (!Number.isFinite(n) || n <= 0) return '';
  return `${Math.round(n * 10) / 10}s`;
});

const resolutionLabel = computed(() => {
  if (isAudio.value) return durationLabel.value || '—';
  return `${props.item.width || 0}×${props.item.height || 0}`;
});

const audioTileStyle = computed(() => {
  const text = props.item.prompt || props.item.name || '';
  let h = 0;
  for (let i = 0; i < text.length; i += 1) h = (h * 31 + text.charCodeAt(i)) % 360;
  return { '--dq-audio-hue': String(h) } as Record<string, string>;
});

const actions = computed<CardAction[]>(() => {
  const rows: CardAction[] = [
    { id: 'compose-from-item', label: t('studio.composeFromItem'), icon: MagicStick },
  ];
  if (props.galleryCanvasMode && (isImage.value || isVideo.value || isAudio.value)) {
    rows.push({ id: 'add-to-canvas', label: t('canvas.addToCanvas'), icon: Plus });
  } else if (isImage.value) {
    rows.push(
      { id: 'retouch', label: t('action.image.retouch'), icon: Brush },
      { id: 'extend', label: t('action.image.extend'), icon: Grid },
      { id: 'upscale', label: t('action.image.upscale'), icon: ZoomIn },
    );
  }
  if (props.media === 'video' && isImage.value) {
    rows.push({ id: 'animate', label: t('action.video.animate'), icon: VideoCamera });
  }
  if (isVideo.value && props.media === 'video') {
    rows.push(
      { id: 'video-edit', label: t('action.video.edit'), icon: VideoCamera },
      { id: 'upscale', label: t('action.video.upscale'), icon: ZoomIn },
    );
  }
  rows.push(
    { id: 'lineage', label: t('gallery.lineage'), icon: Aim },
    { id: 'download', label: t('gallery.download'), icon: Download },
    { id: 'delete', label: t('common.delete'), icon: Delete },
  );
  return rows;
});

const primaryActions = computed(() => actions.value.slice(0, props.galleryCanvasMode ? 2 : 2));
const primaryActionIds = computed(() => new Set(primaryActions.value.map((action) => action.id)));
const menuActions = computed(() => actions.value.filter((action) => !primaryActionIds.value.has(action.id)));

function emitAction(action: string) {
  emit('action', { action, item: props.item });
}

function handleVideoEnter(e: Event) {
  void (e.target as HTMLVideoElement).play().catch(() => {});
}

function handleVideoLeave(e: Event) {
  const video = e.target as HTMLVideoElement;
  video.pause();
  video.currentTime = 0;
}
</script>

<style scoped>
.studio-card {
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-width: 0;
}

.studio-card__media {
  position: relative;
  aspect-ratio: 1 / 1;
  overflow: hidden;
  border: 1px solid var(--dq-border-subtle);
  border-radius: var(--dq-radius-control);
  background: var(--dq-surface-inset);
  transition: border-color 140ms ease, box-shadow 140ms ease;
}

.studio-card:hover .studio-card__media,
.studio-card:focus-within .studio-card__media {
  border-color: var(--dq-border);
  box-shadow: var(--dq-shadow-sm);
}

.studio-card--selected .studio-card__media {
  border-color: var(--dq-accent);
  box-shadow: 0 0 0 2px color-mix(in srgb, var(--dq-accent) 20%, transparent);
}

.studio-card__open {
  position: absolute;
  inset: 0;
  z-index: 1;
  width: 100%;
  height: 100%;
  padding: 0;
  border: 0;
  background: transparent;
  cursor: pointer;
}

.studio-card__open:focus-visible {
  box-shadow: inset var(--dq-focus-ring);
}

.studio-card__media img,
.studio-card__media video {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.studio-card__fallback {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  color: var(--dq-label-tertiary);
}

.studio-card__checkbox {
  position: absolute;
  top: 8px;
  left: 8px;
  z-index: 12;
}

.studio-card__checkbox-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  padding: 0;
  border: 0;
  border-radius: var(--dq-radius-control-sm);
  background: var(--dq-bg-base);
  box-shadow: var(--dq-shadow-sm);
  cursor: pointer;
}

.studio-card__audio-tile {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  width: 100%;
  height: 100%;
  overflow: hidden;
  background: linear-gradient(145deg, hsl(var(--dq-audio-hue, 220) 26% 20%), hsl(var(--dq-audio-hue, 220) 20% 12%));
}

.studio-card__audio-visualizer {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: flex-end;
  justify-content: center;
  gap: 3px;
  height: 32px;
}

.studio-card__audio-bar {
  width: 3px;
  height: calc(8px + min(var(--bar-i), 7 - var(--bar-i)) * 6px);
  border-radius: 2px;
  background: hsl(var(--dq-audio-hue, 220) 60% 60% / 0.55);
}

.studio-card__audio-icon {
  color: hsl(var(--dq-audio-hue, 220) 50% 70%);
}

.studio-card__audio-label,
.studio-card__audio-dur {
  z-index: 1;
  color: var(--dq-label-on-media);
  font-size: var(--dq-font-size-caption);
}

.studio-card__audio-label {
  font-weight: 600;
}

.studio-card__play-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--dq-label-on-media);
  background: var(--dq-overlay-light);
  pointer-events: none;
  transition: opacity 140ms ease;
}

.studio-card:hover .studio-card__play-overlay,
.studio-card:focus-within .studio-card__play-overlay {
  opacity: 0;
}

.studio-card__overlay {
  position: absolute;
  inset: 0;
  z-index: 3;
  display: flex;
  align-items: flex-start;
  justify-content: flex-end;
  padding: 8px;
  background: linear-gradient(to bottom, var(--dq-overlay-light), transparent 42%);
  opacity: 0;
  pointer-events: none;
  transition: opacity 140ms ease;
}

.studio-card:hover .studio-card__overlay,
.studio-card:focus-within .studio-card__overlay {
  opacity: 1;
}

.studio-card__overlay-actions {
  display: flex;
  gap: 4px;
  pointer-events: auto;
}

.studio-card__overlay-actions :deep(.dq-icon-btn) {
  border: 1px solid var(--dq-border-subtle);
  border-radius: var(--dq-radius-control-sm);
  background: var(--dq-bg-base);
  box-shadow: var(--dq-shadow-sm);
}

.studio-card__danger-action {
  color: var(--dq-danger);
}

.studio-card__footer {
  display: flex;
  flex-direction: column;
  gap: 3px;
  min-width: 0;
  padding: 0 2px;
}

.studio-card__prompt,
.studio-card__meta {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.studio-card__prompt {
  color: var(--dq-label-primary);
  font-size: var(--dq-font-size-caption);
  line-height: 1.4;
}

.studio-card__meta {
  color: var(--dq-label-tertiary);
  font-size: var(--dq-font-size-caption);
  line-height: 1.35;
}

.studio-card--selection-mode .studio-card__open {
  cursor: default;
}
</style>
