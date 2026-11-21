<template>
  <div
    v-if="item"
    class="canvas-item"
    :class="{
      'canvas-item--selected': selected,
      'canvas-item--multi': multiSelected,
      'canvas-item--editing': editing,
      'canvas-item--dragging': dragging,
    }"
    :style="itemStyle"
    @mousedown.stop="onMouseDown"
    @dblclick.stop="$emit('open-preview')"
  >
    <div class="canvas-item__card">
      <div v-if="isAudio" class="canvas-item__audio-card" :style="audioCardStyle">
        <div class="canvas-item__audio-copy">
          <span v-if="audioHeading" class="canvas-item__audio-heading">{{ audioHeading }}</span>
          <span
            v-for="(line, i) in audioLyricLines"
            :key="i"
            class="canvas-item__audio-line"
          >{{ line }}</span>
          <span
            v-if="!audioLyricLines.length && audioBrief"
            class="canvas-item__audio-line canvas-item__audio-line--brief"
          >{{ audioBrief }}</span>
          <span
            v-if="!audioHeading && !audioLyricLines.length && !audioBrief"
            class="canvas-item__audio-fallback"
          >
            <DqIcon :size="20"><Headset /></DqIcon>
            {{ $t('gallery.audioLabel') }}
          </span>
        </div>
        <div class="canvas-item__audio-visualizer" aria-hidden="true">
          <span
            v-for="i in 8"
            :key="i"
            class="canvas-item__audio-bar"
            :style="{ '--bar-i': i - 1 }"
          />
        </div>
      </div>
      <template v-else-if="isVideo">
        <video
          class="canvas-item__img"
          :src="videoUrl"
          muted
          playsinline
          preload="metadata"
          :draggable="false"
        />
      </template>
      <template v-else>
        <img
          v-if="!thumbFailed"
          :src="thumbUrl"
          :alt="item.name"
          class="canvas-item__img"
          :draggable="false"
          @error="thumbFailed = true"
        />
        <div v-else class="canvas-item__img-fallback">
          <DqIcon :size="20"><PictureFilled /></DqIcon>
        </div>
      </template>

      <div v-if="relationBadge" class="canvas-item__rel-badge" :title="relationBadge">
        {{ relationBadge }}
      </div>
      <div v-if="isAudio && item.duration_seconds" class="canvas-item__badge canvas-item__badge--audio">
        {{ formatDuration(item.duration_seconds) }}
      </div>
      <div v-else-if="isVideo && item.duration_seconds" class="canvas-item__badge canvas-item__badge--video">
        {{ formatDuration(item.duration_seconds) }}
      </div>
      <div v-else-if="item.width && item.height" class="canvas-item__badge">
        {{ item.width }}×{{ item.height }}
      </div>
    </div>

    <div v-if="displayNote" class="canvas-item__note" :title="displayNote">
      {{ truncate(displayNote, 80) }}
    </div>
    <div v-else-if="displayLabel" class="canvas-item__label" :title="displayLabel">
      {{ truncate(displayLabel, 40) }}
    </div>
    <div v-else-if="item.prompt" class="canvas-item__prompt" :title="item.prompt">
      {{ truncate(item.prompt, 40) }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue';
import { PictureFilled, Headset } from '@danqing/dq-shell';
import { useI18n } from 'vue-i18n';
import type { GalleryItem, CanvasItemState } from '@/types';
import { api } from '@/utils/api';
import {
  formatGalleryDuration,
  isAudioGalleryItem,
  isVideoGalleryItem,
  previewUrlForGalleryItem,
} from '@/utils/canvasAssets';
import { lyricsSnippetLines } from '@/utils/lyricsTimeline';
import { lineageRelationLabel } from '@/utils/lineageRelationLabel';

const props = defineProps<{
  item: GalleryItem | null | undefined;
  state: CanvasItemState;
  selected: boolean;
  multiSelected?: boolean;
  editing?: boolean;
}>();

const emit = defineEmits<{
  (e: 'select', shiftKey: boolean): void;
  (e: 'open-preview'): void;
  (e: 'drag-move', delta: { dx: number; dy: number }): void;
  (e: 'drag-end', final: { x: number; y: number }): void;
}>();

const { t: $t } = useI18n();

const dragging = ref(false);
const thumbFailed = ref(false);
let dragStartX = 0;
let dragStartY = 0;
let moved = false;

const isAudio = computed(() => isAudioGalleryItem(props.item));
const isVideo = computed(() => isVideoGalleryItem(props.item));

const thumbUrl = computed(() => previewUrlForGalleryItem(props.item ?? null));

/** Most videos have no image poster; load the file like StudioCard / strip thumbs. */
const videoUrl = computed(() => {
  const item = props.item;
  if (!item?.path) return '';
  try {
    return api.gallery.getImageUrl(item.path);
  } catch {
    return thumbUrl.value;
  }
});

function formatDuration(sec: number): string {
  return formatGalleryDuration(sec);
}

const displayNote = computed(() => (props.state.note || '').trim());
const displayLabel = computed(() => (props.state.label || '').trim());

const relationBadge = computed(() => {
  const rt = String(props.item?.metadata?.relation_type || '').trim();
  if (!rt || rt === 'create') return '';
  return lineageRelationLabel(rt);
});
const audioHeading = computed(() => {
  const t = String(props.item?.title || displayLabel.value || '').trim();
  return t ? truncate(t, 28) : '';
});

const audioLyricLines = computed(() => {
  const meta = (props.item?.metadata || {}) as {
    lyrics_effective?: unknown;
    lyrics_segments?: unknown;
  };
  return lyricsSnippetLines(meta, 3);
});

const audioBrief = computed(() => {
  const p = String(props.item?.prompt || '').trim();
  if (!p) return '';
  return truncate(p, 72);
});

const audioCardStyle = computed(() => {
  const text =
    (audioLyricLines.value.join(' ') ||
      props.item?.title ||
      props.item?.prompt ||
      props.item?.name ||
      props.item?.path ||
      '').trim();
  let h = 0;
  for (let i = 0; i < text.length; i += 1) h = (h * 31 + text.charCodeAt(i)) % 360;
  return { '--dq-audio-hue': String(h) } as Record<string, string>;
});

const itemStyle = computed(() => {
  const w = isAudio.value ? 300 : props.item?.width || 512;
  const h = isAudio.value ? 168 : props.item?.height || 512;
  const sw = w * props.state.scale;
  const sh = h * props.state.scale;
  return {
    left: `${props.state.x}px`,
    top: `${props.state.y}px`,
    width: `${sw}px`,
    zIndex: props.state.zIndex,
    '--card-h': `${sh}px`,
  };
});

function truncate(text: string, max: number): string {
  return text.length > max ? text.slice(0, max) + '…' : text;
}

function onMouseDown(e: MouseEvent) {
  if (e.button !== 0) return;
  emit('select', e.shiftKey);
  dragStartX = e.clientX;
  dragStartY = e.clientY;
  moved = false;
  dragging.value = true;

  const onMove = (ev: MouseEvent) => {
    const dx = ev.clientX - dragStartX;
    const dy = ev.clientY - dragStartY;
    if (Math.abs(dx) > 2 || Math.abs(dy) > 2) {
      moved = true;
      dragStartX = ev.clientX;
      dragStartY = ev.clientY;
      emit('drag-move', { dx, dy });
    }
  };

  const onUp = () => {
    dragging.value = false;
    window.removeEventListener('mousemove', onMove);
    window.removeEventListener('mouseup', onUp);
    if (moved) {
      emit('drag-end', { x: props.state.x, y: props.state.y });
    }
  };

  window.addEventListener('mousemove', onMove);
  window.addEventListener('mouseup', onUp);
}
</script>

<style scoped>
.canvas-item {
  position: absolute;
  cursor: grab;
  transform-origin: 0 0;
  user-select: none;
}

.canvas-item--dragging {
  cursor: grabbing;
  z-index: 9999 !important;
}

.canvas-item--multi .canvas-item__card {
  outline: 2px solid var(--dq-warning);
  outline-offset: 2px;
}

.canvas-item--selected .canvas-item__card {
  outline: 2px solid var(--dq-accent);
  outline-offset: 2px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
}

.canvas-item--editing .canvas-item__card {
  outline: 2px solid var(--dq-accent);
  outline-offset: 3px;
  box-shadow:
    0 0 0 4px color-mix(in srgb, var(--dq-accent) 18%, transparent),
    0 8px 24px rgba(0, 0, 0, 0.35);
}

.canvas-item__card {
  position: relative;
  width: 100%;
  height: var(--card-h, 200px);
  border-radius: 8px;
  overflow: hidden;
  background: var(--dq-fill-tertiary);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}

.canvas-item__img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
  pointer-events: none;
}

.canvas-item__img-fallback {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--dq-label-tertiary);
}

.canvas-item__audio-card {
  position: relative;
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
  gap: 0;
  padding: 14px 14px 28px;
  background:
    linear-gradient(
      180deg,
      hsl(var(--dq-audio-hue, 220) 28% 18% / 0.35) 0%,
      transparent 42%
    ),
    linear-gradient(
      145deg,
      hsl(var(--dq-audio-hue, 220) 32% 24%),
      hsl(var(--dq-audio-hue, 220) 22% 11%)
    );
  color: var(--dq-label-primary);
}

.canvas-item__audio-copy {
  position: relative;
  z-index: 1;
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-height: 0;
  overflow: hidden;
}

.canvas-item__audio-heading {
  color: hsl(var(--dq-audio-hue, 220) 40% 92%);
  font-size: var(--dq-font-size-body);
  font-weight: 650;
  line-height: 1.25;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.canvas-item__audio-line {
  color: hsl(var(--dq-audio-hue, 220) 28% 82% / 0.94);
  font-size: var(--dq-font-size-caption);
  font-weight: 500;
  line-height: 1.4;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.canvas-item__audio-line--brief {
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 3;
  line-clamp: 3;
  white-space: normal;
  opacity: 0.88;
}

.canvas-item__audio-fallback {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: hsl(var(--dq-audio-hue, 220) 40% 78%);
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
}

.canvas-item__audio-visualizer {
  position: absolute;
  inset: auto 0 0;
  display: flex;
  align-items: flex-end;
  justify-content: center;
  gap: 3px;
  height: 20px;
  padding-bottom: 6px;
  pointer-events: none;
  opacity: 0.7;
}

.canvas-item__audio-bar {
  width: 2px;
  height: calc(4px + min(var(--bar-i), 7 - var(--bar-i)) * 3px);
  border-radius: 1px;
  background: hsl(var(--dq-audio-hue, 220) 55% 68% / 0.45);
}

.canvas-item__rel-badge {
  position: absolute;
  top: 6px;
  left: 6px;
  max-width: calc(100% - 12px);
  padding: 2px 6px;
  border-radius: 4px;
  font-size: var(--dq-font-size-caption);
  line-height: 1.2;
  background: rgba(0, 0, 0, 0.62);
  color: var(--dq-color-white);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  pointer-events: none;
}

.canvas-item__badge {
  position: absolute;
  bottom: 6px;
  right: 6px;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: var(--dq-font-size-caption);
  background: rgba(0, 0, 0, 0.55);
  color: var(--dq-color-white);
}

.canvas-item__prompt,
.canvas-item__note,
.canvas-item__label {
  margin-top: 4px;
  font-size: var(--dq-font-size-caption);
  line-height: 1.35;
  color: var(--dq-label-secondary);
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.canvas-item__note {
  color: var(--dq-accent);
  white-space: normal;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}
</style>
