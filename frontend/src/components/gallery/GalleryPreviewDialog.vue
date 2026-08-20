<template>
  <DqDialog
    v-model:open="dialogVisible"
    :title="media === 'image' ? undefined : dialogTitle"
    :width="dialogWidth"
    center
    variant="glass"
    closable
    :destroy-on-close="media === 'video' || media === 'audio'"
    :class="[
      'gallery-preview-dialog',
      media === 'image' ? 'gallery-preview-dialog--image' : '',
      media === 'audio' ? 'gallery-preview-dialog--audio' : '',
      media === 'video' ? 'gallery-preview-dialog--video' : '',
    ]"
  >
    <template v-if="media === 'image'" #header>
      <span class="gallery-preview-header-fill" aria-hidden="true" />
    </template>

    <div
      v-if="currentItem"
      ref="containerRef"
      class="gallery-preview-container"
      :class="{
        'gallery-preview-container--image': media === 'image',
        'gallery-preview-container--video': media === 'video',
        'gallery-preview-container--audio': media === 'audio',
      }"
      tabindex="0"
    >
      <div
        class="gallery-preview-nav gallery-preview-nav--left"
        :class="{ 'is-disabled': !canGoPrev }"
        @click="goPrev"
      >
        <DqIcon><ArrowLeft /></DqIcon>
      </div>

      <div
        class="gallery-preview-media"
        :class="{
          'gallery-preview-media--audio': media === 'audio',
          'gallery-preview-media--video': media === 'video',
          'gallery-preview-media--image': media === 'image',
        }"
      >
        <div
          v-if="media === 'image'"
          ref="stageRef"
          class="gallery-preview-zoom-stage"
          :class="{
            'is-zoomed': imageZoom > 1,
            'is-panning': imagePanning,
          }"
          @wheel.prevent="onImageWheel"
          @pointerdown="onImagePointerDown"
          @pointermove="onImagePointerMove"
          @pointerup="onImagePointerUp"
          @pointercancel="onImagePointerUp"
          @dblclick.prevent="onImageDblClick"
        >
          <img
            class="gallery-preview-img"
            :style="imageTransformStyle"
            :src="getImageUrl(currentItem)"
            :alt="imageCaption || currentItem.name"
            draggable="false"
          />
        </div>
        <CreateVideoPlayer
          v-else-if="media === 'video'"
          :key="currentItem.path"
          layout="gallery"
          :src="getVideoUrl(currentItem)"
          :aspect-width="currentItem.width || 0"
          :aspect-height="currentItem.height || 0"
          :duration-seconds="videoDurationSeconds"
          :show-download="true"
          @download="downloadCurrent"
        />
        <GalleryAudioDetail
          v-else-if="media === 'audio'"
          :item="currentItem"
          :src="getAudioUrl(currentItem)"
          variant="lightbox"
          :duration-label="audioDurationLabel"
          @download="downloadCurrent"
        />
      </div>

      <div
        class="gallery-preview-nav gallery-preview-nav--right"
        :class="{ 'is-disabled': !canGoNext }"
        @click="goNext"
      >
        <DqIcon><ArrowRight /></DqIcon>
      </div>

      <div v-if="media === 'video' && currentItem" class="gallery-preview-detail">
        <GalleryAssetDetailMeta :item="currentItem" show-prompt />
      </div>

      <div
        v-if="media === 'image' && currentItem"
        class="gallery-preview-detail gallery-preview-detail--image dq-glass--popover"
      >
        <GalleryAssetDetailMeta :item="currentItem" show-prompt />
      </div>

      <div v-if="media === 'image'" class="gallery-preview-zoom-bar">
        <button
          type="button"
          class="gallery-preview-zoom-btn"
          :disabled="imageZoom <= IMAGE_ZOOM_MIN"
          :aria-label="$t('gallery.zoomOut')"
          @click="nudgeImageZoom(1 / 1.25)"
        >
          <span aria-hidden="true">&minus;</span>
        </button>
        <button
          type="button"
          class="gallery-preview-zoom-label"
          :aria-label="$t('gallery.zoomReset')"
          @click="resetImageZoom"
        >
          {{ Math.round(imageZoom * 100) }}%
        </button>
        <button
          type="button"
          class="gallery-preview-zoom-btn"
          :disabled="imageZoom >= IMAGE_ZOOM_MAX"
          :aria-label="$t('gallery.zoomIn')"
          @click="nudgeImageZoom(1.25)"
        >
          <DqIcon :size="14"><ZoomIn /></DqIcon>
        </button>
      </div>

      <div v-if="items.length > 1" class="gallery-preview-counter">
        {{ currentIndex + 1 }} / {{ items.length }}
      </div>
    </div>
  </DqDialog>
</template>

<script setup lang="ts">
import { ref, computed, watch, nextTick, onBeforeUnmount } from 'vue';
import { ArrowLeft, ArrowRight, ZoomIn } from '@danqing/dq-shell';
import { api } from '@/utils/api';
import { $tt } from '@/utils/i18n';
import type { GalleryItem } from '@/types';
import CreateVideoPlayer from '@/components/create/CreateVideoPlayer.vue';
import GalleryAudioDetail from '@/components/gallery/GalleryAudioDetail.vue';
import GalleryAssetDetailMeta from '@/components/gallery/GalleryAssetDetailMeta.vue';

const props = defineProps<{
  open: boolean;
  items: GalleryItem[];
  index: number;
  media: 'image' | 'video' | 'audio';
}>();

const emit = defineEmits<{
  (e: 'update:open', value: boolean): void;
  (e: 'update:index', value: number): void;
}>();

const dialogVisible = computed({
  get: () => props.open,
  set: (val) => emit('update:open', val),
});

const containerRef = ref<HTMLElement | null>(null);
const stageRef = ref<HTMLElement | null>(null);

const IMAGE_ZOOM_MIN = 1;
const IMAGE_ZOOM_MAX = 8;
const imageZoom = ref(1);
const imagePanX = ref(0);
const imagePanY = ref(0);
const imagePanning = ref(false);
const panLastX = ref(0);
const panLastY = ref(0);

const imageTransformStyle = computed(() => ({
  transform: `translate(${imagePanX.value}px, ${imagePanY.value}px) scale(${imageZoom.value})`,
}));

function resetImageZoom() {
  imageZoom.value = 1;
  imagePanX.value = 0;
  imagePanY.value = 0;
  imagePanning.value = false;
}

function clampImageZoom(next: number) {
  return Math.min(IMAGE_ZOOM_MAX, Math.max(IMAGE_ZOOM_MIN, next));
}

function applyImageZoomAt(clientX: number, clientY: number, nextZoom: number) {
  const stage = stageRef.value;
  const z = imageZoom.value;
  const nz = clampImageZoom(nextZoom);
  if (nz === z) {
    if (nz <= IMAGE_ZOOM_MIN) resetImageZoom();
    return;
  }
  if (!stage) {
    imageZoom.value = nz;
    if (nz <= IMAGE_ZOOM_MIN) resetImageZoom();
    return;
  }
  const rect = stage.getBoundingClientRect();
  const mx = clientX - rect.left;
  const my = clientY - rect.top;
  const cx = rect.width / 2;
  const cy = rect.height / 2;
  const ratio = nz / z;
  imagePanX.value = (mx - cx) - ratio * ((mx - cx) - imagePanX.value);
  imagePanY.value = (my - cy) - ratio * ((my - cy) - imagePanY.value);
  imageZoom.value = nz;
  if (nz <= IMAGE_ZOOM_MIN) resetImageZoom();
}

function nudgeImageZoom(factor: number) {
  const stage = stageRef.value;
  if (!stage) {
    const nz = clampImageZoom(imageZoom.value * factor);
    imageZoom.value = nz;
    if (nz <= IMAGE_ZOOM_MIN) resetImageZoom();
    return;
  }
  const rect = stage.getBoundingClientRect();
  applyImageZoomAt(rect.left + rect.width / 2, rect.top + rect.height / 2, imageZoom.value * factor);
}

function onImageWheel(e: WheelEvent) {
  if (props.media !== 'image') return;
  const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
  applyImageZoomAt(e.clientX, e.clientY, imageZoom.value * factor);
}

function onImagePointerDown(e: PointerEvent) {
  if (props.media !== 'image') return;
  if (e.button !== 0) return;
  if (imageZoom.value <= IMAGE_ZOOM_MIN) return;
  imagePanning.value = true;
  panLastX.value = e.clientX;
  panLastY.value = e.clientY;
  (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
}

function onImagePointerMove(e: PointerEvent) {
  if (!imagePanning.value) return;
  imagePanX.value += e.clientX - panLastX.value;
  imagePanY.value += e.clientY - panLastY.value;
  panLastX.value = e.clientX;
  panLastY.value = e.clientY;
}

function onImagePointerUp(e: PointerEvent) {
  if (!imagePanning.value) return;
  imagePanning.value = false;
  const el = e.currentTarget as HTMLElement | null;
  if (el?.hasPointerCapture(e.pointerId)) {
    el.releasePointerCapture(e.pointerId);
  }
}

function onImageDblClick(e: MouseEvent) {
  if (props.media !== 'image') return;
  if (imageZoom.value > 1) {
    resetImageZoom();
    return;
  }
  applyImageZoomAt(e.clientX, e.clientY, 2.5);
}

const currentIndex = computed({
  get: () => props.index,
  set: (val) => emit('update:index', val),
});

const currentItem = computed(() => {
  if (currentIndex.value < 0 || currentIndex.value >= props.items.length) {
    return null;
  }
  return props.items[currentIndex.value];
});

const dialogTitle = computed(() => {
  const item = currentItem.value;
  if (!item) return '';
  const prompt = (item.prompt || '').trim();
  if (prompt) {
    return prompt.length > 52 ? `${prompt.slice(0, 52)}…` : prompt;
  }
  return item.name || $tt('gallery.preview');
});

const imageCaption = computed(() => (currentItem.value?.prompt || '').trim());

const canGoPrev = computed(() => currentIndex.value > 0);
const canGoNext = computed(() => currentIndex.value < props.items.length - 1);

const dialogWidth = computed(() => {
  if (props.media === 'video') {
    const item = currentItem.value;
    const w = item?.width || 0;
    const h = item?.height || 0;
    if (w > 0 && h > w) return 'min(440px, 92vw)';
    if (w > 0 && h > 0 && w >= h) return 'min(860px, 92vw)';
    return 'min(640px, 92vw)';
  }
  if (props.media === 'audio') return '640px';
  return 'min(94vw, 1120px)';
});

const audioDurationLabel = computed(() => {
  const item = currentItem.value;
  if (!item) return '';
  const dur = item.duration_seconds ?? (item.metadata?.duration_seconds as number | undefined);
  if (!dur) return '';
  return formatClock(Number(dur));
});

const videoDurationSeconds = computed(() => {
  const item = currentItem.value;
  if (!item) return 0;
  const raw = item.duration_seconds ?? (item.metadata?.duration_seconds as number | undefined);
  const n = Number(raw);
  return Number.isFinite(n) && n > 0 ? n : 0;
});

function formatClock(sec: number) {
  const s = Math.max(0, Math.floor(sec || 0));
  const m = Math.floor(s / 60);
  return m + ':' + String(s % 60).padStart(2, '0');
}

function getImageUrl(item: GalleryItem): string {
  return api.gallery.getImageUrl(item.path);
}

function getVideoUrl(item: GalleryItem): string {
  return api.gallery.getImageUrl(item.path);
}

function getAudioUrl(item: GalleryItem): string {
  return api.gallery.getImageUrl(item.path);
}

function downloadCurrent() {
  const item = currentItem.value;
  if (!item) return;
  const a = document.createElement('a');
  a.href = api.gallery.getImageUrl(item.path);
  a.download = item.name || 'download';
  a.click();
}

function goPrev() {
  if (!canGoPrev.value) return;
  resetImageZoom();
  currentIndex.value--;
}

function goNext() {
  if (!canGoNext.value) return;
  resetImageZoom();
  currentIndex.value++;
}

function handleKeydown(e: KeyboardEvent) {
  if (!dialogVisible.value) return;
  if (e.key === 'ArrowLeft') {
    e.preventDefault();
    goPrev();
  } else if (e.key === 'ArrowRight') {
    e.preventDefault();
    goNext();
  } else if (e.key === '+' || e.key === '=') {
    if (props.media === 'image') {
      e.preventDefault();
      nudgeImageZoom(1.25);
    }
  } else if (e.key === '-' || e.key === '_') {
    if (props.media === 'image') {
      e.preventDefault();
      nudgeImageZoom(1 / 1.25);
    }
  } else if (e.key === '0') {
    if (props.media === 'image') {
      e.preventDefault();
      resetImageZoom();
    }
  } else if (e.key === 'Escape') {
    dialogVisible.value = false;
  }
}

watch(currentIndex, () => {
  resetImageZoom();
});

watch(dialogVisible, (val) => {
  if (val) {
    resetImageZoom();
    nextTick(() => {
      containerRef.value?.focus();
    });
    document.addEventListener('keydown', handleKeydown);
  } else {
    document.removeEventListener('keydown', handleKeydown);
    resetImageZoom();
  }
});

onBeforeUnmount(() => {
  document.removeEventListener('keydown', handleKeydown);
});
</script>

<style scoped>
.gallery-preview-container {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 300px;
  outline: none;
}

.gallery-preview-container--image {
  min-height: min(78vh, 720px);
}

.gallery-preview-container--video {
  display: grid;
  grid-template-columns: 40px minmax(0, 1fr) 40px;
  column-gap: 8px;
  align-items: start;
  padding: 0 4px 8px;
  min-height: 0;
}

.gallery-preview-container--video .gallery-preview-nav {
  position: static;
  top: auto;
  transform: none;
  align-self: center;
  justify-self: center;
}

.gallery-preview-container--video .gallery-preview-nav--left,
.gallery-preview-container--video .gallery-preview-nav--right {
  left: auto;
  right: auto;
}

.gallery-preview-container--video .gallery-preview-media {
  grid-column: 2;
  grid-row: 1;
  padding: 8px 0 12px;
}

.gallery-preview-container--video .gallery-preview-detail {
  position: static;
  grid-column: 2;
  grid-row: 2;
  left: auto;
  right: auto;
  bottom: auto;
  margin: 0;
  max-height: none;
  box-shadow: none;
}

.gallery-preview-container--video .gallery-preview-counter {
  grid-column: 1 / -1;
  position: static;
  justify-self: center;
  margin-top: 8px;
  transform: none;
}

.gallery-preview-header-fill {
  flex: 1;
}

.gallery-preview-media {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px 56px;
  width: 100%;
  box-sizing: border-box;
}

.gallery-preview-container--image .gallery-preview-media {
  padding: 12px 52px 220px;
  min-height: min(78vh, 720px);
  min-width: 0;
}

.gallery-preview-zoom-stage {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: min(72vh, calc(78vh - 80px));
  min-height: 240px;
  overflow: hidden;
  touch-action: none;
  cursor: zoom-in;
  user-select: none;
}

.gallery-preview-zoom-stage.is-zoomed {
  cursor: grab;
}

.gallery-preview-zoom-stage.is-panning {
  cursor: grabbing;
}

.gallery-preview-img {
  max-width: 100%;
  max-height: 72vh;
  border-radius: var(--dq-radius-group);
  object-fit: contain;
  box-shadow: var(--dq-shadow-lg);
  transform-origin: center center;
  will-change: transform;
  pointer-events: none;
}

.gallery-preview-zoom-bar {
  position: absolute;
  right: 56px;
  top: 12px;
  left: auto;
  bottom: auto;
  z-index: 11;
  display: flex;
  align-items: center;
  gap: 2px;
  padding: 2px;
  border-radius: var(--dq-radius-pill);
  border: 0.5px solid var(--dq-glass-border);
  background: var(--dq-glass-tooltip-bg);
  -webkit-backdrop-filter: var(--dq-glass-blur-light);
  backdrop-filter: var(--dq-glass-blur-light);
}

.gallery-preview-zoom-btn,
.gallery-preview-zoom-label {
  appearance: none;
  border: 0;
  background: transparent;
  color: var(--dq-label-secondary);
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 28px;
  height: 28px;
  padding: 0 6px;
  border-radius: var(--dq-radius-pill);
  font-size: var(--dq-font-size-caption);
  font-variant-numeric: tabular-nums;
}

.gallery-preview-zoom-btn:hover:not(:disabled),
.gallery-preview-zoom-label:hover {
  background: var(--dq-fill-on-glass-hover);
  color: var(--dq-label-primary);
}

.gallery-preview-zoom-btn:disabled {
  opacity: 0.35;
  cursor: not-allowed;
}

.gallery-preview-media--audio {
  padding: 16px 0 32px;
  min-width: 0;
  width: 100%;
}

.gallery-preview-media--video {
  padding: 8px 0 12px;
  min-width: 0;
  width: 100%;
}

.gallery-preview-container--audio {
  display: grid;
  grid-template-columns: 40px minmax(0, 1fr) 40px;
  column-gap: 8px;
  align-items: start;
  padding: 0 4px;
}

.gallery-preview-container--audio .gallery-preview-nav {
  position: static;
  top: auto;
  transform: none;
  align-self: center;
  justify-self: center;
}

.gallery-preview-container--audio .gallery-preview-nav--left,
.gallery-preview-container--audio .gallery-preview-nav--right {
  left: auto;
  right: auto;
}

.gallery-preview-container--audio .gallery-preview-media {
  grid-column: 2;
}

.gallery-preview-nav {
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  color: var(--dq-label-primary);
  transition: background-color 0.15s ease, opacity 0.15s ease;
  z-index: 10;
  border-radius: 50%;
  border: 0.5px solid var(--dq-glass-border);
  background: var(--dq-glass-tooltip-bg);
  -webkit-backdrop-filter: var(--dq-glass-blur-light);
  backdrop-filter: var(--dq-glass-blur-light);
}

.gallery-preview-nav:hover:not(.is-disabled) {
  background: var(--dq-fill-on-glass-hover);
}

.gallery-preview-nav--left {
  left: 8px;
}

.gallery-preview-nav--right {
  right: 8px;
}

.gallery-preview-nav.is-disabled {
  opacity: 0.15;
  cursor: not-allowed;
  pointer-events: none;
}

.gallery-preview-caption {
  position: absolute;
  left: 50%;
  bottom: 44px;
  transform: translateX(-50%);
  width: min(560px, calc(100% - 96px));
  padding: 10px 14px;
  z-index: 10;
  text-align: center;
}

.gallery-preview-caption__text {
  margin: 0 0 4px;
  font-size: var(--dq-font-size-body);
  line-height: 1.45;
  color: var(--dq-label-primary);
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.gallery-preview-caption__meta {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
  letter-spacing: 0.01em;
}

.gallery-preview-counter {
  position: absolute;
  bottom: 12px;
  left: 50%;
  transform: translateX(-50%);
  padding: 4px 12px;
  background: var(--dq-glass-tooltip-bg);
  color: var(--dq-label-secondary);
  border-radius: var(--dq-radius-pill);
  font-size: var(--dq-font-size-caption);
  font-variant-numeric: tabular-nums;
  z-index: 10;
  border: 0.5px solid var(--dq-glass-border);
  -webkit-backdrop-filter: var(--dq-glass-blur-light);
  backdrop-filter: var(--dq-glass-blur-light);
}

.gallery-preview-detail {
  position: absolute;
  bottom: 48px;
  left: 60px;
  right: 60px;
  padding: 14px 18px;
  background: var(--dq-surface-inset);
  border: 0.5px solid var(--dq-border-subtle);
  border-radius: var(--dq-radius-group);
  color: var(--dq-label-primary);
  z-index: 10;
  max-height: min(240px, 34vh);
  overflow-y: auto;
  box-shadow: var(--dq-shadow-md);
}

.gallery-preview-detail--image {
  left: 50%;
  right: auto;
  transform: translateX(-50%);
  width: min(640px, calc(100% - 96px));
  bottom: 44px;
}

.gallery-preview-detail__section {
  margin-bottom: 10px;
}

.gallery-preview-detail__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 6px;
}

.gallery-preview-detail__label {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  letter-spacing: var(--dq-tracking-wide);
  text-transform: uppercase;
  color: var(--dq-label-tertiary);
}

.gallery-preview-detail__prompt {
  margin: 0;
  font-size: var(--dq-font-size-body);
  line-height: 1.5;
  color: var(--dq-label-primary);
  word-break: break-word;
}

.gallery-preview-detail__meta {
  display: flex;
  flex-wrap: wrap;
  gap: 10px 20px;
  margin: 0;
  padding-top: 10px;
  border-top: 0.5px solid var(--dq-border-subtle);
}

.gallery-preview-detail__meta-row {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: var(--dq-font-size-caption);
}

.gallery-preview-detail__meta-row dt {
  color: var(--dq-label-tertiary);
}

.gallery-preview-detail__meta-row dd {
  margin: 0;
  color: var(--dq-label-secondary);
}
</style>

<style>
.gallery-preview-dialog .dq-dialog-content {
  max-height: min(94vh, 920px);
}

.gallery-preview-dialog .dq-dialog-body {
  padding: 8px 12px 16px;
}

.gallery-preview-dialog--image .dq-dialog-body {
  padding: 0;
}

.gallery-preview-dialog--image .dq-dialog-content {
  overflow: hidden;
}

.gallery-preview-dialog--audio .dq-dialog-body {
  padding: 0 4px 12px;
}

.gallery-preview-dialog--video .dq-dialog-body {
  padding: 0 4px 12px;
}
</style>
