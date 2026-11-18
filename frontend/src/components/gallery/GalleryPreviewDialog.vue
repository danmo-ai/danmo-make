<template>
  <DqDialog
    v-model:open="dialogVisible"
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
    <template #header>
      <div class="gallery-preview-header">
        <div class="gallery-preview-header__identity" :title="resourceSummary">
          <DqIcon :size="16">
            <Picture v-if="media === 'image'" />
            <Film v-else-if="media === 'video'" />
            <Headset v-else />
          </DqIcon>
          <span class="gallery-preview-header__summary">{{ resourceSummary }}</span>
        </div>
        <p v-if="headerPrompt" class="gallery-preview-header__prompt" :title="headerPrompt">
          {{ headerPrompt }}
        </p>
        <div class="gallery-preview-header__actions">
          <span v-if="items.length > 1" class="gallery-preview-header__count">
            {{ currentIndex + 1 }} / {{ items.length }}
          </span>
          <div v-if="currentItem" class="gallery-preview-header__tip">
            <DqIconButton
              type="text"
              size="sm"
              :label="$t('gallery.assetInfo')"
              :aria-expanded="infoOpen"
              @click="infoOpen = !infoOpen"
            >
              <DqIcon :size="14">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                  <circle cx="12" cy="12" r="10" />
                  <path d="M12 16v-4" />
                  <path d="M12 8h.01" />
                </svg>
              </DqIcon>
            </DqIconButton>
            <div v-if="infoOpen && dialogVisible" class="gallery-asset-tip" role="tooltip">
              <GalleryAssetDetailMeta :item="currentItem" show-prompt />
            </div>
          </div>
          <DqIconButton
            type="text"
            size="sm"
            :label="$t('gallery.download')"
            @click="downloadCurrent"
          >
            <DqIcon :size="14"><Download /></DqIcon>
          </DqIconButton>
          <div v-if="media === 'image'" class="gallery-preview-header__zoom">
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
        </div>
      </div>
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
      <button
        type="button"
        class="gallery-preview-nav gallery-preview-nav--left"
        :disabled="!canGoPrev"
        :aria-label="$t('gallery.prev')"
        @click="goPrev"
      >
        <DqIcon><ArrowLeft /></DqIcon>
      </button>

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
          :show-meta="false"
          @download="downloadCurrent"
        />
      </div>

      <button
        type="button"
        class="gallery-preview-nav gallery-preview-nav--right"
        :disabled="!canGoNext"
        :aria-label="$t('gallery.next')"
        @click="goNext"
      >
        <DqIcon><ArrowRight /></DqIcon>
      </button>

    </div>
  </DqDialog>
</template>

<script setup lang="ts">
import { ref, computed, watch, nextTick, onBeforeUnmount } from 'vue';
import { ArrowLeft, ArrowRight, Download, Film, Headset, Picture, ZoomIn } from '@danqing/dq-shell';
import { formatGalleryResolution } from '@/utils/galleryAssetMeta';
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
const infoOpen = ref(false);
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

const headerPrompt = computed(() => (currentItem.value?.prompt || '').trim());

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
  if (props.media === 'audio') {
    const segs = currentItem.value?.metadata?.lyrics_segments;
    return Array.isArray(segs) && segs.length ? 'min(960px, 94vw)' : '640px';
  }
  return 'min(94vw, 1120px)';
});

const audioDurationLabel = computed(() => {
  const item = currentItem.value;
  if (!item) return '';
  const dur = item.duration_seconds ?? (item.metadata?.duration_seconds as number | undefined);
  if (!dur) return '';
  return formatClock(Number(dur));
});

const resourceSummary = computed(() => {
  const item = currentItem.value;
  if (!item) return $tt('gallery.assetInfo');
  const bits: string[] = [];
  if (props.media !== 'audio') {
    const resolution = formatGalleryResolution(item);
    if (resolution) bits.push(resolution);
  }
  if (props.media === 'audio' && audioDurationLabel.value) bits.push(audioDurationLabel.value);
  if (props.media === 'video' && videoDurationSeconds.value > 0) {
    bits.push(formatClock(videoDurationSeconds.value));
  }
  if (item.model) bits.push(item.model);
  if (bits.length) return bits.join(' · ');
  return item.name || $tt('gallery.assetInfo');
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
  infoOpen.value = false;
});

function onDismissInfo(event: Event) {
  if (!infoOpen.value) return;
  const target = event.target;
  if (!(target instanceof Node)) return;
  const root = document;
  const tip = root.querySelector('.gallery-asset-tip');
  const toggle = root.querySelector('.gallery-preview-header__tip');
  if (tip?.contains(target) || toggle?.contains(target)) return;
  infoOpen.value = false;
}

watch(dialogVisible, (val) => {
  if (val) {
    resetImageZoom();
    infoOpen.value = false;
    nextTick(() => {
      containerRef.value?.focus();
    });
    document.addEventListener('keydown', handleKeydown);
    document.addEventListener('pointerdown', onDismissInfo, true);
  } else {
    infoOpen.value = false;
    document.removeEventListener('keydown', handleKeydown);
    document.removeEventListener('pointerdown', onDismissInfo, true);
    resetImageZoom();
  }
});

onBeforeUnmount(() => {
  infoOpen.value = false;
  document.removeEventListener('keydown', handleKeydown);
  document.removeEventListener('pointerdown', onDismissInfo, true);
});
</script>

<style scoped>
.gallery-preview-container {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 300px;
}

.gallery-preview-container:focus-visible {
  border-radius: var(--dq-radius-control);
  box-shadow: var(--dq-focus-ring);
}

.gallery-preview-container--image {
  height: min(78vh, 720px);
  min-height: 300px;
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

.gallery-preview-header {
  position: relative;
  display: flex;
  align-items: center;
  gap: 12px;
  flex: 1;
  min-width: 0;
}

.gallery-preview-header__identity {
  display: flex;
  align-items: center;
  gap: 8px;
  flex: 0 1 auto;
  min-width: 0;
  max-width: 42%;
  color: var(--dq-label-primary);
  font-weight: 600;
}

.gallery-preview-header__summary {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.gallery-preview-header__prompt {
  flex: 1 1 auto;
  min-width: 0;
  margin: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--dq-label-tertiary);
  font-size: var(--dq-font-size-caption);
}

.gallery-preview-header__actions {
  display: flex;
  align-items: center;
  gap: 2px;
  flex: 0 0 auto;
  margin-left: auto;
}

.gallery-preview-header__count {
  margin-right: 8px;
  color: var(--dq-label-tertiary);
  font-size: var(--dq-font-size-caption);
  font-variant-numeric: tabular-nums;
}

.gallery-preview-header__zoom {
  display: inline-flex;
  align-items: center;
  margin-left: 4px;
  padding-left: 4px;
  border-left: 1px solid var(--dq-border-subtle);
}

.gallery-asset-tip {
  position: absolute;
  top: calc(100% + 10px);
  left: 0;
  width: min(440px, 72vw);
  padding: 12px 14px;
  border: 1px solid var(--dq-border-subtle);
  border-radius: var(--dq-radius-control);
  background: var(--dq-bg-base);
  box-shadow: var(--dq-shadow-popover);
  text-align: left;
  z-index: 20;
}

.gallery-asset-tip :deep(.gallery-asset-detail-meta__section) {
  display: grid;
  grid-template-columns: 5em minmax(0, 1fr) auto;
  column-gap: 12px;
  align-items: start;
  margin-bottom: 10px;
}

.gallery-asset-tip :deep(.gallery-asset-detail-meta__head) {
  display: contents;
  margin: 0;
}

.gallery-asset-tip :deep(.gallery-asset-detail-meta__prompt) {
  grid-column: 2;
  grid-row: 1;
  margin: 0;
}

.gallery-asset-tip :deep(.gallery-asset-detail-meta__head .dq-icon-btn) {
  grid-column: 3;
  grid-row: 1;
}

.gallery-asset-tip :deep(.gallery-asset-detail-meta__label),
.gallery-asset-tip :deep(.task-id-badge__label),
.gallery-asset-tip :deep(.gallery-asset-detail-meta__row dt) {
  font-size: var(--dq-font-size-caption);
  font-weight: 400;
  letter-spacing: 0;
  text-transform: none;
  color: var(--dq-label-tertiary);
}

.gallery-asset-tip :deep(.gallery-asset-detail-meta__task) {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 4px;
  margin: 0 0 10px;
}

.gallery-asset-tip :deep(.task-id-badge) {
  display: grid;
  grid-template-columns: 5em minmax(0, 1fr) auto;
  column-gap: 12px;
  align-items: center;
  width: 100%;
  margin: 0;
}

.gallery-asset-tip :deep(.task-id-badge__value) {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
}

.gallery-asset-tip :deep(.gallery-asset-detail-meta__task > .dq-btn) {
  justify-content: flex-start;
  margin-left: calc(5em + 12px);
  padding-left: 0;
}

.gallery-asset-tip :deep(.gallery-asset-detail-meta__rows) {
  display: grid;
  grid-template-columns: 5em minmax(0, 1fr);
  column-gap: 12px;
  row-gap: 8px;
  align-items: center;
  margin: 0;
}

.gallery-asset-tip :deep(.gallery-asset-detail-meta__row) {
  display: contents;
}

.gallery-asset-tip :deep(.gallery-asset-detail-meta__rows) {
  padding-top: 10px;
  border-top: 0.5px solid var(--dq-border-subtle);
}

.gallery-asset-tip :deep(.gallery-asset-detail-meta__mono) {
  min-width: 0;
}

.gallery-asset-tip :deep(.gallery-asset-detail-meta__mono code) {
  max-width: none;
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
  height: 100%;
  min-height: 0;
  padding: 20px 52px 24px;
  min-width: 0;
  transition: padding-bottom 140ms ease;
}

.gallery-preview-zoom-stage {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  min-height: 0;
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
  max-height: 100%;
  border-radius: var(--dq-radius-group);
  object-fit: contain;
  box-shadow: var(--dq-shadow-lg);
  transform-origin: center center;
  will-change: transform;
  pointer-events: none;
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
  padding: 0;
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

.gallery-preview-nav:hover:not(:disabled) {
  background: var(--dq-fill-on-glass-hover);
}

.gallery-preview-nav--left {
  left: 8px;
}

.gallery-preview-nav--right {
  right: 8px;
}

.gallery-preview-nav:disabled {
  opacity: 0.3;
  cursor: not-allowed;
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

</style>

<style>
.gallery-preview-dialog .dq-dialog-header {
  padding: 10px 14px;
  gap: 8px;
}

.gallery-preview-dialog .dq-dialog-content {
  max-height: min(94vh, 920px);
  overflow: visible;
}

.gallery-preview-dialog .dq-dialog-body {
  padding: 8px 12px 16px;
}

.gallery-preview-dialog--image .dq-dialog-body {
  padding: 0;
}

.gallery-preview-dialog--image .dq-dialog-content {
  overflow: visible;
}

.gallery-preview-dialog--audio .dq-dialog-body {
  padding: 0 4px 12px;
  overflow-y: auto;
}

.gallery-preview-dialog--video .dq-dialog-body {
  padding: 0 4px 12px;
}
</style>
