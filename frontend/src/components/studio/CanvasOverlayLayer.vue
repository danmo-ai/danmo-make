<template>
  <div
    v-if="layer && layer.visible && galleryItem"
    class="canvas-overlay"
    :class="`canvas-overlay--${kind}`"
    :style="boxStyle"
    @mousedown.stop="onMoveStart"
  >
    <img
      :src="thumbUrl"
      :alt="kind"
      class="canvas-overlay__img"
      :draggable="false"
    />
    <span class="canvas-overlay__badge">{{ badge }}</span>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount } from 'vue';
import type { CanvasOverlayLayer, CanvasOverlayKind, GalleryItem } from '@/types';
import { previewUrlForGalleryItem } from '@/utils/canvasAssets';
import { OVERLAY_BADGES } from '@/utils/canvasOverlays';

const props = defineProps<{
  kind: CanvasOverlayKind;
  layer: CanvasOverlayLayer | null | undefined;
  galleryItems: GalleryItem[];
  zoom?: number;
}>();

const emit = defineEmits<{
  (e: 'move', payload: { x: number; y: number }): void;
}>();

const galleryItem = computed(() => {
  const path = props.layer?.path;
  if (!path) return null;
  return props.galleryItems.find((g) => g.path === path) ?? null;
});

const thumbUrl = computed(() => {
  const gi = galleryItem.value;
  if (!gi) return '';
  return previewUrlForGalleryItem(gi);
});

const badge = computed(() => OVERLAY_BADGES[props.kind] || props.kind.toUpperCase());

const boxStyle = computed(() => {
  const layer = props.layer;
  const gi = galleryItem.value;
  if (!layer || !gi) return {};
  const w = (gi.width || 512) * layer.scale;
  const h = (gi.height || 512) * layer.scale;
  return {
    left: `${layer.x}px`,
    top: `${layer.y}px`,
    width: `${w}px`,
    height: `${h}px`,
    opacity: String(layer.opacity),
  };
});

let dragging = false;
let startX = 0;
let startY = 0;
let originX = 0;
let originY = 0;

const zoomFactor = () => Math.max(0.1, props.zoom ?? 1);

function onMoveStart(e: MouseEvent) {
  if (e.button !== 0 || !props.layer) return;
  dragging = true;
  startX = e.clientX;
  startY = e.clientY;
  originX = props.layer.x;
  originY = props.layer.y;
  window.addEventListener('mousemove', onMove);
  window.addEventListener('mouseup', onMoveEnd);
}

function onMove(e: MouseEvent) {
  if (!dragging || !props.layer) return;
  const z = zoomFactor();
  const dx = (e.clientX - startX) / z;
  const dy = (e.clientY - startY) / z;
  emit('move', { x: originX + dx, y: originY + dy });
}

function onMoveEnd() {
  dragging = false;
  window.removeEventListener('mousemove', onMove);
  window.removeEventListener('mouseup', onMoveEnd);
}

onBeforeUnmount(() => {
  window.removeEventListener('mousemove', onMove);
  window.removeEventListener('mouseup', onMoveEnd);
});
</script>

<style scoped>
.canvas-overlay {
  position: absolute;
  pointer-events: auto;
  z-index: 2;
  border-radius: 8px;
  overflow: hidden;
  cursor: grab;
}

.canvas-overlay:active {
  cursor: grabbing;
}

.canvas-overlay--reference {
  outline: 2px dashed color-mix(in srgb, var(--dq-success) 70%, transparent);
}

.canvas-overlay--control {
  outline: 2px dashed color-mix(in srgb, var(--dq-warning) 70%, transparent);
}

.canvas-overlay--start_frame {
  outline: 2px dashed color-mix(in srgb, var(--dq-accent) 70%, transparent);
}

.canvas-overlay--tail_frame {
  outline: 2px dashed color-mix(in srgb, #9b59b6 70%, transparent);
}

.canvas-overlay--video_source {
  outline: 2px dashed color-mix(in srgb, var(--dq-danger) 65%, transparent);
}

.canvas-overlay--cover_source {
  outline: 2px dashed color-mix(in srgb, #c678dd 70%, transparent);
}

.canvas-overlay__img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
  pointer-events: none;
}

.canvas-overlay__badge {
  position: absolute;
  top: 6px;
  left: 6px;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: var(--dq-font-size-caption);
  font-weight: 700;
  letter-spacing: var(--dq-tracking-wider);
  background: rgba(0, 0, 0, 0.55);
  color: var(--dq-color-white);
  pointer-events: none;
}
</style>
