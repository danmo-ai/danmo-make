<template>
  <div
    v-if="preview && boxStyle"
    class="canvas-mask-preview"
    :style="boxStyle"
  >
    <img
      :src="preview.dataUrl"
      alt=""
      class="canvas-mask-preview__img"
      :draggable="false"
    />
    <span class="canvas-mask-preview__badge">{{ $t('canvas.maskPreviewBadge') }}</span>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import type { CanvasMaskPreviewState, CanvasItemState, GalleryItem } from '@/types';

const props = defineProps<{
  preview: CanvasMaskPreviewState | null;
  editingPath: string;
  items: Record<string, CanvasItemState>;
  galleryItems: GalleryItem[];
}>();

const boxStyle = computed(() => {
  if (!props.preview || !props.editingPath) return null;
  const state = props.items[props.editingPath];
  const gi = props.galleryItems.find((g) => g.path === props.editingPath);
  if (!state || !gi) return null;
  const w = (gi.width || 512) * state.scale;
  const h = (gi.height || 512) * state.scale;
  return {
    left: `${state.x}px`,
    top: `${state.y}px`,
    width: `${w}px`,
    height: `${h}px`,
  };
});
</script>

<style scoped>
.canvas-mask-preview {
  position: absolute;
  pointer-events: none;
  z-index: 3;
  border-radius: 8px;
  overflow: hidden;
  outline: 2px solid color-mix(in srgb, #e94560 75%, transparent);
  outline-offset: 1px;
}

.canvas-mask-preview__img {
  width: 100%;
  height: 100%;
  object-fit: fill;
  display: block;
  mix-blend-mode: multiply;
}

.canvas-mask-preview__badge {
  position: absolute;
  bottom: 6px;
  right: 6px;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: var(--dq-font-size-caption);
  font-weight: 700;
  letter-spacing: var(--dq-tracking-wider);
  background: rgba(233, 69, 96, 0.85);
  color: var(--dq-color-white);
}
</style>
