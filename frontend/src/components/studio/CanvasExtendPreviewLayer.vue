<template>
  <template v-for="(zone, idx) in zones" :key="idx">
    <div class="canvas-extend-zone" :style="zone.style">
      <span class="canvas-extend-zone__label">{{ zone.label }}</span>
    </div>
  </template>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import type {
  CanvasExtendPreviewState,
  CanvasExtendDirection,
  CanvasItemState,
  GalleryItem,
} from '@/types';
import { computeExtendPreviewZones } from '@/utils/canvasExtendPreview';

const props = defineProps<{
  preview: CanvasExtendPreviewState;
  editingPath: string;
  items: Record<string, CanvasItemState>;
  galleryItems: GalleryItem[];
}>();

const { t } = useI18n();

const directionLabel: Record<CanvasExtendDirection, string> = {
  top: 'create.extendTop',
  bottom: 'create.extendBottom',
  left: 'create.extendLeft',
  right: 'create.extendRight',
};

const zones = computed(() =>
  computeExtendPreviewZones(
    props.preview,
    props.editingPath,
    props.items,
    props.galleryItems
  ).map((zone) => ({
    label: t(directionLabel[zone.dir]),
    style: {
      left: `${zone.x}px`,
      top: `${zone.y}px`,
      width: `${zone.width}px`,
      height: `${zone.height}px`,
    },
  }))
);
</script>

<style scoped>
.canvas-extend-zone {
  position: absolute;
  pointer-events: none;
  z-index: 3;
  border: 2px dashed color-mix(in srgb, var(--dq-accent) 75%, transparent);
  border-radius: 6px;
  background: color-mix(in srgb, var(--dq-accent) 14%, transparent);
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

.canvas-extend-zone__label {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  letter-spacing: var(--dq-tracking-wide);
  color: var(--dq-accent);
  text-transform: uppercase;
  padding: 2px 6px;
  border-radius: 4px;
  background: rgba(0, 0, 0, 0.35);
}
</style>
