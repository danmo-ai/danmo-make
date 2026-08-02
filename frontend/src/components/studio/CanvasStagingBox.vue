<template>
  <div
    v-if="staging.visible"
    class="canvas-staging"
    :class="{ 'canvas-staging--active': active }"
    :style="boxStyle"
  >
    <div class="canvas-staging__body" @mousedown.stop="onMoveStart">
      <span class="canvas-staging__label">{{ $t('canvas.stagingLabel') }}</span>
      <span class="canvas-staging__hint">{{ $t('canvas.stagingHint') }}</span>
      <DqButton
        v-if="canSnap"
        type="text"
        size="sm"
        class="canvas-staging__snap-btn"
        @click.stop="$emit('snap-staging')"
      >
        {{ $t('canvas.snapStagingToNode') }}
      </DqButton>
    </div>
    <div
      class="canvas-staging__resize"
      :title="$t('canvas.stagingResize')"
      @mousedown.stop="onResizeStart"
    />
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount } from 'vue';
import type { CanvasStagingState } from '@/types';

const props = defineProps<{
  staging: CanvasStagingState;
  active?: boolean;
  canSnap?: boolean;
  zoom?: number;
}>();

const emit = defineEmits<{
  (e: 'move', payload: { x: number; y: number }): void;
  (e: 'resize', payload: { width: number; height: number }): void;
  (e: 'snap-staging'): void;
}>();

const boxStyle = computed(() => ({
  left: `${props.staging.x}px`,
  top: `${props.staging.y}px`,
  width: `${props.staging.width}px`,
  height: `${props.staging.height}px`,
}));

let dragging = false;
let resizing = false;
let startX = 0;
let startY = 0;
let originX = 0;
let originY = 0;
let originW = 0;
let originH = 0;

const zoomFactor = () => Math.max(0.1, props.zoom ?? 1);

function onMoveStart(e: MouseEvent) {
  if (e.button !== 0) return;
  dragging = true;
  startX = e.clientX;
  startY = e.clientY;
  originX = props.staging.x;
  originY = props.staging.y;
  window.addEventListener('mousemove', onMove);
  window.addEventListener('mouseup', onUp);
}

function onResizeStart(e: MouseEvent) {
  if (e.button !== 0) return;
  resizing = true;
  startX = e.clientX;
  startY = e.clientY;
  originW = props.staging.width;
  originH = props.staging.height;
  window.addEventListener('mousemove', onMove);
  window.addEventListener('mouseup', onUp);
}

function onMove(e: MouseEvent) {
  const z = zoomFactor();
  if (dragging) {
    emit('move', {
      x: originX + (e.clientX - startX) / z,
      y: originY + (e.clientY - startY) / z,
    });
  } else if (resizing) {
    emit('resize', {
      width: Math.max(128, originW + (e.clientX - startX) / z),
      height: Math.max(128, originH + (e.clientY - startY) / z),
    });
  }
}

function onUp() {
  dragging = false;
  resizing = false;
  window.removeEventListener('mousemove', onMove);
  window.removeEventListener('mouseup', onUp);
}

onBeforeUnmount(onUp);
</script>

<style scoped>
.canvas-staging {
  position: absolute;
  border: 2px dashed var(--dq-accent);
  border-radius: 12px;
  background: color-mix(in srgb, var(--dq-accent) 6%, transparent);
  pointer-events: none;
  z-index: 0;
  user-select: none;
}

.canvas-staging__body {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 4px;
  pointer-events: auto;
  cursor: grab;
}

.canvas-staging--active {
  border-color: var(--dq-success);
  background: color-mix(in srgb, var(--dq-success) 8%, transparent);
}

.canvas-staging__resize {
  position: absolute;
  right: -4px;
  bottom: -4px;
  width: 14px;
  height: 14px;
  border-radius: 4px;
  background: var(--dq-accent);
  pointer-events: auto;
  cursor: nwse-resize;
}

.canvas-staging__label {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  color: var(--dq-label-primary);
}

.canvas-staging__hint {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
  text-align: center;
  padding: 0 12px;
  line-height: 1.4;
}

.canvas-staging__snap-btn {
  margin-top: 4px;
  font-size: var(--dq-font-size-caption);
}
</style>
