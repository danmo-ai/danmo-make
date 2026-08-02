<template>
  <div
    ref="viewportRef"
    class="canvas-viewport"
    :class="{
      'canvas-viewport--space-pan': spaceHeld && pointerInside,
      'canvas-viewport--panning': isPanning,
    }"
    @wheel.prevent="onWheel"
    @mousedown="onViewportMouseDown"
    @mouseenter="pointerInside = true"
    @mouseleave="pointerInside = false"
    @contextmenu.prevent
  >
    <div class="canvas-viewport__grid" :style="gridStyle" aria-hidden="true" />

    <div class="canvas-viewport__world" :style="worldStyle">
      <CanvasEdges
        v-if="showEdges && (edges?.length ?? 0) > 0"
        :edges="edges ?? []"
        :items="items"
        :gallery-items="galleryItems"
        :focus-paths="edgeFocusPaths"
      />

      <CanvasRegionGuides
        :visible="!!showRegionGuides"
        :staging="staging"
        :overlays="overlays"
        :gallery-items="galleryItems"
        :media="media"
      />

      <CanvasStagingBox
        v-if="staging"
        :staging="staging"
        :zoom="viewport.zoom"
        :active="!!selectedPath"
        :can-snap="!!selectedPath"
        @move="$emit('staging-move', $event)"
        @resize="$emit('staging-resize', $event)"
        @snap-staging="$emit('snap-staging')"
      />

      <CanvasOverlayLayer
        v-for="kind in overlayKinds"
        :key="kind"
        :kind="kind"
        :layer="overlays?.[kind]"
        :gallery-items="galleryItems"
        :zoom="viewport.zoom"
        @move="$emit('overlay-move', { kind, ...$event })"
      />

      <CanvasMaskPreviewLayer
        v-if="maskPreview && editingPath"
        :preview="maskPreview"
        :editing-path="editingPath"
        :items="items"
        :gallery-items="galleryItems"
      />

      <CanvasExtendPreviewLayer
        v-if="extendPreview && editingPath"
        :preview="extendPreview"
        :editing-path="editingPath"
        :items="items"
        :gallery-items="galleryItems"
      />

      <div
        v-if="marquee.active"
        class="canvas-viewport__marquee"
        :style="marqueeStyle"
      />

      <CanvasItem
        v-for="[path, state] in visibleEntries"
        :key="path"
        :item="itemMap[path]!"
        :state="state"
        :selected="selectedPath === path"
        :multi-selected="selectedPaths.includes(path) && selectedPath !== path"
        :editing="path === editingPath"
        @select="(shift) => $emit('select-item', { path, shiftKey: shift })"
        @open-preview="$emit('open-preview', path)"
        @drag-move="onItemDragMove(path, $event)"
        @drag-end="onItemDragEnd(path, $event)"
      />
    </div>

    <CanvasItemToolbar
      v-if="selectedPaths.length <= 1"
      :visible="!!selectedItem && !!toolbarPos"
      :item="selectedItem"
      :left="toolbarPos?.left ?? 0"
      :top="toolbarPos?.top ?? 0"
      :placement="toolbarPos?.placement ?? 'above'"
      :node-height="toolbarPos?.nodeHeight ?? 0"
      :media="media"
      :describing="describing"
      @action="(action) => $emit('toolbar-action', action)"
    />

    <div v-if="visibleEntries.length === 0" class="canvas-viewport__empty">
      <div class="canvas-viewport__empty-card dq-glass--popover">
        <p class="canvas-viewport__empty-title">{{ $t('canvas.emptyTitle') }}</p>
        <p class="canvas-viewport__empty-hint">{{ $t('canvas.emptyHint') }}</p>
        <ol class="canvas-viewport__empty-steps">
          <li>{{ $t('canvas.emptyStep1') }}</li>
          <li>{{ $t('canvas.emptyStep2') }}</li>
          <li>{{ $t('canvas.emptyStep3') }}</li>
          <li>{{ $t('canvas.emptyStep4') }}</li>
        </ol>
        <ul class="canvas-viewport__empty-shortcuts">
          <li>{{ $t('canvas.shortcutLayers') }}</li>
          <li>{{ $t('canvas.shortcutRename') }}</li>
          <li>{{ $t('canvas.shortcutPan') }}</li>
          <li>{{ $t('canvas.shortcutRemove') }}</li>
          <li>{{ $t('canvas.scaleHint') }}</li>
        </ul>
        <DqButton type="primary" @click="$emit('open-import-picker')">
          {{ $t('canvas.importWorks') }}
        </DqButton>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, reactive, onMounted, onBeforeUnmount } from 'vue';
import { readComposerReservePx } from '@/utils/composerReserve';
import CanvasItem from '@/components/studio/CanvasItem.vue';
import CanvasItemToolbar from '@/components/studio/CanvasItemToolbar.vue';
import CanvasStagingBox from '@/components/studio/CanvasStagingBox.vue';
import CanvasEdges from '@/components/studio/CanvasEdges.vue';
import CanvasOverlayLayer from '@/components/studio/CanvasOverlayLayer.vue';
import CanvasMaskPreviewLayer from '@/components/studio/CanvasMaskPreviewLayer.vue';
import CanvasExtendPreviewLayer from '@/components/studio/CanvasExtendPreviewLayer.vue';
import CanvasRegionGuides from '@/components/studio/CanvasRegionGuides.vue';
import type {
  GalleryItem,
  CanvasItemState,
  CanvasViewportState,
  CanvasStagingState,
  CanvasOverlaysState,
  CanvasOverlayKind,
  CanvasMaskPreviewState,
  CanvasExtendPreviewState,
  CanvasEdge,
} from '@/types';
import { OVERLAY_KINDS_BY_MEDIA } from '@/utils/canvasOverlays';

const props = defineProps<{
  items: Record<string, CanvasItemState>;
  viewport: CanvasViewportState;
  staging?: CanvasStagingState | null;
  overlays?: CanvasOverlaysState | null;
  edges?: CanvasEdge[];
  showEdges?: boolean;
  showRegionGuides?: boolean;
  galleryItems: GalleryItem[];
  selectedPath: string;
  selectedPaths: string[];
  media?: import('@/composables/useCanvasStore').CanvasMedia;
  describing?: boolean;
  editingPath?: string;
  maskPreview?: CanvasMaskPreviewState | null;
  extendPreview?: CanvasExtendPreviewState | null;
}>();

const emit = defineEmits<{
  (e: 'update:viewport', vp: CanvasViewportState): void;
  (e: 'select-item', payload: { path: string; shiftKey: boolean }): void;
  (e: 'clear-selection'): void;
  (e: 'marquee-select', rect: { x: number; y: number; w: number; h: number }): void;
  (e: 'item-drag-move', payload: { path: string; dx: number; dy: number }): void;
  (e: 'item-drag-end', payload: { path: string; x: number; y: number }): void;
  (e: 'toolbar-action', action: string): void;
  (e: 'scale-item', payload: { path: string; scale: number }): void;
  (e: 'open-import-picker'): void;
  (e: 'open-preview', path: string): void;
  (e: 'staging-move', payload: { x: number; y: number }): void;
  (e: 'staging-resize', payload: { width: number; height: number }): void;
  (e: 'snap-staging'): void;
  (e: 'overlay-move', payload: { kind: CanvasOverlayKind; x: number; y: number }): void;
}>();

const overlayKinds = computed(() => {
  const media = props.media || 'image';
  return OVERLAY_KINDS_BY_MEDIA[media].filter((kind) => props.overlays?.[kind]?.path);
});

const edgeFocusPaths = computed(() => {
  if (props.selectedPaths.length > 0) return props.selectedPaths;
  return props.selectedPath ? [props.selectedPath] : [];
});

const viewportRef = ref<HTMLElement | null>(null);

const worldStyle = computed(() => ({
  transform: `translate(${props.viewport.panX}px, ${props.viewport.panY}px) scale(${props.viewport.zoom})`,
  transformOrigin: '0 0',
}));

/** Screen-fixed dot grid synced to pan/zoom (must not live inside transformed world). */
const gridStyle = computed(() => {
  const step = 20 * props.viewport.zoom;
  return {
    backgroundSize: `${step}px ${step}px`,
    backgroundPosition: `${props.viewport.panX}px ${props.viewport.panY}px`,
  };
});

const itemMap = computed(() => {
  const map: Record<string, GalleryItem> = {};
  for (const gi of props.galleryItems) {
    map[gi.path] = gi;
  }
  return map;
});

const visibleEntries = computed(() =>
  Object.entries(props.items)
    .filter(([path, s]) => s.visible && itemMap.value[path] != null)
    .sort((a, b) => a[1].zIndex - b[1].zIndex)
);

const selectedItem = computed(() =>
  props.selectedPath ? itemMap.value[props.selectedPath] ?? null : null
);

const spaceHeld = ref(false);
const pointerInside = ref(false);
const isPanning = ref(false);

const toolbarPos = computed(() => {
  if (!props.selectedPath || !viewportRef.value) return null;
  const state = props.items[props.selectedPath];
  const gi = itemMap.value[props.selectedPath];
  if (!state || !gi) return null;
  const w = (gi.width || 512) * state.scale;
  const h = (gi.height || 512) * state.scale;
  const zoom = props.viewport.zoom;
  const nodeHeight = h * zoom;
  const left = props.viewport.panX + (state.x + w / 2) * zoom;
  const top = props.viewport.panY + state.y * zoom;
  const vh = viewportRef.value.clientHeight;
  const layoutEl = viewportRef.value.closest('.studio-layout') as HTMLElement | null;
  const reserve = readComposerReservePx(layoutEl, vh);
  const toolbarBand = 52;
  const tooHigh = top < toolbarBand;
  const nearComposer = top + nodeHeight * 0.35 > vh - reserve;
  const placement: 'above' | 'below' =
    tooHigh || nearComposer ? 'below' : 'above';
  return { left, top, nodeHeight, placement };
});

const marquee = reactive({
  active: false,
  x1: 0,
  y1: 0,
  x2: 0,
  y2: 0,
});

const marqueeStyle = computed(() => {
  const x = Math.min(marquee.x1, marquee.x2);
  const y = Math.min(marquee.y1, marquee.y2);
  const w = Math.abs(marquee.x2 - marquee.x1);
  const h = Math.abs(marquee.y2 - marquee.y1);
  return {
    left: `${x}px`,
    top: `${y}px`,
    width: `${w}px`,
    height: `${h}px`,
  };
});

let isMarquee = false;
let panStartX = 0;
let panStartY = 0;
let panStartPanX = 0;
let panStartPanY = 0;
let marqueeStartX = 0;
let marqueeStartY = 0;

function screenToWorld(cx: number, cy: number) {
  return {
    x: (cx - props.viewport.panX) / props.viewport.zoom,
    y: (cy - props.viewport.panY) / props.viewport.zoom,
  };
}

function onWheel(e: WheelEvent) {
  const el = viewportRef.value;
  if (!el) return;
  const rect = el.getBoundingClientRect();
  const cx = e.clientX - rect.left;
  const cy = e.clientY - rect.top;

  if (props.selectedPath && (e.altKey || e.shiftKey)) {
    const state = props.items[props.selectedPath];
    if (state) {
      const delta = e.deltaY > 0 ? -0.05 : 0.05;
      emit('scale-item', {
        path: props.selectedPath,
        scale: Math.max(0.1, Math.min(3, state.scale + delta)),
      });
    }
    return;
  }

  const delta = e.deltaY > 0 ? -0.1 : 0.1;
  const newZoom = Math.max(0.1, Math.min(5, props.viewport.zoom + delta));
  const worldX = (cx - props.viewport.panX) / props.viewport.zoom;
  const worldY = (cy - props.viewport.panY) / props.viewport.zoom;
  emit('update:viewport', {
    zoom: newZoom,
    panX: cx - worldX * newZoom,
    panY: cy - worldY * newZoom,
  });
}

function onViewportMouseDown(e: MouseEvent) {
  const el = viewportRef.value;
  if (!el) return;
  const rect = el.getBoundingClientRect();
  const cx = e.clientX - rect.left;
  const cy = e.clientY - rect.top;

  const panGesture =
    e.button === 1 ||
    (e.button === 0 && (e.ctrlKey || e.metaKey)) ||
    (e.button === 0 && spaceHeld.value);

  if (panGesture) {
    isPanning.value = true;
    panStartX = e.clientX;
    panStartY = e.clientY;
    panStartPanX = props.viewport.panX;
    panStartPanY = props.viewport.panY;

    const onMove = (ev: MouseEvent) => {
      if (!isPanning.value) return;
      emit('update:viewport', {
        ...props.viewport,
        panX: panStartPanX + (ev.clientX - panStartX),
        panY: panStartPanY + (ev.clientY - panStartY),
      });
    };
    const onUp = () => {
      isPanning.value = false;
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return;
  }

  if (e.button === 0) {
    isMarquee = true;
    marqueeStartX = cx;
    marqueeStartY = cy;
    const w0 = screenToWorld(cx, cy);
    marquee.active = false;
    marquee.x1 = w0.x;
    marquee.y1 = w0.y;
    marquee.x2 = w0.x;
    marquee.y2 = w0.y;

    const onMove = (ev: MouseEvent) => {
      if (!isMarquee || !el) return;
      const r = el.getBoundingClientRect();
      const mx = ev.clientX - r.left;
      const my = ev.clientY - r.top;
      if (
        !marquee.active &&
        (Math.abs(mx - marqueeStartX) > 4 || Math.abs(my - marqueeStartY) > 4)
      ) {
        marquee.active = true;
      }
      const w = screenToWorld(mx, my);
      marquee.x2 = w.x;
      marquee.y2 = w.y;
    };

    const onUp = () => {
      if (marquee.active) {
        const x = Math.min(marquee.x1, marquee.x2);
        const y = Math.min(marquee.y1, marquee.y2);
        const w = Math.abs(marquee.x2 - marquee.x1);
        const h = Math.abs(marquee.y2 - marquee.y1);
        if (w > 8 && h > 8) {
          emit('marquee-select', { x, y, w, h });
        } else {
          emit('clear-selection');
        }
      } else {
        emit('clear-selection');
      }
      isMarquee = false;
      marquee.active = false;
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };

    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  }
}

function onItemDragMove(path: string, { dx, dy }: { dx: number; dy: number }) {
  emit('item-drag-move', { path, dx: dx / props.viewport.zoom, dy: dy / props.viewport.zoom });
}

function onItemDragEnd(path: string, { x, y }: { x: number; y: number }) {
  emit('item-drag-end', { path, x, y });
}

function isTypingTarget(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el) return false;
  const tag = el.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable;
}

function onWindowKeyDown(e: KeyboardEvent) {
  if (e.code !== 'Space' || e.repeat || isTypingTarget(e.target)) return;
  if (!pointerInside.value) return;
  e.preventDefault();
  spaceHeld.value = true;
}

function onWindowKeyUp(e: KeyboardEvent) {
  if (e.code === 'Space') {
    spaceHeld.value = false;
    isPanning.value = false;
  }
}

function onWindowBlur() {
  spaceHeld.value = false;
  isPanning.value = false;
}

onMounted(() => {
  window.addEventListener('keydown', onWindowKeyDown);
  window.addEventListener('keyup', onWindowKeyUp);
  window.addEventListener('blur', onWindowBlur);
});

onBeforeUnmount(() => {
  window.removeEventListener('keydown', onWindowKeyDown);
  window.removeEventListener('keyup', onWindowKeyUp);
  window.removeEventListener('blur', onWindowBlur);
});

defineExpose({ viewportRef });
</script>

<style scoped>
.canvas-viewport {
  position: absolute;
  inset: 0;
  overflow: hidden;
  background: var(--dq-fill-dim);
  cursor: default;
}

.canvas-viewport--space-pan {
  cursor: grab;
}

.canvas-viewport--panning {
  cursor: grabbing;
}

.canvas-viewport__grid {
  position: absolute;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  background-color: var(--dq-fill-dim);
  background-image: radial-gradient(circle, rgba(255, 255, 255, 0.16) 1px, transparent 1px);
  background-repeat: repeat;
}

.canvas-viewport__world {
  position: absolute;
  top: 0;
  left: 0;
  z-index: 1;
  will-change: transform;
}

.canvas-viewport__marquee {
  position: absolute;
  border: 1px solid var(--dq-accent);
  background: color-mix(in srgb, var(--dq-accent) 12%, transparent);
  pointer-events: none;
  z-index: 20;
}

.canvas-viewport__empty {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  pointer-events: none;
  z-index: 5;
  padding-bottom: min(180px, 32vh);
}

.canvas-viewport__empty-card {
  pointer-events: auto;
  max-width: 420px;
  padding: 20px 22px;
  border-radius: 14px;
  text-align: center;
}

.canvas-viewport__empty-title {
  margin: 0 0 8px;
  font-size: var(--dq-font-size-title);
  font-weight: 600;
  color: var(--dq-label-primary);
}

.canvas-viewport__empty-hint {
  margin: 0 0 12px;
  font-size: var(--dq-font-size-body);
  color: var(--dq-label-secondary);
  line-height: 1.5;
}

.canvas-viewport__empty-steps {
  margin: 0 0 12px;
  padding-left: 1.2rem;
  text-align: left;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
  line-height: 1.6;
}

.canvas-viewport__empty-shortcuts {
  margin: 0 0 16px;
  padding: 10px 12px;
  list-style: none;
  text-align: left;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
  line-height: 1.55;
  border-radius: 8px;
  background: color-mix(in srgb, var(--dq-fill-secondary) 65%, transparent);
}

.canvas-viewport__empty-shortcuts li + li {
  margin-top: 4px;
}
</style>
