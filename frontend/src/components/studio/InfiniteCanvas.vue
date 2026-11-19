<template>
  <div class="infinite-canvas">
    <CanvasViewport
      ref="viewportComp"
      :items="store.items"
      :viewport="store.viewport"
      :staging="store.staging"
      :overlays="store.overlays"
      :edges="sessionEdges"
      :show-edges="showEdges"
      :show-region-guides="showRegionGuides"
      :gallery-items="items"
      :selected-path="primaryPath"
      :selected-paths="selectedPaths"
      :media="media"
      :describing="describing"
      :editing-path="editingPath"
      :mask-preview="maskPreview"
      :extend-preview="extendPreview"
      @update:viewport="onViewportChange"
      @select-item="onSelectItem"
      @clear-selection="onClearSelection"
      @marquee-select="onMarqueeSelect"
      @item-drag-move="onItemDragMove"
      @item-drag-end="onItemDragEnd"
      @toolbar-action="onToolbarAction"
      @open-import-picker="showGalleryPicker = true"
      @scale-item="onScaleItem"
      @staging-move="onStagingMove"
      @staging-resize="onStagingResize"
      @snap-staging="onSnapStagingToPrimary"
      @overlay-move="onOverlayMove"
      @open-preview="onOpenPreview"
    />

    <CanvasMultiToolbar
      :visible="selectedPaths.length >= 2"
      :count="selectedPaths.length"
      :center-x="multiToolbarPos.x"
      :top-y="multiToolbarPos.y"
      @align="onAlign"
      @distribute="onDistribute"
      @snap-staging="onSnapStagingToSelection"
      @remove="onRemoveSelected"
      @train-lora="onTrainLoraSelected"
    />

    <CanvasToolbar
      :zoom="store.viewport.zoom"
      :graph-open="showGraph"
      :edges-open="showEdges"
      :guides-open="showRegionGuides"
      @update:zoom="(z: number, cx?: number, cy?: number) => store.setZoom(z, cx, cy)"
      @fit-all="fitAll"
      @toggle-layers="showLayers = true"
      @toggle-graph="showGraph = !showGraph"
      @toggle-edges="showEdges = !showEdges"
      @toggle-guides="toggleRegionGuides"
      @export-json="onExportJson"
      @export-png="onExportPng"
      @import-json="onImportJsonClick"
      @copy-session="onCopySessionShare"
      @open-gallery-picker="showGalleryPicker = true"
      @open-composer="$emit('open-composer')"
    />

    <CanvasGalleryPicker
      v-model:open="showGalleryPicker"
      :items="items"
      :media="media"
      :on-canvas-paths="canvasItemPaths"
      @import="onGalleryPickerImport"
    />

    <input
      ref="importFileInput"
      type="file"
      accept="application/json,.json"
      class="infinite-canvas__import-input"
      @change="onImportJsonFile"
    />

    <CanvasSessionGraph
      ref="graphPanelRef"
      :open="showGraph"
      :items="store.items"
      :edges="sessionEdges"
      :gallery-items="items"
      :active-path="primaryPath"
      @close="showGraph = false"
      @focus-node="onFocusNode"
      @import-works="showGalleryPicker = true"
    />

    <CanvasLineageSidebar
      :open="showLineage"
      :asset-id="lineageAssetId"
      :on-canvas-ids="canvasAssetIds"
      @close="showLineage = false"
      @focus-asset="onLineageFocusAsset"
    />

    <CanvasExportPngDialog
      v-model:open="showExportPngDialog"
      v-model="pngExportOptions"
      @confirm="onExportPngConfirm"
    />

    <CanvasLayerPanel
      ref="layerPanelRef"
      v-model:model-value="showLayers"
      :items="store.items"
      :gallery-items="items"
      :selected-path="primaryPath"
      :show-overlays="media === 'image' || media === 'video' || media === 'audio'"
      :media="media"
      :overlays="store.overlays"
      :staging="store.staging"
      @select="(path) => onSelectItem({ path, shiftKey: false })"
      @remove="onRemoveItem"
      @rename="onRenameNode"
      @toggle-visibility="onToggleLayerVisibility"
      @overlay-update="onOverlayUpdate"
      @overlay-clear="onOverlayClear"
      @staging-update="(patch) => store.updateStaging(patch)"
      @focus-staging="focusStaging"
      @import-works="showGalleryPicker = true"
      @snap-staging="onSnapStagingToPrimary"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onBeforeUnmount, nextTick, unref } from 'vue';
import { useRouter } from 'vue-router';
import { toast } from '@/utils/feedback';
import { $tt } from '@/utils/i18n';
import CanvasViewport from '@/components/studio/CanvasViewport.vue';
import CanvasToolbar from '@/components/studio/CanvasToolbar.vue';
import CanvasLayerPanel from '@/components/studio/CanvasLayerPanel.vue';
import CanvasLineageSidebar from '@/components/studio/CanvasLineageSidebar.vue';
import CanvasSessionGraph from '@/components/studio/CanvasSessionGraph.vue';
import CanvasMultiToolbar from '@/components/studio/CanvasMultiToolbar.vue';
import CanvasExportPngDialog from '@/components/studio/CanvasExportPngDialog.vue';
import CanvasGalleryPicker from '@/components/studio/CanvasGalleryPicker.vue';
import { useCanvasStore } from '@/composables/useCanvasStore';
import {
  assetIdFromPath,
  describeCanvasNode,
  downloadBlob,
  exportCanvasJson,
  exportCanvasPng,
  loadCanvasPngExportOptions,
  saveCanvasPngExportOptions,
  type CanvasPngExportOptions,
} from '@/utils/canvasExport';
import { parseCanvasImportJson } from '@/utils/canvasImport';
import {
  assetIdFromGalleryPath,
  navigateToCopilot,
  thumbnailUrlForAsset,
} from '@/utils/copilotHandoff';
import {
  assetIdsFromGalleryPaths,
  navigateToLoraTrainWithAssets,
} from '@/utils/loraTrainHandoff';
import { DQ_STORAGE, getItem, setItem } from '@/utils/storage';

const router = useRouter();
import { api } from '@/utils/api';
import { previewUrlForGalleryItem, isImageGalleryItem } from '@/utils/canvasAssets';
import { rectIntersects, itemBounds, type AlignMode, type DistributeMode } from '@/utils/canvasGeometry';
import type {
  GalleryItem,
  CanvasViewportState,
  CanvasSessionState,
  CanvasOverlayKind,
} from '@/types';

const props = defineProps<{
  items: GalleryItem[];
  media: import('@/composables/useCanvasStore').CanvasMedia;
  editingPath?: string;
  maskPreview?: import('@/types').CanvasMaskPreviewState | null;
  extendPreview?: import('@/types').CanvasExtendPreviewState | null;
}>();

const emit = defineEmits<{
  (e: 'use-as-reference', payload: { path: string; previewUrl: string; quiet?: boolean }): void;
  (e: 'use-as-control', payload: { path: string; previewUrl: string; quiet?: boolean }): void;
  (e: 'use-as-start-frame', payload: { path: string; previewUrl: string; quiet?: boolean }): void;
  (e: 'use-as-tail-frame', payload: { path: string; previewUrl: string; quiet?: boolean }): void;
  (e: 'use-as-video-source', payload: { path: string; previewUrl: string; quiet?: boolean }): void;
  (e: 'use-as-animate-source', payload: { path: string; previewUrl: string; quiet?: boolean }): void;
  (e: 'use-as-cover-source', payload: { path: string; previewUrl: string; quiet?: boolean }): void;
  (e: 'card-action', payload: { action: string; item: GalleryItem }): void;
  (e: 'download', item: GalleryItem): void;
  (e: 'delete', item: GalleryItem): void;
  (e: 'toggle-grid-view'): void;
  (e: 'node-selected', item: GalleryItem | null): void;
  (e: 'session-ready', payload: { sessionId: string }): void;
  (e: 'open-preview', item: GalleryItem): void;
  (e: 'composer-restore', snapshot: import('@/types').CanvasComposerSnapshot): void;
  (e: 'overlay-cleared', kind: CanvasOverlayKind): void;
  (e: 'open-composer'): void;
  (e: 'request-close-composer'): void;
}>();

const store = useCanvasStore(props.media);
const sessionEdges = computed(() => unref(store.edges));

const showLayers = ref(false);
const showLineage = ref(false);
const showGraph = ref(false);
const showEdges = ref(true);
const showRegionGuides = ref(getItem(DQ_STORAGE.CANVAS_REGION_GUIDES) === '1');
const selectedPaths = ref<string[]>([]);
const describing = ref(false);
const viewportComp = ref<InstanceType<typeof CanvasViewport> | null>(null);
const layerPanelRef = ref<InstanceType<typeof CanvasLayerPanel> | null>(null);
const graphPanelRef = ref<InstanceType<typeof CanvasSessionGraph> | null>(null);
const importFileInput = ref<HTMLInputElement | null>(null);
const showExportPngDialog = ref(false);
const showGalleryPicker = ref(false);
const pngExportOptions = ref<CanvasPngExportOptions>(loadCanvasPngExportOptions());

const canvasItemPaths = computed(() => Object.keys(unref(store.items) || {}));

const primaryPath = computed(() =>
  selectedPaths.value.length === 1 ? selectedPaths.value[0] : selectedPaths.value[0] || ''
);

const lineageAssetId = computed(() => {
  const path = primaryPath.value;
  if (!path.startsWith('asset:')) return '';
  return path.slice('asset:'.length);
});

const canvasAssetIds = computed(() =>
  Object.keys(store.items)
    .filter((p) => p.startsWith('asset:'))
    .map((p) => p.slice('asset:'.length))
);

function onLineageFocusAsset(assetId: string) {
  focusLineageAsset(assetId);
}

function focusLineageAsset(assetId: string): boolean {
  const id = String(assetId || '').trim();
  if (!id) return false;
  const path = `asset:${id}`;
  if (store.hasItem(path)) {
    onFocusNode(path);
    toast.success($tt('canvas.lineageJumpFocused'));
    return true;
  }
  if (galleryItem(path)) {
    addPathsToCanvas([path], { placement: 'center', focusLast: true });
    toast.success($tt('canvas.lineageJumpAdded'));
    return true;
  }
  toast.warning($tt('canvas.lineageJumpMissing'));
  return false;
}

const multiToolbarPos = computed(() => {
  const paths = selectedPaths.value;
  if (paths.length < 2) return { x: 0, y: 0 };
  const vp = store.viewport;
  let minX = Infinity;
  let minY = Infinity;
  for (const path of paths) {
    const b = itemBounds(path, store.items[path], galleryItem(path));
    if (!b) continue;
    minX = Math.min(minX, b.x);
    minY = Math.min(minY, b.y);
  }
  if (!isFinite(minX)) return { x: 400, y: 200 };
  const el = getViewportEl();
  const vw = el?.clientWidth || 800;
  const cx = vp.panX + (minX + 120) * vp.zoom;
  const top = vp.panY + minY * vp.zoom;
  return { x: Math.min(vw - 80, Math.max(80, cx)), y: Math.max(60, top) };
});

function galleryItem(path: string): GalleryItem | undefined {
  return props.items.find((i) => i.path === path);
}

function emitPrimarySelection() {
  const path = primaryPath.value;
  if (!path) {
    emit('node-selected', null);
    return;
  }
  emit('node-selected', galleryItem(path) ?? null);
}

function onViewportChange(vp: CanvasViewportState) {
  store.viewport.zoom = vp.zoom;
  store.viewport.panX = vp.panX;
  store.viewport.panY = vp.panY;
}

function onSelectItem({ path, shiftKey }: { path: string; shiftKey: boolean }) {
  if (shiftKey) {
    const set = new Set(selectedPaths.value);
    if (set.has(path)) set.delete(path);
    else set.add(path);
    selectedPaths.value = [...set];
  } else {
    selectedPaths.value = [path];
  }
  store.bringToFront(path);
  store.setActiveAssetPath(path);
  store.placeStagingBeside(path, props.items);
  emitPrimarySelection();
}

function onClearSelection() {
  selectedPaths.value = [];
  store.setActiveAssetPath('');
  emit('node-selected', null);
}

function onMarqueeSelect(rect: { x: number; y: number; w: number; h: number }) {
  const hits: string[] = [];
  for (const [path, state] of Object.entries(store.items)) {
    if (!state.visible) continue;
    const gi = galleryItem(path);
    if (!gi) continue;
    const bw = (gi.width || 512) * state.scale;
    const bh = (gi.height || 512) * state.scale;
    if (rectIntersects(rect.x, rect.y, rect.w, rect.h, state.x, state.y, bw, bh)) {
      hits.push(path);
    }
  }
  selectedPaths.value = hits;
  if (pathsHaveLineageOnCanvas(hits) && !showEdges.value) {
    showEdges.value = true;
  }
  if (hits.length === 1) {
    store.setActiveAssetPath(hits[0]);
    store.placeStagingBeside(hits[0], props.items);
    emitPrimarySelection();
  } else {
    store.setActiveAssetPath('');
    emit('node-selected', null);
  }
}

function onItemDragMove({ path, dx, dy }: { path: string; dx: number; dy: number }) {
  const it = store.items[path];
  if (!it) return;
  store.updatePosition(path, it.x + dx, it.y + dy);
}

function onItemDragEnd() {
  /* persisted during drag-move */
}

function onScaleItem({ path, scale }: { path: string; scale: number }) {
  store.updateScale(path, scale);
}

function onStagingMove({ x, y }: { x: number; y: number }) {
  store.updateStaging({ x, y });
}

function onStagingResize({ width, height }: { width: number; height: number }) {
  store.updateStaging({ width, height });
}

function onOverlayMove(payload: { kind: CanvasOverlayKind; x: number; y: number }) {
  store.updateOverlay(payload.kind, { x: payload.x, y: payload.y });
}

function onRenameNode(path: string, label: string) {
  store.updateLabel(path, label);
  toast.success($tt('canvas.renameNodeDone'));
}

async function onToolbarRename() {
  const path = primaryPath.value;
  if (!path) return;
  showLayers.value = true;
  await nextTick();
  await layerPanelRef.value?.startRenameForPath(path);
}

function onRemoveItem(path: string) {
  store.removeItem(path);
  selectedPaths.value = selectedPaths.value.filter((p) => p !== path);
  if (selectedPaths.value.length === 0) onClearSelection();
  else emitPrimarySelection();
  store.refreshEdges(props.items);
}

function onRemoveSelected() {
  const paths = [...selectedPaths.value];
  store.removeItems(paths);
  selectedPaths.value = [];
  onClearSelection();
  store.refreshEdges(props.items);
}

function onSnapStagingToPrimary() {
  const path = primaryPath.value;
  if (!path) {
    toast.warning($tt('canvas.snapStagingNeedSelection'));
    return;
  }
  store.placeStagingBeside(path, props.items);
  nextTick(() => focusStaging());
  toast.success($tt('canvas.snapStagingDone'));
}

function onSnapStagingToSelection() {
  if (selectedPaths.value.length === 0) {
    toast.warning($tt('canvas.snapStagingNeedSelection'));
    return;
  }
  store.placeStagingBesideSelection(selectedPaths.value, props.items);
  nextTick(() => focusStaging());
  toast.success($tt('canvas.snapStagingDone'));
}

function onAlign(mode: AlignMode) {
  store.alignItems(selectedPaths.value, mode, props.items);
}

function onDistribute(mode: DistributeMode) {
  store.distributeItems(selectedPaths.value, mode, props.items);
}

function onToggleLayerVisibility(path: string, visible?: boolean) {
  if (typeof visible === 'boolean') {
    store.setVisibility(path, visible);
  } else {
    store.toggleVisibility(path);
  }
}

function toggleRegionGuides() {
  showRegionGuides.value = !showRegionGuides.value;
  setItem(DQ_STORAGE.CANVAS_REGION_GUIDES, showRegionGuides.value ? '1' : '0');
}

function onGalleryPickerImport(paths: string[]) {
  if (paths.length === 0) return;
  const added = addPathsToCanvas(paths, { placement: 'center', focusLast: false, fit: true });
  if (added.length === 0) {
    toast.info($tt('canvas.importNothingNew'));
    return;
  }
  toast.success($tt('canvas.batchAdded', { n: added.length }));
}

function onOpenPreview(path: string) {
  const item = galleryItem(path);
  if (item) emit('open-preview', item);
}

function onFocusNode(path: string) {
  onSelectItem({ path, shiftKey: false });
  if (pathsHaveLineageOnCanvas([path]) && !showEdges.value) {
    showEdges.value = true;
  }
  const state = store.items[path];
  if (!state) return;
  const gi = galleryItem(path);
  const w = (gi?.width || 512) * state.scale;
  const h = (gi?.height || 512) * state.scale;
  const el = getViewportEl();
  if (!el) return;
  const vw = el.clientWidth;
  const vh = el.clientHeight;
  const zoom = store.viewport.zoom;
  store.viewport.panX = vw / 2 - (state.x + w / 2) * zoom;
  store.viewport.panY = vh / 2 - (state.y + h / 2) * zoom;
}

function bindPayload(path: string, item: GalleryItem) {
  return { path, previewUrl: previewUrlForGalleryItem(item) };
}

function onTrainLoraSelected() {
  const paths = selectedPaths.value.filter((path) => {
    const item = galleryItem(path);
    return item ? isImageGalleryItem(item) : path.startsWith('asset:');
  });
  const assetIds = assetIdsFromGalleryPaths(paths);
  if (!assetIds.length) {
    toast.warning($tt('loraTrain.needImageAssets'));
    return;
  }
  const ok = navigateToLoraTrainWithAssets(router, assetIds, {
    datasetName: $tt('loraTrain.canvasImportDataset'),
  });
  if (ok) toast.success($tt('loraTrain.handoffStarted'));
}

function onToolbarAction(action: string) {
  const path = primaryPath.value;
  const item = galleryItem(path);
  if (!item) return;

  switch (action) {
    case 'quick-reference':
      store.placeStagingBeside(path, props.items);
      {
        const payload = bindPayload(path, item);
        emit('use-as-reference', { ...payload, quiet: true });
        toast.success($tt('canvas.rewriteBranchHint'));
        emit('open-composer');
      }
      break;
    case 'quick-control':
      store.placeStagingBeside(path, props.items);
      {
        const payload = bindPayload(path, item);
        emit('use-as-control', { ...payload, quiet: true });
        store.setControlOverlay(path, item);
        toast.success($tt('canvas.controlBranchHint'));
      }
      break;
    case 'use-reference': {
      const payload = bindPayload(path, item);
      emit('use-as-reference', payload);
      break;
    }
    case 'use-control': {
      const payload = bindPayload(path, item);
      emit('use-as-control', payload);
      store.setControlOverlay(path, item);
      break;
    }
    case 'quick-animate':
      store.placeStagingBeside(path, props.items);
      {
        const payload = bindPayload(path, item);
        emit('use-as-start-frame', { ...payload, quiet: true });
        toast.success($tt('canvas.animateBranchHint'));
        emit('open-composer');
      }
      break;
    case 'use-start-frame': {
      const payload = bindPayload(path, item);
      emit('use-as-start-frame', payload);
      break;
    }
    case 'use-tail-frame': {
      const payload = bindPayload(path, item);
      emit('use-as-tail-frame', payload);
      break;
    }
    case 'quick-animate-video':
      store.placeStagingBeside(path, props.items);
      {
        const payload = bindPayload(path, item);
        emit('use-as-animate-source', { ...payload, quiet: true });
        toast.success($tt('canvas.animateVideoBranchHint'));
        emit('open-composer');
      }
      break;
    case 'use-animate-source': {
      const payload = bindPayload(path, item);
      emit('use-as-animate-source', payload);
      break;
    }
    case 'use-video-source': {
      const payload = bindPayload(path, item);
      emit('use-as-video-source', payload);
      break;
    }
    case 'quick-cover':
      store.placeStagingBeside(path, props.items);
      {
        const payload = bindPayload(path, item);
        emit('use-as-cover-source', { ...payload, quiet: true });
        store.setOverlay('cover_source', path, item);
        toast.success($tt('canvas.coverBranchHint'));
      }
      break;
    case 'use-cover-source': {
      const payload = bindPayload(path, item);
      emit('use-as-cover-source', payload);
      store.setOverlay('cover_source', path, item);
      break;
    }
    case 'snap-staging':
      store.placeStagingBeside(path, props.items);
      nextTick(() => focusStaging());
      toast.success($tt('canvas.snapStagingDone'));
      break;
    case 'branch':
      store.placeStagingBeside(path, props.items);
      toast.info($tt('canvas.branchHint'));
      emit('open-composer');
      break;
    case 'remove':
      onRemoveItem(path);
      break;
    case 'download':
      emit('download', item);
      break;
    case 'save-note': {
      const text = String(item.prompt || item.title || '').trim();
      if (!text) {
        toast.warning($tt('canvas.noPromptForNote'));
        return;
      }
      store.updateNote(path, text);
      toast.success($tt('canvas.noteSaved'));
      break;
    }
    case 'copilot-image-to-prompt': {
      const id = assetIdFromGalleryPath(path);
      if (!id) {
        toast.warning($tt('canvas.copilotNeedAsset'));
        return;
      }
      navigateToCopilot(router, {
        media: props.media === 'video' ? 'video' : 'image',
        task: 'image_to_prompt',
        assetId: id,
        assetPreview: thumbnailUrlForAsset(id),
        prompt: String(item.prompt || '').trim() || undefined,
      });
      break;
    }
    case 'copilot-analyze': {
      const id = assetIdFromGalleryPath(path);
      if (!id) {
        toast.warning($tt('canvas.copilotNeedAsset'));
        return;
      }
      navigateToCopilot(router, {
        media: 'image',
        task: 'analyze_reference',
        assetId: id,
        assetPreview: thumbnailUrlForAsset(id),
        prompt: $tt('assistant.presetStyleText'),
      });
      break;
    }
    case 'ai-describe':
      void onAiDescribe(path, { preferVision: true });
      break;
    case 'ai-describe-text':
      void onAiDescribe(path, { preferVision: false });
      break;
    case 'rename':
      void onToolbarRename();
      break;
    case 'lineage':
      showLineage.value = true;
      break;
    case 'export-png':
      void onExportNodePng(path, item);
      break;
    case 'train-lora': {
      if (!isImageGalleryItem(item)) {
        toast.warning($tt('loraTrain.needImageAssets'));
        return;
      }
      const id = assetIdFromGalleryPath(path);
      if (!id) {
        toast.warning($tt('canvas.copilotNeedAsset'));
        return;
      }
      const ok = navigateToLoraTrainWithAssets(router, [id], {
        datasetName: $tt('loraTrain.canvasImportDataset'),
      });
      if (ok) toast.success($tt('loraTrain.handoffStarted'));
      break;
    }
    case 'retouch':
    case 'extend':
    case 'upscale':
      store.placeStagingBeside(path, props.items);
      emit('card-action', { action, item });
      if (action === 'retouch') toast.success($tt('canvas.retouchBranchHint'));
      else if (action === 'extend') toast.success($tt('canvas.extendBranchHint'));
      else if (props.media === 'video') toast.success($tt('canvas.upscaleBranchHint'));
      else toast.success($tt('canvas.imageUpscaleBranchHint'));
      break;
    default:
      break;
  }
}

function onOverlayUpdate(
  kind: CanvasOverlayKind,
  patch: Partial<import('@/types').CanvasOverlayLayer>
) {
  store.updateOverlay(kind, patch);
}

function onOverlayClear(kind: CanvasOverlayKind) {
  store.clearOverlay(kind);
  emit('overlay-cleared', kind);
}

async function onAiDescribe(path: string, opts?: { preferVision?: boolean }) {
  const aid = assetIdFromPath(path);
  if (!aid) return;
  describing.value = true;
  try {
    const { note, visionUsed } = await describeCanvasNode(aid, opts);
    store.updateNote(path, note);
    toast.success(
      visionUsed ? $tt('canvas.aiDescribeVisionDone') : $tt('canvas.aiDescribeDone')
    );
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    toast.error($tt('canvas.aiDescribeFailed', { msg }));
  } finally {
    describing.value = false;
  }
}

async function onCopySessionShare() {
  const sid = unref(store.sessionId);
  if (!sid) return;
  const payload = {
    media: props.media,
    sessionId: sid,
    title: unref(store.sessionTitle) || 'Canvas',
    state: {
      items: { ...store.items },
      viewport: { ...store.viewport },
      staging: { ...store.staging },
      active_asset_path: unref(store.activeAssetPath),
      overlays: { ...store.overlays },
      edges: [...unref(store.edges)],
      composer_snapshot: Object.keys(store.composerSnapshot).length
        ? { ...store.composerSnapshot }
        : null,
    },
  };
  try {
    await navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
    toast.success($tt('canvas.sessionCopied'));
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    toast.error($tt('canvas.sessionCopyFailed', { msg }));
  }
}

function onExportJson() {
  const sid = unref(store.sessionId);
  if (!sid) return;
  exportCanvasJson({
    sessionId: sid,
    title: unref(store.sessionTitle) || 'Canvas',
    media: props.media,
    state: {
      items: { ...store.items },
      viewport: { ...store.viewport },
      staging: { ...store.staging },
      active_asset_path: unref(store.activeAssetPath),
      overlays: { ...store.overlays },
      edges: [...unref(store.edges)],
      composer_snapshot: Object.keys(store.composerSnapshot).length
        ? { ...store.composerSnapshot }
        : null,
    },
  });
  toast.success($tt('canvas.exportJsonDone'));
}

function onImportJsonClick() {
  importFileInput.value?.click();
}

async function applyImportedCanvasState(state: CanvasSessionState) {
  const sid = unref(store.sessionId);
  if (!sid) return;
  await api.canvas.updateSession(sid, { state });
  const detail = await api.canvas.getSession(sid);
  store.applyState(detail.state);
  selectedPaths.value = store.activeAssetPath.value ? [store.activeAssetPath.value] : [];
  emitPrimarySelection();
  emit('composer-restore', { ...store.composerSnapshot });
  store.refreshEdges(props.items);
  toast.success($tt('canvas.importJsonDone'));
}

async function importCanvasJsonText(text: string) {
  const parsed = parseCanvasImportJson(text);
  if (parsed.media && parsed.media !== props.media) {
    toast.warning($tt('canvas.importMediaMismatch'));
    return;
  }
  await applyImportedCanvasState(parsed.state as CanvasSessionState);
}

async function onImportJsonFile(ev: Event) {
  const input = ev.target as HTMLInputElement;
  const file = input.files?.[0];
  input.value = '';
  if (!file) return;
  try {
    await importCanvasJsonText(await file.text());
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    toast.error($tt('canvas.importJsonFailed', { msg }));
  }
}

function onPasteImport(ev: ClipboardEvent) {
  if (!unref(store.ready)) return;
  const target = ev.target as HTMLElement | null;
  if (
    target &&
    (target.tagName === 'INPUT' ||
      target.tagName === 'TEXTAREA' ||
      target.isContentEditable)
  ) {
    return;
  }
  const text = ev.clipboardData?.getData('text/plain')?.trim();
  if (!text || !text.startsWith('{')) return;
  void (async () => {
    try {
      await importCanvasJsonText(text);
    } catch {
      /* not canvas JSON — ignore */
    }
  })();
}

function onExportPng() {
  pngExportOptions.value = loadCanvasPngExportOptions();
  showExportPngDialog.value = true;
}

async function onExportNodePng(path: string, item: GalleryItem) {
  const state = store.items[path];
  if (!state) return;
  try {
    const blob = await exportCanvasPng(
      { [path]: state },
      props.items,
      { ...store.staging, visible: false },
      {
        includeStaging: false,
        includeOverlays: false,
        includeEdges: false,
        includeNotes: true,
      }
    );
    const base = (item.name || path.slice('asset:'.length) || 'node').replace(
      /[^\w\u4e00-\u9fff-]+/g,
      '_'
    );
    downloadBlob(blob, `${base}.png`);
    toast.success($tt('canvas.exportNodePngDone'));
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    toast.error($tt('canvas.exportPngFailed', { msg }));
  }
}

async function onExportPngConfirm(opts: CanvasPngExportOptions) {
  saveCanvasPngExportOptions(opts);
  pngExportOptions.value = { ...opts };
  try {
    const blob = await exportCanvasPng(store.items, props.items, store.staging, {
      ...opts,
      overlays: store.overlays,
      edges: [...unref(store.edges)],
      editingPath: props.editingPath,
      maskPreview: props.maskPreview ?? null,
      extendPreview: props.extendPreview ?? null,
    });
    const title = (unref(store.sessionTitle) || 'canvas').replace(/[^\w\u4e00-\u9fff-]+/g, '_');
    downloadBlob(blob, `${title}.png`);
    toast.success($tt('canvas.exportPngDone'));
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    toast.error($tt('canvas.exportPngFailed', { msg }));
  }
}

function getViewportEl(): HTMLElement | null {
  const comp = viewportComp.value as { viewportRef?: HTMLElement | null } | null;
  const raw = comp?.viewportRef;
  if (!raw) return null;
  if (typeof raw === 'object' && raw !== null && 'value' in raw) {
    return (raw as { value: HTMLElement | null }).value;
  }
  return raw as HTMLElement;
}

function fitAll() {
  store.fitAll(props.items, getViewportEl());
}

function focusStaging() {
  const s = store.staging;
  const el = getViewportEl();
  if (!el) return;
  const vw = el.clientWidth;
  const vh = el.clientHeight;
  const zoom = store.viewport.zoom;
  store.viewport.panX = vw / 2 - (s.x + s.width / 2) * zoom;
  store.viewport.panY = vh / 2 - (s.y + s.height / 2) * zoom;
}

function addToCanvas(item: GalleryItem, position?: { x: number; y: number }): boolean {
  const added = store.addItem(item, position);
  if (added) {
    nextTick(() => {
      onSelectItem({ path: item.path, shiftKey: false });
      store.refreshEdges(props.items);
    });
  }
  return added;
}

function pathsHaveLineageOnCanvas(paths: string[]): boolean {
  return paths.some((path) => {
    const gi = galleryItem(path);
    const parentId = String(gi?.metadata?.parent_asset_id || '').trim();
    if (!parentId) return false;
    return store.hasItem(`asset:${parentId}`);
  });
}

function addPathsToCanvas(
  paths: string[],
  opts?: {
    fit?: boolean;
    placement?: 'staging' | 'center';
    focusLast?: boolean;
    /** Prefer caller-fresh gallery rows; props.items can lag one tick after loadGallery. */
    galleryItems?: GalleryItem[];
  }
): string[] {
  const placement = opts?.placement ?? (store.staging.visible ? 'staging' : 'center');
  const catalog = opts?.galleryItems?.length ? opts.galleryItems : props.items;
  const added = store.addPathsFromGallery(paths, catalog, {
    placement,
    selectLast: true,
  });
  if (added.length > 0) {
    const last = added[added.length - 1];
    nextTick(() => {
      onSelectItem({ path: last, shiftKey: false });
      const lineageItems = opts?.galleryItems?.length ? opts.galleryItems : props.items;
      const hasLineage = added.some((path) => {
        const gi = lineageItems.find((i) => i.path === path);
        const parentId = String(gi?.metadata?.parent_asset_id || '').trim();
        return !!parentId && store.hasItem(`asset:${parentId}`);
      });
      if (hasLineage) {
        if (!showEdges.value) {
          showEdges.value = true;
          toast.info($tt('canvas.lineageEdgesShown'));
        }
        store.refreshEdges(lineageItems);
      }
      if (placement === 'staging' && added.length > 0) {
        toast.success($tt('canvas.resultInStaging', { n: added.length }));
      }
      if (opts?.focusLast !== false) {
        onFocusNode(last);
      } else if (opts?.fit) {
        fitAll();
      }
    });
  } else if (opts?.fit) {
    nextTick(() => fitAll());
  }
  return added;
}

function syncControlOverlay(path: string | null) {
  if (!path) {
    store.setControlOverlay(null);
    return;
  }
  store.setControlOverlay(path, galleryItem(path));
}

function pruneOrphans() {
  // Gallery may still be loading; avoid wiping persisted canvas nodes on empty gallery.
  if (props.items.length === 0) return;
  const valid = new Set(props.items.map((i) => i.path));
  store.pruneOrphans(valid);
  store.refreshEdges(props.items);
}

watch(
  () => store.ready,
  (isReady) => {
    if (isReady && store.sessionId.value) {
      emit('session-ready', { sessionId: store.sessionId.value });
      emit('composer-restore', { ...store.composerSnapshot });
    }
  },
  { immediate: true }
);

watch(
  () => unref(store.sessionId),
  () => {
    if (!unref(store.ready)) return;
    selectedPaths.value = store.activeAssetPath.value ? [store.activeAssetPath.value] : [];
    emitPrimarySelection();
  }
);

watch(
  () => props.items.length,
  () => {
    pruneOrphans();
    if (unref(store.ready)) store.refreshEdges(props.items);
  }
);

watch(
  () =>
    props.items
      .map(
        (i) =>
          `${i.path}:${String(i.metadata?.parent_asset_id || '')}:${String(i.metadata?.relation_type || '')}`
      )
      .join('|'),
  () => {
    if (unref(store.ready)) store.refreshEdges(props.items);
  }
);

watch(
  () => store.activeAssetPath.value,
  (path) => {
    if (!unref(store.ready) || !path || !store.hasItem(path)) return;
    if (selectedPaths.value.length === 1 && selectedPaths.value[0] === path) return;
    selectedPaths.value = [path];
    emitPrimarySelection();
  }
);

function isCanvasTypingTarget(target: EventTarget | null): boolean {
  if (!target || !(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true;
  if (target.isContentEditable) return true;
  return !!target.closest('.dq-dialog, [role="dialog"]');
}

function onCanvasKeydown(e: KeyboardEvent) {
  if (isCanvasTypingTarget(e.target)) return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;

  const key = e.key;

  if (key === 'Delete' || key === 'Backspace') {
    if (selectedPaths.value.length >= 2) {
      e.preventDefault();
      onRemoveSelected();
    } else if (primaryPath.value) {
      e.preventDefault();
      onRemoveItem(primaryPath.value);
    }
    return;
  }

  if (key === 'Escape') {
    e.preventDefault();
    if (showGalleryPicker.value) {
      showGalleryPicker.value = false;
      return;
    }
    if (showExportPngDialog.value) {
      showExportPngDialog.value = false;
      return;
    }
    if (showLineage.value) {
      showLineage.value = false;
      return;
    }
    if (showLayers.value) {
      if (layerPanelRef.value?.cancelRenameIfActive?.()) return;
      showLayers.value = false;
      return;
    }
    if (showGraph.value) {
      if (graphPanelRef.value?.clearFilterIfActive?.()) return;
      showGraph.value = false;
      return;
    }
    if (selectedPaths.value.length > 0) {
      onClearSelection();
    }
    return;
  }

  if (key === 'Enter' && primaryPath.value && selectedPaths.value.length === 1) {
    e.preventDefault();
    onOpenPreview(primaryPath.value);
    return;
  }

  if (key === 'F2') {
    e.preventDefault();
    if (primaryPath.value) void onToolbarRename();
    return;
  }

  const lower = key.toLowerCase();
  if (lower === 'f') {
    e.preventDefault();
    fitAll();
    return;
  }
  if (lower === 'l') {
    e.preventDefault();
    showLayers.value = !showLayers.value;
    return;
  }
  if (lower === 'g') {
    e.preventDefault();
    showGraph.value = !showGraph.value;
    return;
  }
  if (lower === 'e') {
    e.preventDefault();
    showEdges.value = !showEdges.value;
    return;
  }
  if (lower === 'i') {
    e.preventDefault();
    showGalleryPicker.value = true;
    return;
  }
  if (lower === 'r') {
    e.preventDefault();
    toggleRegionGuides();
    return;
  }
  if (lower === 's') {
    e.preventDefault();
    if (selectedPaths.value.length >= 2) {
      onSnapStagingToSelection();
    } else if (primaryPath.value) {
      onSnapStagingToPrimary();
    } else {
      toast.warning($tt('canvas.snapStagingNeedSelection'));
    }
    return;
  }
  if (lower === 'c') {
    e.preventDefault();
    emit('open-composer');
    return;
  }
  if (lower === 'y') {
    e.preventDefault();
    if (!primaryPath.value) {
      toast.warning($tt('canvas.lineageSelectHint'));
      return;
    }
    showLineage.value = !showLineage.value;
  }
}

onMounted(() => {
  pruneOrphans();
  window.addEventListener('paste', onPasteImport);
  window.addEventListener('keydown', onCanvasKeydown);
});

onBeforeUnmount(() => {
  window.removeEventListener('paste', onPasteImport);
  window.removeEventListener('keydown', onCanvasKeydown);
});

watch(showLayers, (open) => {
  if (open) emit('request-close-composer');
});

function closeMutexPanels() {
  showLayers.value = false;
  showLineage.value = false;
  showGraph.value = false;
}

defineExpose({
  addToCanvas,
  addPathsToCanvas,
  fitAll,
  focusNode: onFocusNode,
  closeMutexPanels,
  focusLineageAsset,
  hasOnCanvas: store.hasItem,
  sessionId: store.sessionId,
  syncControlOverlay,
  persistComposerSnapshot: store.setComposerSnapshot,
  getSelectedItem: () => (primaryPath.value ? galleryItem(primaryPath.value) ?? null : null),
  getSelectedPaths: () => [...selectedPaths.value],
});
</script>

<style scoped>
.infinite-canvas {
  position: absolute;
  inset: 0;
  overflow: hidden;
}

.infinite-canvas__import-input {
  display: none;
}
</style>
