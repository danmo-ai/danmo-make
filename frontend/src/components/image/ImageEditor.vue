<template>
  <div class="image-editor" ref="editorRoot" tabindex="0" @keydown="onKeyDown">
    <div class="editor-canvas-wrapper" ref="canvasWrapper">
      <canvas
        ref="mainCanvas"
        class="editor-canvas"
        @mousedown="onPointerDown"
        @mousemove="onPointerMove"
        @mouseup="onPointerUp"
        @mouseleave="onPointerUp"
        @touchstart="onTouchStart"
        @touchmove.prevent="onTouchMove"
        @touchend="onTouchEnd"
        @wheel.prevent="onWheel"
      />
      <div v-if="!src" class="editor-empty">
        <DqIcon :size="40"><PictureFilled /></DqIcon>
        <p class="editor-empty__lead">{{ $t('studio.uploadEditImage') }}</p>
        <p class="editor-empty__subhint">{{ $t('studio.editImageEmptyHint') }}</p>
      </div>
    </div>
    <div v-if="src" class="editor-toolbar">
      <div class="toolbar-group">
        <DqButton size="sm" :type="currentTool === 'brush' ? 'primary' : 'default'" :title="$t('studio.brush')" @click="setTool('brush')">
          <DqIcon :size="14"><brush /></DqIcon>
        </DqButton>
        <DqButton size="sm" :type="currentTool === 'lasso' ? 'primary' : 'default'" :title="$t('studio.lasso')" @click="setTool('lasso')">
          <DqIcon :size="14"><Pencil /></DqIcon>
        </DqButton>
        <DqButton size="sm" :type="currentTool === 'eraser' ? 'primary' : 'default'" :title="$t('studio.eraser')" @click="setTool('eraser')">
          <DqIcon :size="14"><Delete /></DqIcon>
        </DqButton>
      </div>
      <div v-if="currentTool === 'brush' || currentTool === 'eraser'" class="toolbar-group">
        <span class="toolbar-label">{{ $t('studio.brushSize') }}</span>
        <DqSlider v-model="brushSize" :min="4" :max="200" :step="2" class="editor-brush-slider" />
        <span class="toolbar-val">{{ brushSize }}px</span>
      </div>
      <div v-if="currentTool === 'lasso' && lassoPoints.length > 0" class="toolbar-group">
        <DqButton size="sm" @click="closeLasso">{{ $t('studio.closeLasso') }}</DqButton>
      </div>
      <div v-if="hasMaskContent" class="toolbar-group">
        <DqButton size="sm" @click="invertMask">{{ $t('studio.invertMask') }}</DqButton>
      </div>
      <div class="toolbar-group">
        <DqButton size="sm" :disabled="undoStack.length === 0" :title="`${$t('studio.undo')} Ctrl+Z`" @click="undo">
          <DqIcon :size="14"><RotateCcw /></DqIcon>
        </DqButton>
        <DqButton size="sm" :disabled="redoStack.length === 0" :title="`${$t('studio.redo')} Ctrl+Shift+Z`" @click="redo">
          <DqIcon :size="14"><refresh-right /></DqIcon>
        </DqButton>
      </div>
      <div class="toolbar-group">
        <DqButton type="text" size="sm" :disabled="!hasMaskContent" @click="clearMask">{{ $t('studio.clearMask') }}</DqButton>
      </div>
      <div class="toolbar-divider" />
      <div class="toolbar-group">
        <DqButton size="sm" circle :disabled="zoom <= 0.25" title="Zoom out" @click="zoomOut">
          <DqIcon :size="14"><zoom-out /></DqIcon>
        </DqButton>
        <span class="toolbar-val toolbar-val--zoom">{{ Math.round(zoom * 100) }}%</span>
        <DqButton size="sm" circle :disabled="zoom >= 4" title="Zoom in" @click="zoomIn">
          <DqIcon :size="14"><zoom-in /></DqIcon>
        </DqButton>
        <DqButton size="sm" circle title="Reset view" @click="resetView">
          <DqIcon :size="14"><refresh /></DqIcon>
        </DqButton>
      </div>
      <div class="toolbar-divider" />
      <div v-if="showSubmitButton" class="toolbar-group">
        <DqButton type="primary" :disabled="!hasMaskContent" @click="$emit('submit')">
          {{ $t('studio.generate') }}
        </DqButton>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
// @ts-nocheck
import { ref, onMounted, watch, nextTick } from 'vue';

const props = withDefaults(
  defineProps<{
    src?: string;
    mode?: string;
    recentGallery?: unknown[];
    showSubmitButton?: boolean;
  }>(),
  {
    src: '',
    mode: 'inpainting',
    recentGallery: () => [],
    showSubmitButton: true,
  },
);

const emit = defineEmits<{
  'pick-edit-source': [];
  'submit': [];
  'mask-preview': [payload: { dataUrl: string; width: number; height: number } | null];
}>();

const editorRoot = ref<HTMLElement | null>(null);
const mainCanvas = ref<HTMLCanvasElement | null>(null);

let img: CanvasImageSource | null = null;
let imgBitmap: ImageBitmap | null = null;
let loadToken = 0;
let fitAttempts = 0;
let maskCtx: CanvasRenderingContext2D | null = null;
let mainCtx: CanvasRenderingContext2D | null = null;
let offscreenCanvas: HTMLCanvasElement | null = null;
let displayWidth = 0;
let displayHeight = 0;

const currentTool = ref('brush');
const brushSize = ref(40);
const undoStack = ref<string[]>([]);
const redoStack = ref<string[]>([]);
const hasMaskContent = ref(false);
const lassoPoints = ref<{ x: number; y: number }[]>([]);
let isDrawing = false;

const zoom = ref(1);
const panX = ref(0);
const panY = ref(0);
let isPanning = false;
let lastPanX = 0;
let lastPanY = 0;
let isPinching = false;
let lastPinchDist = 0;

const MAX_UNDO = 20;
const MAX_IMAGE_EDGE = 2048;

function disposeImage() {
  if (imgBitmap) {
    imgBitmap.close();
    imgBitmap = null;
  }
  img = null;
}

function pushUndo() {
  if (!offscreenCanvas) return;
  const data = offscreenCanvas.toDataURL();
  undoStack.value.push(data);
  if (undoStack.value.length > MAX_UNDO) undoStack.value.shift();
  redoStack.value = [];
  updateHasMaskContent();
}

function updateHasMaskContent() {
  if (!offscreenCanvas) {
    hasMaskContent.value = false;
    return;
  }
  const ctx = offscreenCanvas.getContext('2d');
  if (!ctx) return;
  const { width, height } = offscreenCanvas;
  if (!width || !height) {
    hasMaskContent.value = false;
    return;
  }
  const imageData = ctx.getImageData(0, 0, width, height);
  const data = imageData.data;
  const stride = Math.max(4, Math.floor((width * 4) / 64));
  for (let i = 3; i < data.length; i += stride) {
    if (data[i] > 0) {
      hasMaskContent.value = true;
      return;
    }
  }
  hasMaskContent.value = false;
}

function draw() {
  if (!mainCtx || !img) return;
  const w = displayWidth;
  const h = displayHeight;
  mainCtx.clearRect(0, 0, w, h);
  mainCtx.save();
  mainCtx.translate(panX.value, panY.value);
  mainCtx.scale(zoom.value, zoom.value);
  mainCtx.drawImage(img, 0, 0, displayWidth, displayHeight);
  if (offscreenCanvas) {
    mainCtx.save();
    mainCtx.globalAlpha = 0.5;
    mainCtx.drawImage(offscreenCanvas, 0, 0, displayWidth, displayHeight);
    mainCtx.restore();
  }
  if (currentTool.value === 'lasso' && lassoPoints.value.length > 0) {
    mainCtx.save();
    mainCtx.strokeStyle = '#e94560';
    mainCtx.lineWidth = 2 / zoom.value;
    mainCtx.setLineDash([4 / zoom.value, 4 / zoom.value]);
    mainCtx.beginPath();
    mainCtx.moveTo(lassoPoints.value[0].x, lassoPoints.value[0].y);
    for (let i = 1; i < lassoPoints.value.length; i++) {
      mainCtx.lineTo(lassoPoints.value[i].x, lassoPoints.value[i].y);
    }
    mainCtx.stroke();
    mainCtx.setLineDash([]);
    mainCtx.restore();
  }
  mainCtx.restore();
  scheduleMaskPreviewEmit();
}

let maskPreviewTimer: ReturnType<typeof setTimeout> | null = null;

function emitMaskPreview() {
  if (!offscreenCanvas || !hasMaskContent.value || !displayWidth || !displayHeight) {
    emit('mask-preview', null);
    return;
  }
  const canvas = document.createElement('canvas');
  canvas.width = displayWidth;
  canvas.height = displayHeight;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.fillStyle = 'rgba(233, 69, 96, 0.62)';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.globalCompositeOperation = 'destination-in';
  ctx.drawImage(offscreenCanvas, 0, 0, displayWidth, displayHeight);
  emit('mask-preview', {
    dataUrl: canvas.toDataURL('image/png'),
    width: displayWidth,
    height: displayHeight,
  });
}

function scheduleMaskPreviewEmit() {
  if (maskPreviewTimer) return;
  maskPreviewTimer = setTimeout(() => {
    maskPreviewTimer = null;
    emitMaskPreview();
  }, 100);
}

function fitCanvas() {
  if (!img || !mainCanvas.value) return;
  const container = mainCanvas.value.parentElement;
  if (!container) return;
  const maxW = Math.max(0, container.clientWidth - 16);
  const maxH = Math.max(0, container.clientHeight - 16);
  if (maxW < 8 || maxH < 8) {
    if (fitAttempts < 24) {
      fitAttempts += 1;
      requestAnimationFrame(() => fitCanvas());
    }
    return;
  }
  fitAttempts = 0;
  const srcW = 'naturalWidth' in img ? img.naturalWidth : img.width;
  const srcH = 'naturalHeight' in img ? img.naturalHeight : img.height;
  if (!srcW || !srcH) return;
  let w: number;
  let h: number;
  if (srcW / srcH > maxW / maxH) {
    w = maxW;
    h = (srcH / srcW) * maxW;
  } else {
    h = maxH;
    w = (srcW / srcH) * maxH;
  }
  displayWidth = Math.max(1, Math.floor(w));
  displayHeight = Math.max(1, Math.floor(h));
  mainCanvas.value.width = displayWidth;
  mainCanvas.value.height = displayHeight;
  mainCtx = mainCanvas.value.getContext('2d');
  offscreenCanvas = document.createElement('canvas');
  offscreenCanvas.width = displayWidth;
  offscreenCanvas.height = displayHeight;
  maskCtx = offscreenCanvas.getContext('2d');
  if (maskCtx) maskCtx.fillStyle = '#e94560';
  draw();
}

async function loadImage(srcUrl: string) {
  const token = ++loadToken;
  disposeImage();
  try {
    const res = await fetch(srcUrl, { credentials: 'same-origin' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const blob = await res.blob();
    if (token !== loadToken) return;
    let bmp = await createImageBitmap(blob);
    if (token !== loadToken) {
      bmp.close();
      return;
    }
    if (bmp.width > MAX_IMAGE_EDGE || bmp.height > MAX_IMAGE_EDGE) {
      const scale = MAX_IMAGE_EDGE / Math.max(bmp.width, bmp.height);
      const rw = Math.max(1, Math.round(bmp.width * scale));
      const rh = Math.max(1, Math.round(bmp.height * scale));
      const resized = await createImageBitmap(bmp, {
        resizeWidth: rw,
        resizeHeight: rh,
        resizeQuality: 'high',
      });
      bmp.close();
      bmp = resized;
    }
    if (token !== loadToken) {
      bmp.close();
      return;
    }
    imgBitmap = bmp;
    img = bmp;
    await nextTick();
    if (token !== loadToken) return;
    fitCanvas();
  } catch (err) {
    console.error('ImageEditor load failed:', err);
    if (token === loadToken) disposeImage();
  }
}

function getCanvasPos(e: MouseEvent | TouchEvent) {
  const canvas = mainCanvas.value;
  if (!canvas) return { x: 0, y: 0 };
  const rect = canvas.getBoundingClientRect();
  let clientX: number;
  let clientY: number;
  if ('touches' in e && e.touches.length > 0) {
    clientX = e.touches[0].clientX;
    clientY = e.touches[0].clientY;
  } else if ('clientX' in e) {
    clientX = e.clientX;
    clientY = e.clientY;
  } else {
    return { x: 0, y: 0 };
  }
  const x = (clientX - rect.left - panX.value) / zoom.value;
  const y = (clientY - rect.top - panY.value) / zoom.value;
  return { x, y };
}

function drawBrushAt(x: number, y: number) {
  if (!maskCtx) return;
  maskCtx.save();
  maskCtx.globalCompositeOperation = currentTool.value === 'eraser' ? 'destination-out' : 'source-over';
  maskCtx.fillStyle = '#e94560';
  maskCtx.beginPath();
  const radius = brushSize.value / 2 / zoom.value;
  maskCtx.arc(x, y, radius, 0, Math.PI * 2);
  maskCtx.fill();
  maskCtx.restore();
}

function onPointerDown(e: MouseEvent) {
  if (!img || !maskCtx) return;
  editorRoot.value?.focus();
  if (e.button === 1 || (e.button === 0 && e.ctrlKey)) {
    isPanning = true;
    lastPanX = e.clientX;
    lastPanY = e.clientY;
    if (mainCanvas.value) mainCanvas.value.style.cursor = 'grabbing';
    e.preventDefault();
    return;
  }
  if (e.button !== 0) return;
  isDrawing = true;
  const pos = getCanvasPos(e);
  if (currentTool.value === 'brush' || currentTool.value === 'eraser') {
    pushUndo();
    drawBrushAt(pos.x, pos.y);
    draw();
  } else if (currentTool.value === 'lasso') {
    lassoPoints.value.push(pos);
    draw();
  }
}

function onPointerMove(e: MouseEvent) {
  if (isPanning) {
    panX.value += e.clientX - lastPanX;
    panY.value += e.clientY - lastPanY;
    lastPanX = e.clientX;
    lastPanY = e.clientY;
    draw();
    e.preventDefault();
    return;
  }
  if (!isDrawing || !maskCtx) return;
  const pos = getCanvasPos(e);
  if (currentTool.value === 'brush' || currentTool.value === 'eraser') {
    drawBrushAt(pos.x, pos.y);
    draw();
  }
}

function onPointerUp() {
  if (isPanning) {
    isPanning = false;
    if (mainCanvas.value) mainCanvas.value.style.cursor = 'crosshair';
    return;
  }
  if (!isDrawing) return;
  isDrawing = false;
  if (currentTool.value === 'brush' || currentTool.value === 'eraser') {
    updateHasMaskContent();
  }
}

function onTouchStart(e: TouchEvent) {
  if (!img || !maskCtx) return;
  editorRoot.value?.focus();
  e.preventDefault();
  if (e.touches.length === 2) {
    isPinching = true;
    isDrawing = false;
    const dx = e.touches[0].clientX - e.touches[1].clientX;
    const dy = e.touches[0].clientY - e.touches[1].clientY;
    lastPinchDist = Math.sqrt(dx * dx + dy * dy);
    return;
  }
  isDrawing = true;
  const pos = getCanvasPos(e);
  if (currentTool.value === 'brush' || currentTool.value === 'eraser') {
    pushUndo();
    drawBrushAt(pos.x, pos.y);
    draw();
  } else if (currentTool.value === 'lasso') {
    lassoPoints.value.push(pos);
    draw();
  }
}

function onTouchMove(e: TouchEvent) {
  e.preventDefault();
  if (e.touches.length === 2 && isPinching) {
    const dx = e.touches[0].clientX - e.touches[1].clientX;
    const dy = e.touches[0].clientY - e.touches[1].clientY;
    const dist = Math.sqrt(dx * dx + dy * dy);
    const center = {
      x: (e.touches[0].clientX + e.touches[1].clientX) / 2,
      y: (e.touches[0].clientY + e.touches[1].clientY) / 2,
    };
    const scale = dist / lastPinchDist;
    const newZoom = Math.max(0.25, Math.min(4, zoom.value * scale));
    const canvas = mainCanvas.value;
    if (canvas) {
      const rect = canvas.getBoundingClientRect();
      const canvasCenterX = center.x - rect.left;
      const canvasCenterY = center.y - rect.top;
      const worldX = (canvasCenterX - panX.value) / zoom.value;
      const worldY = (canvasCenterY - panY.value) / zoom.value;
      panX.value = canvasCenterX - worldX * newZoom;
      panY.value = canvasCenterY - worldY * newZoom;
    }
    zoom.value = newZoom;
    lastPinchDist = dist;
    draw();
    return;
  }
  if (!isDrawing || !maskCtx) return;
  const pos = getCanvasPos(e);
  if (currentTool.value === 'brush' || currentTool.value === 'eraser') {
    drawBrushAt(pos.x, pos.y);
    draw();
  }
}

function onTouchEnd(e: TouchEvent) {
  e.preventDefault();
  if (e.touches.length < 2) isPinching = false;
  if (!isDrawing) return;
  isDrawing = false;
  if (currentTool.value === 'brush' || currentTool.value === 'eraser') {
    updateHasMaskContent();
  }
}

function onWheel(e: WheelEvent) {
  if (!img || !mainCanvas.value) return;
  e.preventDefault();
  const rect = mainCanvas.value.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;
  const worldX = (mouseX - panX.value) / zoom.value;
  const worldY = (mouseY - panY.value) / zoom.value;
  const delta = e.deltaY > 0 ? 0.9 : 1.1;
  const newZoom = Math.max(0.25, Math.min(4, zoom.value * delta));
  panX.value = mouseX - worldX * newZoom;
  panY.value = mouseY - worldY * newZoom;
  zoom.value = newZoom;
  draw();
}

function zoomIn() {
  if (!img) return;
  const newZoom = Math.min(4, zoom.value * 1.25);
  const cx = displayWidth / 2;
  const cy = displayHeight / 2;
  const worldX = (cx - panX.value) / zoom.value;
  const worldY = (cy - panY.value) / zoom.value;
  panX.value = cx - worldX * newZoom;
  panY.value = cy - worldY * newZoom;
  zoom.value = newZoom;
  draw();
}

function zoomOut() {
  if (!img) return;
  const newZoom = Math.max(0.25, zoom.value / 1.25);
  const cx = displayWidth / 2;
  const cy = displayHeight / 2;
  const worldX = (cx - panX.value) / zoom.value;
  const worldY = (cy - panY.value) / zoom.value;
  panX.value = cx - worldX * newZoom;
  panY.value = cy - worldY * newZoom;
  zoom.value = newZoom;
  draw();
}

function resetView() {
  zoom.value = 1;
  panX.value = 0;
  panY.value = 0;
  draw();
}

function setTool(tool: string) {
  currentTool.value = tool;
  lassoPoints.value = [];
  draw();
  if (mainCanvas.value) mainCanvas.value.style.cursor = 'crosshair';
}

function closeLasso() {
  if (lassoPoints.value.length < 3 || !maskCtx) return;
  pushUndo();
  maskCtx.save();
  maskCtx.fillStyle = '#e94560';
  maskCtx.beginPath();
  maskCtx.moveTo(lassoPoints.value[0].x, lassoPoints.value[0].y);
  for (let i = 1; i < lassoPoints.value.length; i++) {
    maskCtx.lineTo(lassoPoints.value[i].x, lassoPoints.value[i].y);
  }
  maskCtx.closePath();
  maskCtx.fill();
  maskCtx.restore();
  lassoPoints.value = [];
  updateHasMaskContent();
  draw();
}

function invertMask() {
  if (!offscreenCanvas) return;
  pushUndo();
  const ctx = offscreenCanvas.getContext('2d');
  if (!ctx) return;
  const imageData = ctx.getImageData(0, 0, offscreenCanvas.width, offscreenCanvas.height);
  for (let i = 3; i < imageData.data.length; i += 4) {
    imageData.data[i] = imageData.data[i] > 0 ? 0 : 255;
  }
  ctx.putImageData(imageData, 0, 0);
  updateHasMaskContent();
  draw();
}

function clearMask() {
  if (!offscreenCanvas || !maskCtx) return;
  pushUndo();
  maskCtx.clearRect(0, 0, offscreenCanvas.width, offscreenCanvas.height);
  hasMaskContent.value = false;
  draw();
  emit('mask-preview', null);
}

function restoreMaskFromDataUrl(dataUrl: string) {
  if (!maskCtx || !offscreenCanvas) return;
  const tempImg = new Image();
  tempImg.onload = () => {
    maskCtx!.clearRect(0, 0, offscreenCanvas!.width, offscreenCanvas!.height);
    maskCtx!.drawImage(tempImg, 0, 0);
    updateHasMaskContent();
    draw();
  };
  tempImg.src = dataUrl;
}

function undo() {
  if (undoStack.value.length === 0 || !offscreenCanvas) return;
  redoStack.value.push(offscreenCanvas.toDataURL());
  const prev = undoStack.value.pop();
  if (prev) restoreMaskFromDataUrl(prev);
}

function redo() {
  if (redoStack.value.length === 0 || !offscreenCanvas) return;
  undoStack.value.push(offscreenCanvas.toDataURL());
  const next = redoStack.value.pop();
  if (next) restoreMaskFromDataUrl(next);
}

function onKeyDown(e: KeyboardEvent) {
  if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
    e.preventDefault();
    if (e.shiftKey) redo();
    else undo();
  }
  if (e.key === 'Enter' && currentTool.value === 'lasso' && lassoPoints.value.length >= 3) {
    e.preventDefault();
    closeLasso();
  }
}

/** Export a binary RGB mask: painted brush → white, clear → black (preview stays pink-red). */
function getMaskBlob(): Promise<Blob | null> {
  return new Promise((resolve) => {
    if (!offscreenCanvas) {
      resolve(null);
      return;
    }
    const w = offscreenCanvas.width;
    const h = offscreenCanvas.height;
    if (!w || !h) {
      resolve(null);
      return;
    }
    const srcCtx = offscreenCanvas.getContext('2d');
    if (!srcCtx) {
      resolve(null);
      return;
    }
    const src = srcCtx.getImageData(0, 0, w, h).data;
    const out = document.createElement('canvas');
    out.width = w;
    out.height = h;
    const outCtx = out.getContext('2d');
    if (!outCtx) {
      resolve(null);
      return;
    }
    const image = outCtx.createImageData(w, h);
    const dst = image.data;
    for (let i = 0; i < src.length; i += 4) {
      const painted = src[i + 3] > 0 || src[i] > 0 || src[i + 1] > 0 || src[i + 2] > 0;
      const v = painted ? 255 : 0;
      dst[i] = v;
      dst[i + 1] = v;
      dst[i + 2] = v;
      dst[i + 3] = 255;
    }
    outCtx.putImageData(image, 0, 0);
    out.toBlob((blob) => resolve(blob), 'image/png');
  });
}

let lastLoadedSrc = '';

onMounted(() => {
  if (props.src) {
    lastLoadedSrc = props.src;
    void loadImage(props.src);
  }
});

watch(
  () => props.src,
  (newSrc) => {
    if (!newSrc) {
      lastLoadedSrc = '';
      loadToken += 1;
      disposeImage();
      return;
    }
    if (newSrc === lastLoadedSrc) return;
    lastLoadedSrc = newSrc;
    undoStack.value = [];
    redoStack.value = [];
    hasMaskContent.value = false;
    lassoPoints.value = [];
    zoom.value = 1;
    panX.value = 0;
    panY.value = 0;
    emit('mask-preview', null);
    void loadImage(newSrc);
  },
);

defineExpose({ getMaskBlob, clearMask });
</script>

<style scoped>
.image-editor {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}

.editor-canvas-wrapper {
  flex: 1;
  min-height: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: auto;
  padding: 8px;
}

.editor-canvas {
  max-width: 100%;
  max-height: 100%;
  display: block;
}

.editor-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  color: var(--dq-label-tertiary);
}

.editor-empty__lead {
  font-size: var(--dq-font-size-body);
  font-weight: 500;
}

.editor-empty__subhint {
  font-size: var(--dq-font-size-caption);
  opacity: 0.7;
}

.editor-toolbar {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-top: 1px solid var(--dq-border-subtle);
  flex-wrap: wrap;
  flex-shrink: 0;
}

.toolbar-group {
  display: flex;
  align-items: center;
  gap: 4px;
}

.toolbar-divider {
  width: 1px;
  height: 20px;
  background: var(--dq-border-subtle);
}

.toolbar-label {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
  white-space: nowrap;
}

.toolbar-val {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
  min-width: 32px;
  text-align: right;
}

.editor-brush-slider {
  width: 120px;
  flex: 0 0 auto;
}

.toolbar-val--zoom {
  min-width: 48px;
  text-align: center;
}
</style>
