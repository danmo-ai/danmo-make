import { reactive, ref } from 'vue';
import { api } from '@/utils/api';
import type {
  GalleryItem,
  CanvasItemState,
  CanvasViewportState,
  CanvasStagingState,
  CanvasSessionState,
  CanvasSessionSummary,
  CanvasOverlaysState,
  CanvasOverlayLayer,
  CanvasOverlayKind,
  CanvasEdge,
  CanvasComposerSnapshot,
} from '@/types';
import {
  CANVAS_OVERLAY_KINDS,
  OVERLAY_DEFAULT_OPACITY,
} from '@/utils/canvasOverlays';
import { buildSessionEdges } from '@/utils/canvasEdges';
import {
  computeStagingDimensions,
  countItemsInStaging,
  layoutInStaging,
  lineageSlotBesideParent,
  parentAssetPath,
} from '@/utils/canvasStaging';
import {
  computeAlignPositions,
  computeDistributePositions,
  itemBounds,
  type AlignMode,
  type DistributeMode,
} from '@/utils/canvasGeometry';
import { DQ_STORAGE, getItem, setItem, removeItem } from '@/utils/storage';

const DEFAULT_VIEWPORT: CanvasViewportState = { zoom: 1, panX: 0, panY: 0 };
const DEFAULT_STAGING: CanvasStagingState = {
  x: 240,
  y: 180,
  width: 512,
  height: 512,
  visible: true,
};

function normalizeItemState(v: Record<string, unknown>): CanvasItemState {
  return {
    x: (v.x as number) ?? 0,
    y: (v.y as number) ?? 0,
    scale: (v.scale as number) ?? 0.5,
    visible: (v.visible as boolean) ?? true,
    zIndex: (v.zIndex as number) ?? 1,
    note: typeof v.note === 'string' ? v.note : '',
    label: typeof v.label === 'string' ? v.label : '',
    layerRole:
      v.layerRole === 'reference' || v.layerRole === 'control' || v.layerRole === 'asset'
        ? v.layerRole
        : 'asset',
  };
}

function normalizeOverlayLayer(v: unknown): CanvasOverlayLayer | null {
  if (!v || typeof v !== 'object') return null;
  const o = v as Record<string, unknown>;
  const path = String(o.path || '').trim();
  if (!path) return null;
  return {
    path,
    x: Number(o.x) || 0,
    y: Number(o.y) || 0,
    scale: Number(o.scale) || 0.5,
    opacity: Number(o.opacity) || 0.45,
    visible: o.visible !== false,
  };
}

function emptyOverlays(): CanvasOverlaysState {
  const base: CanvasOverlaysState = {};
  for (const key of CANVAS_OVERLAY_KINDS) {
    base[key] = null;
  }
  return base;
}

function normalizeOverlays(v: unknown): CanvasOverlaysState {
  const base = emptyOverlays();
  if (!v || typeof v !== 'object') return base;
  const o = v as Record<string, unknown>;
  for (const key of CANVAS_OVERLAY_KINDS) {
    base[key] = normalizeOverlayLayer(o[key]);
  }
  return base;
}

function normalizeStaging(v: Record<string, unknown> | undefined): CanvasStagingState {
  if (!v) return { ...DEFAULT_STAGING };
  return {
    x: Number(v.x) || DEFAULT_STAGING.x,
    y: Number(v.y) || DEFAULT_STAGING.y,
    width: Number(v.width) || DEFAULT_STAGING.width,
    height: Number(v.height) || DEFAULT_STAGING.height,
    visible: v.visible !== false,
  };
}

function normalizeComposerSnapshot(v: unknown): CanvasComposerSnapshot {
  if (!v || typeof v !== 'object') return {};
  const o = v as Record<string, unknown>;
  const snap: CanvasComposerSnapshot = {};
  for (const key of [
    'prompt',
    'title',
    'model',
    'version',
    'negative_prompt',
    'seed',
    'mode',
    'reference_path',
    'control_path',
    'controlnet',
    'controlnet_strength',
    'start_image_path',
    'tail_image_path',
    'source_video_path',
    'cover_source_path',
    'extend_directions',
    'extend_pixels',
    'editor_mode',
    'edit_asset_path',
    'retouch_model_version',
    'extend_model_version',
    'upscale_model_version',
    'upscale_scale',
    'upscale_denoise',
    'fill_edit_steps',
    'fill_edit_guidance',
  ] as const) {
    if (typeof o[key] === 'string') snap[key] = o[key];
  }
  if (Array.isArray(o.reference_image_paths)) {
    snap.reference_image_paths = o.reference_image_paths.filter(
      (p): p is string => typeof p === 'string' && p.length > 0,
    );
  }
  return snap;
}

function snapshotState(
  items: Record<string, CanvasItemState>,
  viewport: CanvasViewportState,
  staging: CanvasStagingState,
  activeAssetPath: string,
  overlays: CanvasOverlaysState,
  edges: CanvasEdge[],
  composerSnapshot: CanvasComposerSnapshot,
): CanvasSessionState {
  return {
    items: JSON.parse(JSON.stringify(items)),
    viewport: { ...viewport },
    staging: { ...staging },
    active_asset_path: activeAssetPath,
    overlays: JSON.parse(JSON.stringify(overlays)),
    edges: [...edges],
    composer_snapshot: Object.keys(composerSnapshot).length
      ? { ...composerSnapshot }
      : null,
  };
}

const LEGACY_ACTIVE_SESSION_KEY = 'dq-studio.canvas.activeSession.v4';

export type CanvasMedia = 'image' | 'video' | 'audio';

function migrateLegacyActiveSessionKey(media: CanvasMedia) {
  const current = activeSessionStorageKey(media);
  if (getItem(current)) return;
  const legacy = localStorage.getItem(LEGACY_ACTIVE_SESSION_KEY);
  if (legacy) {
    setItem(current, legacy);
    localStorage.removeItem(LEGACY_ACTIVE_SESSION_KEY);
  }
}

function activeSessionStorageKey(media: CanvasMedia) {
  if (media === 'video') return DQ_STORAGE.CANVAS_ACTIVE_SESSION_VIDEO;
  if (media === 'audio') return DQ_STORAGE.CANVAS_ACTIVE_SESSION_AUDIO;
  return DQ_STORAGE.CANVAS_ACTIVE_SESSION;
}

function legacyCanvasStorageKey(media: CanvasMedia) {
  if (media === 'video') return DQ_STORAGE.CANVAS_VIDEO;
  if (media === 'audio') return DQ_STORAGE.CANVAS_AUDIO;
  return DQ_STORAGE.CANVAS_IMAGE;
}

function createCanvasStore(media: CanvasMedia) {
  const legacyStorageKey = legacyCanvasStorageKey(media);
  const items = reactive<Record<string, CanvasItemState>>({});
  const viewport = reactive<CanvasViewportState>({ ...DEFAULT_VIEWPORT });
  const staging = reactive<CanvasStagingState>({ ...DEFAULT_STAGING });
  const overlays = reactive<CanvasOverlaysState>(emptyOverlays());
  const edges = ref<CanvasEdge[]>([]);
  const activeAssetPath = ref('');
  const sessionId = ref('');
  const sessionTitle = ref('Canvas');
  const sessions = ref<CanvasSessionSummary[]>([]);
  const ready = ref(false);
  const syncing = ref(false);
  const composerSnapshot = reactive<CanvasComposerSnapshot>({});

  let nextZIndex = 1;
  let persistTimer: ReturnType<typeof setTimeout> | null = null;
  let lastAddX = 80;
  let lastAddY = 80;

  function applyState(state: CanvasSessionState) {
    for (const key of Object.keys(items)) {
      delete items[key];
    }
    const rawItems = state.items || {};
    for (const [key, val] of Object.entries(rawItems)) {
      items[key] = normalizeItemState(val as unknown as Record<string, unknown>);
      nextZIndex = Math.max(nextZIndex, items[key].zIndex + 1);
    }
    const vp = state.viewport || DEFAULT_VIEWPORT;
    viewport.zoom = vp.zoom ?? 1;
    viewport.panX = vp.panX ?? 0;
    viewport.panY = vp.panY ?? 0;
    const st = normalizeStaging(state.staging as unknown as Record<string, unknown>);
    staging.x = st.x;
    staging.y = st.y;
    staging.width = st.width;
    staging.height = st.height;
    staging.visible = st.visible;
    activeAssetPath.value = state.active_asset_path || '';
    const ov = normalizeOverlays(state.overlays);
    for (const key of CANVAS_OVERLAY_KINDS) {
      overlays[key] = ov[key] ?? null;
    }
    edges.value = Array.isArray(state.edges) ? [...state.edges] : [];
    const snap = normalizeComposerSnapshot(state.composer_snapshot);
    for (const key of Object.keys(composerSnapshot)) {
      delete (composerSnapshot as Record<string, string | undefined>)[key];
    }
    Object.assign(composerSnapshot, snap);
  }

  function refreshEdges(galleryItems: GalleryItem[]) {
    edges.value = buildSessionEdges(items, galleryItems);
    schedulePersist();
  }

  function loadLegacyLocal(): CanvasSessionState | null {
    try {
      const raw = getItem(legacyStorageKey);
      if (!raw) return null;
      const data = JSON.parse(raw);
      return {
        items: data.items || {},
        viewport: data.viewport || DEFAULT_VIEWPORT,
        staging: normalizeStaging(data.staging),
        active_asset_path: data.active_asset_path || '',
      };
    } catch {
      return null;
    }
  }

  async function ensureSession(): Promise<void> {
    if (ready.value) return;
    migrateLegacyActiveSessionKey(media);
    syncing.value = true;
    try {
      let list = await api.canvas.listSessions(media);
      if (list.length === 0) {
        const legacy = loadLegacyLocal();
        const created = await api.canvas.createSession({
          media,
          title: 'Canvas 1',
          state: legacy || undefined,
        });
        if (legacy) removeItem(legacyStorageKey);
        sessionId.value = created.id;
        sessionTitle.value = created.title;
        applyState(created.state);
        list = await api.canvas.listSessions(media);
      } else {
        const savedId = getItem(activeSessionStorageKey(media));
        const pick = savedId && list.some((s) => s.id === savedId) ? savedId : list[0].id;
        const detail = await api.canvas.getSession(pick);
        sessionId.value = detail.id;
        sessionTitle.value = detail.title;
        applyState(detail.state);
        setItem(activeSessionStorageKey(media), pick);
      }
      sessions.value = list;
      ready.value = true;
    } catch (e) {
      console.error('Canvas session load failed, using local fallback:', e);
      const legacy = loadLegacyLocal();
      if (legacy) applyState(legacy);
      ready.value = true;
    } finally {
      syncing.value = false;
    }
  }

  function schedulePersist() {
    if (!ready.value || !sessionId.value) return;
    if (persistTimer) clearTimeout(persistTimer);
    persistTimer = setTimeout(persist, 400);
  }

  async function persist() {
    persistTimer = null;
    if (!sessionId.value) return;
    const state = snapshotState(
      items,
      viewport,
      staging,
      activeAssetPath.value,
      overlays,
      edges.value,
      composerSnapshot,
    );
    syncing.value = true;
    try {
      const updated = await api.canvas.updateSession(sessionId.value, { state });
      sessionTitle.value = updated.title;
      const idx = sessions.value.findIndex((s) => s.id === sessionId.value);
      if (idx >= 0) {
        sessions.value[idx] = {
          ...sessions.value[idx],
          item_count: Object.keys(state.items).length,
          updated_at: updated.updated_at,
        };
      }
    } catch (e) {
      console.error('Canvas session persist failed:', e);
      setItem(legacyStorageKey, JSON.stringify(state));
    } finally {
      syncing.value = false;
    }
  }

  async function switchSession(id: string) {
    if (id === sessionId.value) return;
    await persist();
    const detail = await api.canvas.getSession(id);
    sessionId.value = detail.id;
    sessionTitle.value = detail.title;
    applyState(detail.state);
    setItem(activeSessionStorageKey(media), id);
  }

  async function createSession(title?: string) {
    await persist();
    const n = sessions.value.length + 1;
    const created = await api.canvas.createSession({
      media,
      title: title || `Canvas ${n}`,
    });
    sessions.value = await api.canvas.listSessions(media);
    await switchSession(created.id);
    return created;
  }

  async function renameSession(title: string) {
    if (!sessionId.value) return;
    const t = title.trim();
    if (!t) return;
    syncing.value = true;
    try {
      const updated = await api.canvas.updateSession(sessionId.value, { title: t });
      sessionTitle.value = updated.title;
      const idx = sessions.value.findIndex((s) => s.id === sessionId.value);
      if (idx >= 0) sessions.value[idx] = { ...sessions.value[idx], title: updated.title };
    } finally {
      syncing.value = false;
    }
  }

  async function deleteSession(id: string) {
    if (sessions.value.length <= 1) return false;
    await api.canvas.deleteSession(id);
    sessions.value = await api.canvas.listSessions(media);
    if (sessionId.value === id) {
      const next = sessions.value[0];
      if (next) await switchSession(next.id);
    }
    return true;
  }

  function setActiveAssetPath(path: string) {
    activeAssetPath.value = path;
    schedulePersist();
  }

  function setComposerSnapshot(patch: CanvasComposerSnapshot) {
    const next = normalizeComposerSnapshot(patch);
    for (const key of Object.keys(composerSnapshot)) {
      delete (composerSnapshot as Record<string, unknown>)[key];
    }
    Object.assign(composerSnapshot, next);
    schedulePersist();
  }

  function updateStaging(patch: Partial<CanvasStagingState>) {
    Object.assign(staging, patch);
    schedulePersist();
  }

  function overlayPlacement(
    kind: CanvasOverlayKind,
    path: string,
    galleryItem: GalleryItem | undefined,
    scale: number
  ): { x: number; y: number } {
    const gw = (galleryItem?.width || 512) * scale;
    const gh = (galleryItem?.height || 512) * scale;
    if (kind === 'control') {
      return { x: staging.x, y: staging.y };
    }
    if (kind === 'tail_frame') {
      return {
        x: staging.x + staging.width - gw - 12,
        y: staging.y + (staging.height - gh) / 2,
      };
    }
    if (
      kind === 'start_frame' ||
      kind === 'video_source' ||
      kind === 'cover_source'
    ) {
      return {
        x: staging.x + (staging.width - gw) / 2,
        y: staging.y + (staging.height - gh) / 2,
      };
    }
    return {
      x: staging.x + (staging.width - gw) / 2,
      y: staging.y + (staging.height - gh) / 2,
    };
  }

  function setOverlay(
    kind: CanvasOverlayKind,
    path: string | null,
    galleryItem?: GalleryItem
  ) {
    if (!path) {
      overlays[kind] = null;
      schedulePersist();
      return;
    }
    const maxDim = Math.max(galleryItem?.width || 512, galleryItem?.height || 512);
    const scale = Math.min(1, staging.width / (maxDim || 512));
    const pos = overlayPlacement(kind, path, galleryItem, scale);
    overlays[kind] = {
      path,
      x: pos.x,
      y: pos.y,
      scale,
      opacity: OVERLAY_DEFAULT_OPACITY[kind],
      visible: true,
    };
    schedulePersist();
  }

  function setControlOverlay(path: string | null, galleryItem?: GalleryItem) {
    setOverlay('control', path, galleryItem);
  }

  function clearOverlay(kind: CanvasOverlayKind) {
    setOverlay(kind, null);
  }

  function updateOverlay(kind: CanvasOverlayKind, patch: Partial<CanvasOverlayLayer>) {
    const layer = overlays[kind];
    if (!layer) return;
    Object.assign(layer, patch);
    schedulePersist();
  }

  function applyStagingBesideBounds(
    minX: number,
    minY: number,
    maxX: number,
    refGi?: GalleryItem | null
  ) {
    const { width, height } = computeStagingDimensions(refGi);
    staging.x = maxX + 48;
    staging.y = minY;
    staging.width = width;
    staging.height = height;
    staging.visible = true;
    schedulePersist();
  }

  function placeStagingBeside(path: string, galleryItems: GalleryItem[]) {
    const state = items[path];
    const gi = galleryItems.find((g) => g.path === path);
    if (!state || !gi) return;
    const w = (gi.width || 512) * state.scale;
    applyStagingBesideBounds(state.x, state.y, state.x + w, gi);
  }

  function placeStagingBesideSelection(paths: string[], galleryItems: GalleryItem[]) {
    if (paths.length === 0) return;
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let refGi: GalleryItem | undefined;
    for (const path of paths) {
      const state = items[path];
      const gi = galleryItems.find((g) => g.path === path);
      if (!state || !gi) continue;
      const w = (gi.width || 512) * state.scale;
      const h = (gi.height || 512) * state.scale;
      minX = Math.min(minX, state.x);
      minY = Math.min(minY, state.y);
      maxX = Math.max(maxX, state.x + w);
      if (!refGi) refGi = gi;
    }
    if (!Number.isFinite(minX) || !Number.isFinite(maxX)) return;
    applyStagingBesideBounds(minX, minY, maxX, refGi);
  }

  function alignItems(paths: string[], mode: AlignMode, galleryItems: GalleryItem[]) {
    const bounds = paths
      .map((p) => itemBounds(p, items[p], galleryItems.find((g) => g.path === p)))
      .filter((b): b is NonNullable<typeof b> => b != null);
    const pos = computeAlignPositions(bounds, mode);
    for (const [path, xy] of Object.entries(pos)) {
      updatePosition(path, xy.x, xy.y);
    }
  }

  function distributeItems(paths: string[], mode: DistributeMode, galleryItems: GalleryItem[]) {
    const bounds = paths
      .map((p) => itemBounds(p, items[p], galleryItems.find((g) => g.path === p)))
      .filter((b): b is NonNullable<typeof b> => b != null);
    const pos = computeDistributePositions(bounds, mode);
    for (const [path, xy] of Object.entries(pos)) {
      updatePosition(path, xy.x, xy.y);
    }
  }

  function removeItems(paths: string[]) {
    for (const p of paths) removeItem(p);
  }

  function stagingCenter(): { x: number; y: number } {
    return {
      x: staging.x + staging.width / 2 - 120,
      y: staging.y + staging.height / 2 - 120,
    };
  }

  function addItem(item: GalleryItem, position?: { x: number; y: number }): boolean {
    const key = item.path;
    if (items[key]) return false;

    const isAudio =
      item.metadata?.asset_kind === 'audio' || item.mime_type?.startsWith('audio/');
    const baseW = isAudio ? 280 : item.width || 512;
    const baseH = isAudio ? 120 : item.height || 512;
    const maxDim = Math.max(baseW, baseH);
    const defaultScale = Math.min(1, 400 / (maxDim || 400));
    const pos = position ?? stagingCenter();

    items[key] = {
      x: pos.x,
      y: pos.y,
      scale: defaultScale,
      visible: true,
      zIndex: nextZIndex++,
      note: '',
    };

    lastAddX = pos.x + 40;
    lastAddY = pos.y + 40;
    schedulePersist();
    return true;
  }

  function removeItem(path: string) {
    delete items[path];
    if (activeAssetPath.value === path) activeAssetPath.value = '';
    schedulePersist();
  }

  function hasItem(path: string): boolean {
    return path in items;
  }

  function updatePosition(path: string, x: number, y: number) {
    const it = items[path];
    if (!it) return;
    it.x = x;
    it.y = y;
    schedulePersist();
  }

  function updateScale(path: string, scale: number) {
    const it = items[path];
    if (!it) return;
    it.scale = Math.max(0.1, Math.min(3, scale));
    schedulePersist();
  }

  function updateNote(path: string, note: string) {
    const it = items[path];
    if (!it) return;
    it.note = note;
    schedulePersist();
  }

  function updateLabel(path: string, label: string) {
    const it = items[path];
    if (!it) return;
    const trimmed = label.trim();
    if (trimmed) it.label = trimmed;
    else delete it.label;
    schedulePersist();
  }

  function toggleVisibility(path: string) {
    const it = items[path];
    if (!it) return;
    it.visible = !it.visible;
    schedulePersist();
  }

  function setVisibility(path: string, visible: boolean) {
    const it = items[path];
    if (!it || it.visible === visible) return;
    it.visible = visible;
    schedulePersist();
  }

  function bringToFront(path: string) {
    const it = items[path];
    if (!it) return;
    it.zIndex = nextZIndex++;
    schedulePersist();
  }

  function reorderItem(path: string, newZIndex: number) {
    const it = items[path];
    if (!it) return;
    it.zIndex = newZIndex;
    schedulePersist();
  }

  function setZoom(z: number, cx?: number, cy?: number) {
    const newZoom = Math.max(0.1, Math.min(5, z));
    if (cx !== undefined && cy !== undefined) {
      const worldX = (cx - viewport.panX) / viewport.zoom;
      const worldY = (cy - viewport.panY) / viewport.zoom;
      viewport.zoom = newZoom;
      viewport.panX = cx - worldX * newZoom;
      viewport.panY = cy - worldY * newZoom;
    } else {
      viewport.zoom = newZoom;
    }
    schedulePersist();
  }

  function setPan(x: number, y: number) {
    viewport.panX = x;
    viewport.panY = y;
    schedulePersist();
  }

  function expandBounds(
    minX: number,
    minY: number,
    maxX: number,
    maxY: number,
    x: number,
    y: number,
    w: number,
    h: number
  ): [number, number, number, number] {
    return [
      Math.min(minX, x),
      Math.min(minY, y),
      Math.max(maxX, x + w),
      Math.max(maxY, y + h),
    ];
  }

  function overlayBounds(
    layer: CanvasOverlayLayer | null,
    galleryItems: GalleryItem[]
  ): { x: number; y: number; w: number; h: number } | null {
    if (!layer?.path || layer.visible === false) return null;
    const gi = galleryItems.find((g) => g.path === layer.path);
    const baseW = gi?.width || 512;
    const baseH = gi?.height || 512;
    const scale = layer.scale || 0.5;
    return {
      x: layer.x,
      y: layer.y,
      w: baseW * scale,
      h: baseH * scale,
    };
  }

  function contentBounds(
    galleryItems: GalleryItem[]
  ): { minX: number; minY: number; maxX: number; maxY: number } | null {
    const keys = Object.keys(items);
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;

    for (const key of keys) {
      const it = items[key];
      const gi = galleryItems.find((g) => g.path === key);
      if (!gi || !it.visible) continue;
      const w = (gi.width || 512) * it.scale;
      const h = (gi.height || 512) * it.scale;
      [minX, minY, maxX, maxY] = expandBounds(minX, minY, maxX, maxY, it.x, it.y, w, h);
    }

    if (staging.visible) {
      [minX, minY, maxX, maxY] = expandBounds(
        minX,
        minY,
        maxX,
        maxY,
        staging.x,
        staging.y,
        staging.width,
        staging.height
      );
    }

    for (const key of CANVAS_OVERLAY_KINDS) {
      const ob = overlayBounds(overlays[key] ?? null, galleryItems);
      if (ob) {
        [minX, minY, maxX, maxY] = expandBounds(
          minX,
          minY,
          maxX,
          maxY,
          ob.x,
          ob.y,
          ob.w,
          ob.h
        );
      }
    }

    if (!isFinite(minX)) return null;
    return { minX, minY, maxX, maxY };
  }

  function applyDefaultViewport() {
    viewport.zoom = DEFAULT_VIEWPORT.zoom;
    viewport.panX = DEFAULT_VIEWPORT.panX;
    viewport.panY = DEFAULT_VIEWPORT.panY;
    schedulePersist();
  }

  /** Zoom 100% and center on canvas content (nodes / staging / overlays). */
  function resetView(galleryItems: GalleryItem[] = [], viewportEl?: HTMLElement | null) {
    const bounds = contentBounds(galleryItems);
    if (!bounds || !viewportEl) {
      applyDefaultViewport();
      return;
    }
    const { minX, minY, maxX, maxY } = bounds;
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const vw = viewportEl.clientWidth || 800;
    const vh = viewportEl.clientHeight || 600;
    viewport.zoom = 1;
    viewport.panX = vw / 2 - cx;
    viewport.panY = vh / 2 - cy;
    schedulePersist();
  }

  function fitAll(galleryItems: GalleryItem[], viewportEl?: HTMLElement | null) {
    const bounds = contentBounds(galleryItems);
    if (!bounds) {
      applyDefaultViewport();
      return;
    }

    const { minX, minY, maxX, maxY } = bounds;
    const pad = 48;
    const contentW = maxX - minX + pad * 2;
    const contentH = maxY - minY + pad * 2;

    if (viewportEl) {
      const vw = viewportEl.clientWidth || 800;
      const vh = viewportEl.clientHeight || 600;
      const zoomX = vw / contentW;
      const zoomY = vh / contentH;
      viewport.zoom = Math.max(0.15, Math.min(1.2, Math.min(zoomX, zoomY)));
      viewport.panX = (vw - contentW * viewport.zoom) / 2 - (minX - pad) * viewport.zoom;
      viewport.panY = (vh - contentH * viewport.zoom) / 2 - (minY - pad) * viewport.zoom;
    } else {
      viewport.zoom = 0.8;
      viewport.panX = -minX + 50;
      viewport.panY = -minY + 50;
    }
    schedulePersist();
  }

  function worldCenter(viewportEl: HTMLElement | null): { x: number; y: number } {
    if (!viewportEl) return stagingCenter();
    const cx = viewportEl.clientWidth / 2;
    const cy = viewportEl.clientHeight / 2;
    return {
      x: (cx - viewport.panX) / viewport.zoom - 120,
      y: (cy - viewport.panY) / viewport.zoom - 120,
    };
  }

  function pruneOrphans(validPaths: Set<string>) {
    let changed = false;
    for (const key of Object.keys(items)) {
      if (!validPaths.has(key)) {
        delete items[key];
        if (activeAssetPath.value === key) activeAssetPath.value = '';
        changed = true;
      }
    }
    if (changed) schedulePersist();
  }

  function addPathsFromGallery(
    paths: string[],
    galleryItems: GalleryItem[],
    opts?: { selectLast?: boolean; placement?: 'staging' | 'center' }
  ): string[] {
    const useStaging = opts?.placement === 'staging' && staging.visible;
    const center = stagingCenter();
    let offset = 0;
    let stagingIndex = useStaging ? countItemsInStaging(items, staging) : 0;
    const added: string[] = [];

    for (const path of paths) {
      const item = galleryItems.find((g) => g.path === path);
      if (!item || hasItem(path)) continue;

      const parentPath = parentAssetPath(item);
      if (parentPath && hasItem(parentPath)) {
        if (useStaging) {
          placeStagingBeside(parentPath, galleryItems);
        } else {
          const pos = lineageSlotBesideParent(item, galleryItems, items, added.length);
          if (pos && addItem(item, pos)) {
            added.push(path);
            continue;
          }
        }
      }

      if (useStaging) {
        if (stagingIndex === 0) {
          const dims = computeStagingDimensions(item);
          staging.width = dims.width;
          staging.height = dims.height;
        }
        const slot = layoutInStaging(stagingIndex, item, staging);
        const key = item.path;
        if (items[key]) continue;
        const isAudio =
          item.metadata?.asset_kind === 'audio' || item.mime_type?.startsWith('audio/');
        items[key] = {
          x: slot.x,
          y: slot.y,
          scale: slot.scale,
          visible: true,
          zIndex: nextZIndex++,
          note: '',
        };
        lastAddX = slot.x + 40;
        lastAddY = slot.y + 40;
        stagingIndex += 1;
        schedulePersist();
        added.push(path);
      } else {
        if (addItem(item, { x: center.x + offset, y: center.y + offset })) {
          offset += 48;
          added.push(path);
        }
      }
    }

    if (added.length > 0 && opts?.selectLast !== false) {
      activeAssetPath.value = added[added.length - 1];
    }
    refreshEdges(galleryItems);
    return added;
  }

  void ensureSession();

  return {
    items,
    viewport,
    staging,
    overlays,
    edges,
    activeAssetPath,
    sessionId,
    sessionTitle,
    sessions,
    ready,
    syncing,
    composerSnapshot,
    setComposerSnapshot,
    applyState,
    ensureSession,
    switchSession,
    createSession,
    renameSession,
    deleteSession,
    refreshEdges,
    setActiveAssetPath,
    updateStaging,
    setOverlay,
    setControlOverlay,
    clearOverlay,
    updateOverlay,
    placeStagingBeside,
    placeStagingBesideSelection,
    alignItems,
    distributeItems,
    removeItems,
    stagingCenter,
    addItem,
    removeItem,
    hasItem,
    updatePosition,
    updateScale,
    updateNote,
    updateLabel,
    toggleVisibility,
    setVisibility,
    bringToFront,
    reorderItem,
    setZoom,
    setPan,
    resetView,
    fitAll,
    worldCenter,
    pruneOrphans,
    addPathsFromGallery,
    persist,
  };
}

const storeByMedia = new Map<CanvasMedia, ReturnType<typeof createCanvasStore>>();

/** One reactive store per media; safe to call from create views while canvas is unmounted. */
export function useCanvasStore(media: CanvasMedia) {
  let inst = storeByMedia.get(media);
  if (!inst) {
    inst = createCanvasStore(media);
    storeByMedia.set(media, inst);
  }
  return inst;
}

export function canvasAutoAddEnabled(media: CanvasMedia = 'image'): boolean {
  const key =
    media === 'video'
      ? DQ_STORAGE.CANVAS_AUTO_ADD_VIDEO
      : media === 'audio'
        ? DQ_STORAGE.CANVAS_AUTO_ADD_AUDIO
        : DQ_STORAGE.CANVAS_AUTO_ADD;
  return getItem(key) === '1';
}

export function setCanvasAutoAdd(enabled: boolean, media: CanvasMedia = 'image'): void {
  const key =
    media === 'video'
      ? DQ_STORAGE.CANVAS_AUTO_ADD_VIDEO
      : media === 'audio'
        ? DQ_STORAGE.CANVAS_AUTO_ADD_AUDIO
        : DQ_STORAGE.CANVAS_AUTO_ADD;
  setItem(key, enabled ? '1' : '0');
}
