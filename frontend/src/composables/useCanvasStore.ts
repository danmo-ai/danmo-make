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
  clampStagingSize,
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
import { galleryStubForAssetPath } from '@/utils/canvasAssets';

const DEFAULT_VIEWPORT: CanvasViewportState = { zoom: 1, panX: 0, panY: 0 };
const DEFAULT_STAGING: CanvasStagingState = {
  x: 240,
  y: 180,
  width: 360,
  height: 360,
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
  const size = clampStagingSize(
    Number(v.width) || DEFAULT_STAGING.width,
    Number(v.height) || DEFAULT_STAGING.height,
  );
  return {
    x: Number(v.x) || DEFAULT_STAGING.x,
    y: Number(v.y) || DEFAULT_STAGING.y,
    width: size.width,
    height: size.height,
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

function normalizeShelf(raw: unknown): string[] {
  if (!Array.isArray(raw)) return [];
  const seen = new Set<string>();
  const out: string[] = [];
  for (const v of raw) {
    const p = typeof v === 'string' ? v.trim() : '';
    if (!p || seen.has(p)) continue;
    seen.add(p);
    out.push(p);
  }
  return out;
}

function snapshotState(
  items: Record<string, CanvasItemState>,
  shelf: string[],
  viewport: CanvasViewportState,
  staging: CanvasStagingState,
  activeAssetPath: string,
  overlays: CanvasOverlaysState,
  edges: CanvasEdge[],
  composerSnapshot: CanvasComposerSnapshot,
): CanvasSessionState {
  return {
    items: JSON.parse(JSON.stringify(items)),
    shelf: [...shelf],
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
  const shelf = ref<string[]>([]);
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
  /** Bumped to drop stale in-flight persist writes (e.g. after cross-session transfer). */
  let persistGeneration = 0;
  let lastAddX = 80;
  let lastAddY = 80;
  let ensurePromise: Promise<void> | null = null;
  let adoptPromise: Promise<{ adopted_count: number }> | null = null;

  function applyState(state: CanvasSessionState) {
    for (const key of Object.keys(items)) {
      delete items[key];
    }
    const rawItems = state.items || {};
    for (const [key, val] of Object.entries(rawItems)) {
      items[key] = normalizeItemState(val as unknown as Record<string, unknown>);
      nextZIndex = Math.max(nextZIndex, items[key].zIndex + 1);
    }
    // Shelf paths must not also sit on the board.
    shelf.value = normalizeShelf(state.shelf).filter((p) => !(p in items));
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

  function membershipCount(): number {
    const ids = new Set<string>();
    for (const key of Object.keys(items)) {
      if (key.startsWith('asset:')) ids.add(key.slice('asset:'.length));
    }
    for (const path of shelf.value) {
      if (path.startsWith('asset:')) ids.add(path.slice('asset:'.length));
    }
    return ids.size || Object.keys(items).length + shelf.value.length;
  }

  function previewAssetIds(limit = 6): string[] {
    const out: string[] = [];
    for (const key of Object.keys(items)) {
      if (out.length >= limit) break;
      if (key.startsWith('asset:')) out.push(key.slice('asset:'.length));
    }
    for (const path of shelf.value) {
      if (out.length >= limit) break;
      if (!path.startsWith('asset:')) continue;
      const id = path.slice('asset:'.length);
      if (!out.includes(id)) out.push(id);
    }
    return out;
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

  function defaultSessionTitle(): string {
    // Server/API titles stay English; UI localizes display. Matches adopt-orphans default.
    return 'Default canvas';
  }

  function isDefaultSessionTitle(title: string | undefined): boolean {
    const t = String(title || '').trim().toLowerCase();
    // Include legacy media-specific defaults created before titles were unified.
    return (
      t === 'default canvas' ||
      t === '默认画布' ||
      t === 'image canvas' ||
      t === 'video canvas' ||
      t === 'audio canvas'
    );
  }

  async function ensureSession(): Promise<void> {
    if (ready.value) return;
    if (ensurePromise) return ensurePromise;
    ensurePromise = (async () => {
      migrateLegacyActiveSessionKey(media);
      syncing.value = true;
      try {
        let list = await api.canvas.listSessions(media);
        if (list.length === 0) {
          const legacy = loadLegacyLocal();
          const created = await api.canvas.createSession({
            media,
            title: defaultSessionTitle(),
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
        // Fail loud: do not mark ready with a local-only fake session — that hides
        // server shelf membership (Option A) and blocks later ensureSession retries.
        console.error('Canvas session load failed:', e);
        ready.value = false;
        throw e;
      } finally {
        syncing.value = false;
        ensurePromise = null;
      }
    })();
    return ensurePromise;
  }

  /**
   * Keep a single named default canvas. Prefer one with assets; else oldest.
   * Empty duplicate defaults are deleted; non-empty extras are renamed.
   */
  async function consolidateDefaultCanvases(): Promise<void> {
    let list = await api.canvas.listSessions(media);
    // Unify legacy media-specific default titles onto "Default canvas".
    let renamed = false;
    for (const s of list) {
      const raw = String(s.title || '').trim();
      const lower = raw.toLowerCase();
      if (
        lower === 'video canvas' ||
        lower === 'audio canvas' ||
        lower === 'image canvas' ||
        raw === '默认画布'
      ) {
        await api.canvas.updateSession(s.id, { title: defaultSessionTitle() });
        renamed = true;
      }
    }
    if (renamed) list = await api.canvas.listSessions(media);

    const defaults = list
      .filter((s) => isDefaultSessionTitle(s.title))
      .sort((a, b) => {
        const byCount = (b.item_count || 0) - (a.item_count || 0);
        if (byCount !== 0) return byCount;
        const c = String(a.created_at || '').localeCompare(String(b.created_at || ''));
        if (c !== 0) return c;
        return String(a.id).localeCompare(String(b.id));
      });
    if (defaults.length <= 1) {
      sessions.value = list;
      return;
    }
    const keep = defaults[0];
    const prev = sessionId.value;
    for (let i = 1; i < defaults.length; i += 1) {
      const extra = defaults[i];
      if ((extra.item_count || 0) <= 0) {
        await api.canvas.deleteSession(extra.id);
        if (prev === extra.id) {
          sessionId.value = keep.id;
          setItem(activeSessionStorageKey(media), keep.id);
        }
      } else {
        const renamed = `Canvas ${list.length + i}`;
        await api.canvas.updateSession(extra.id, { title: renamed });
      }
    }
    sessions.value = await api.canvas.listSessions(media);
    if (sessionId.value === keep.id || !sessionId.value) {
      const detail = await api.canvas.getSession(keep.id);
      sessionId.value = detail.id;
      sessionTitle.value = detail.title;
      applyState(detail.state);
      setItem(activeSessionStorageKey(media), keep.id);
    }
  }

  /** Ensure ≥1 session, then adopt orphan gallery assets of this media into default canvas. */
  async function ensureDefaultAndAdoptOrphans(): Promise<{ adopted_count: number }> {
    // Guard BEFORE any await so concurrent callers share one run (avoids dual Default canvas).
    if (adoptPromise) return adoptPromise;
    adoptPromise = (async () => {
      // Recover from a prior failed load that left ready=true without a session id.
      if (ready.value && !sessionId.value) ready.value = false;
      await ensureSession();
      syncing.value = true;
      try {
        await consolidateDefaultCanvases();
        let list = sessions.value.length
          ? sessions.value
          : await api.canvas.listSessions(media);
        if (!list.some((s) => isDefaultSessionTitle(s.title))) {
          // Create via API only — do not use createSession() (it switches focus).
          await api.canvas.createSession({
            media,
            title: defaultSessionTitle(),
          });
          // Race-safe: collapse if another tab/call also created one.
          await consolidateDefaultCanvases();
          list = sessions.value;
        }
        sessions.value = list;

        let adopted = 0;
        try {
          const result = await api.canvas.adoptOrphans(media);
          adopted = result.adopted_count || 0;
        } catch (e) {
          // Keep the default canvas usable even if adopt fails (stale API, etc.).
          console.error('Canvas adopt-orphans failed:', e);
          sessions.value = await api.canvas.listSessions(media);
          throw e;
        }
        sessions.value = await api.canvas.listSessions(media);
        // Re-fetch when orphans landed on the shelf (or to heal a stale board).
        if (adopted > 0 && sessionId.value) {
          const detail = await api.canvas.getSession(sessionId.value);
          applyState(detail.state);
        }
        return { adopted_count: adopted };
      } catch (e) {
        console.error('Canvas ensureDefaultAndAdoptOrphans failed:', e);
        throw e;
      } finally {
        syncing.value = false;
        adoptPromise = null;
      }
    })();
    return adoptPromise;
  }

  async function refreshSessions(): Promise<void> {
    sessions.value = await api.canvas.listSessions(media);
  }

  function cancelPendingPersist() {
    if (persistTimer) {
      clearTimeout(persistTimer);
      persistTimer = null;
    }
    persistGeneration += 1;
  }

  function schedulePersist() {
    if (!ready.value || !sessionId.value) return;
    if (persistTimer) clearTimeout(persistTimer);
    persistTimer = setTimeout(() => {
      void persist();
    }, 400);
  }

  async function persist() {
    persistTimer = null;
    if (!sessionId.value) return;
    const gen = persistGeneration;
    const sid = sessionId.value;
    const state = snapshotState(
      items,
      shelf.value,
      viewport,
      staging,
      activeAssetPath.value,
      overlays,
      edges.value,
      composerSnapshot,
    );
    syncing.value = true;
    try {
      const updated = await api.canvas.updateSession(sid, { state });
      // A newer cancel/transfer won — do not clobber session list metadata from a stale write.
      if (gen !== persistGeneration || sid !== sessionId.value) return;
      sessionTitle.value = updated.title;
      const idx = sessions.value.findIndex((s) => s.id === sid);
      if (idx >= 0) {
        sessions.value[idx] = {
          ...sessions.value[idx],
          item_count: membershipCount(),
          preview_asset_ids: previewAssetIds(6),
          updated_at: updated.updated_at,
        };
      }
    } catch (e) {
      console.error('Canvas session persist failed:', e);
      if (gen === persistGeneration) {
        setItem(legacyStorageKey, JSON.stringify(state));
      }
    } finally {
      if (gen === persistGeneration) syncing.value = false;
      else if (!persistTimer) syncing.value = false;
    }
  }

  /** Wait until any in-flight persist finishes (after canceling the debounce timer). */
  async function flushPersistBarrier(): Promise<void> {
    cancelPendingPersist();
    const start = Date.now();
    while (syncing.value && Date.now() - start < 5000) {
      await new Promise((r) => setTimeout(r, 20));
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
      title: title || (media === 'image' ? `Canvas ${n}` : `Canvas ${n}`),
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
    if (sessions.value.length <= 1) {
      throw new Error('cannot_delete_last_canvas');
    }
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
    if (patch.width != null || patch.height != null) {
      const size = clampStagingSize(staging.width, staging.height);
      staging.width = size.width;
      staging.height = size.height;
    }
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

  /**
   * Drop board/shelf paths that are known-deleted.
   * Caller must pass an authoritative path set (full library), not a single gallery page.
   */
  function pruneOrphans(validPaths: Set<string>, opts?: { authoritative?: boolean }) {
    if (!opts?.authoritative) return;
    if (validPaths.size === 0) return;
    let changed = false;
    for (const key of Object.keys(items)) {
      if (!validPaths.has(key)) {
        delete items[key];
        if (activeAssetPath.value === key) activeAssetPath.value = '';
        changed = true;
      }
    }
    const nextShelf = shelf.value.filter((p) => validPaths.has(p));
    if (nextShelf.length !== shelf.value.length) {
      shelf.value = nextShelf;
      changed = true;
    }
    if (changed) schedulePersist();
  }

  function addToShelf(paths: string[]): string[] {
    const added: string[] = [];
    const set = new Set(shelf.value);
    for (const path of paths) {
      const p = String(path || '').trim();
      if (!p || set.has(p) || p in items) continue;
      set.add(p);
      added.push(p);
    }
    if (added.length) {
      shelf.value = [...shelf.value, ...added];
      schedulePersist();
    }
    return added;
  }

  function removeFromShelf(paths: string[]) {
    const drop = new Set(paths);
    const next = shelf.value.filter((p) => !drop.has(p));
    if (next.length !== shelf.value.length) {
      shelf.value = next;
      schedulePersist();
    }
  }

  /** Promote shelf paths onto the freeform board (removes from shelf). */
  function promoteToBoard(
    paths: string[],
    galleryItems: GalleryItem[],
    opts?: { placement?: 'staging' | 'center' },
  ): string[] {
    removeFromShelf(paths);
    return addPathsFromGallery(paths, galleryItems, {
      placement: opts?.placement ?? 'staging',
      selectLast: true,
    });
  }

  /** Remove from board and keep membership on the shelf. */
  function demoteToShelf(paths: string[]) {
    const kept: string[] = [];
    for (const path of paths) {
      if (!(path in items)) continue;
      delete items[path];
      if (activeAssetPath.value === path) activeAssetPath.value = '';
      kept.push(path);
    }
    if (kept.length) {
      addToShelf(kept);
      schedulePersist();
    }
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
      if (hasItem(path)) continue;
      const item =
        galleryItems.find((g) => g.path === path) ||
        (path.startsWith('asset:') ? galleryStubForAssetPath(path) : null);
      if (!item) continue;
      // Board placement wins over shelf membership.
      removeFromShelf([path]);

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
    shelf,
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
    ensureDefaultAndAdoptOrphans,
    refreshSessions,
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
    addToShelf,
    removeFromShelf,
    promoteToBoard,
    demoteToShelf,
    membershipCount,
    previewAssetIds,
    persist,
    cancelPendingPersist,
    flushPersistBarrier,
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
