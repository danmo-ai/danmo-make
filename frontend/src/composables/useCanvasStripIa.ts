import { computed, ref, unref, type ComputedRef, type Ref } from 'vue';
import { api } from '@/utils/api';
import { $tt } from '@/utils/i18n';
import { toast } from '@/utils/feedback';
import { assetIdFromGalleryPath } from '@/utils/copilotHandoff';
import { galleryStubForAssetPath } from '@/utils/canvasAssets';
import { CANVAS_OVERLAY_KINDS } from '@/utils/canvasOverlays';
import { lyricsSnippetLines } from '@/utils/lyricsTimeline';
import type { CanvasSessionState, GalleryItem } from '@/types';
import type { CanvasMedia } from '@/composables/useCanvasStore';

type CanvasStore = ReturnType<typeof import('@/composables/useCanvasStore').useCanvasStore>;

function canvasTitleLabel(title: string): string {
  const t = (title || '').trim().toLowerCase();
  if (
    t === 'default canvas' ||
    t === '默认画布' ||
    t === 'image canvas' ||
    t === 'video canvas' ||
    t === 'audio canvas'
  ) {
    return $tt('canvas.defaultCanvasTitle');
  }
  return title || $tt('canvas.untitledSession');
}

function sortGalleryItemsByTime(items: GalleryItem[]): GalleryItem[] {
  return [...items].sort((a, b) => {
    const c = String(b.created_at || '').localeCompare(String(a.created_at || ''));
    if (c !== 0) return c;
    return String(b.path || '').localeCompare(String(a.path || ''));
  });
}

function sessionHasAssetPath(
  state: { items?: Record<string, unknown>; shelf?: string[] } | null | undefined,
  path: string,
): boolean {
  if (!state || !path) return false;
  if (state.items && path in state.items) return true;
  return (state.shelf || []).includes(path);
}

function stripAssetFromSessionState(
  state: CanvasSessionState,
  path: string,
): CanvasSessionState {
  const items = { ...(state.items || {}) };
  delete items[path];
  const shelf = (state.shelf || []).filter((p) => p !== path);
  const overlays = { ...(state.overlays || {}) };
  for (const kind of CANVAS_OVERLAY_KINDS) {
    if (overlays[kind]?.path === path) overlays[kind] = null;
  }
  const active = state.active_asset_path === path ? '' : state.active_asset_path || '';
  const edges = (state.edges || []).filter((e) => e?.from !== path && e?.to !== path);
  return {
    ...state,
    items,
    shelf,
    overlays,
    active_asset_path: active,
    edges,
  };
}

/**
 * Shared strip-list / preview-transfer IA for image, video, and audio create pages.
 */
export function useCanvasStripIa(opts: {
  media: CanvasMedia;
  store: CanvasStore;
  galleryItems: Ref<GalleryItem[]> | ComputedRef<GalleryItem[]>;
  viewMode: Ref<'grid' | 'canvas'>;
  previewVisible: Ref<boolean>;
  selectedIndex: Ref<number>;
  /** Called after switching session so the page can restore composer snapshot. */
  onComposerRestore?: (snapshot: Record<string, unknown>) => void;
  /** Open the media-specific text/create composer (strip 「创作」). */
  openCreateComposer: () => void;
  /** Optional gallery reload after strip refresh. */
  reloadGallery?: () => Promise<void> | void;
}) {
  const {
    store,
    galleryItems,
    viewMode,
    previewVisible,
    selectedIndex,
    onComposerRestore,
    openCreateComposer,
    reloadGallery,
  } = opts;

  const stripSearchText = ref('');
  const stripRenameOpen = ref(false);
  const stripRenameDraft = ref('');
  const stripRenameId = ref('');
  const stripPreviewOverride = ref<GalleryItem[] | null>(null);
  const previewSourceSessionId = ref('');
  const sessionAssetIdsBySession = ref<Record<string, string[]>>({});
  /**
   * Full gallery rows for canvas membership not yet in the paged list.
   * Used so strip tiles + detail board/dock can show lyrics/title.
   */
  const stripExtraItemsById = ref<Record<string, GalleryItem>>({});

  const canvasSessions = computed(() => unref(store.sessions));
  const canvasSessionId = computed(() => unref(store.sessionId));
  const canvasSessionTitle = computed(() => {
    const raw = unref(store.sessionTitle) || '';
    return canvasTitleLabel(raw);
  });
  const canvasReady = computed(() => unref(store.ready));

  function metaFromGalleryItem(item: GalleryItem) {
    const meta = (item.metadata || {}) as {
      lyrics_effective?: unknown;
      lyrics_segments?: unknown;
    };
    return {
      prompt: item.prompt,
      model: item.model,
      title: item.title,
      name: item.name,
      created_at: item.created_at,
      lyricsLines: lyricsSnippetLines(meta, 3),
    };
  }

  /** Gallery items plus hydrated canvas-membership rows (audio lyrics, etc.). */
  const enrichedGalleryItems = computed(() => {
    const map = new Map<string, GalleryItem>();
    for (const item of galleryItems.value) {
      const id = assetIdFromGalleryPath(item.path);
      if (id) map.set(id, item);
    }
    for (const [id, item] of Object.entries(stripExtraItemsById.value)) {
      if (!map.has(id)) map.set(id, item);
    }
    return [...map.values()];
  });

  const galleryByAssetId = computed(() => {
    const map: Record<
      string,
      {
        prompt?: string;
        model?: string;
        title?: string;
        name?: string;
        created_at?: string;
        lyricsLines?: string[];
      }
    > = {};
    for (const item of enrichedGalleryItems.value) {
      const id = assetIdFromGalleryPath(item.path);
      if (!id) continue;
      map[id] = metaFromGalleryItem(item);
    }
    return map;
  });

  async function hydrateStripAssetMeta(assetIds: string[]) {
    if (opts.media !== 'audio' || !assetIds.length) return;
    const have = new Set(enrichedGalleryItems.value.map((g) => assetIdFromGalleryPath(g.path)));
    const missing = assetIds.filter((id) => id && !have.has(id));
    if (!missing.length) return;

    const need = new Set(missing);
    const added: Record<string, GalleryItem> = {};
    let offset = 0;
    const page = 100;
    while (need.size > 0 && offset < 5000) {
      const rows = await api.gallery.listImages(page, offset, { kind: 'audio' });
      if (!rows.length) break;
      for (const item of rows) {
        const id = assetIdFromGalleryPath(item.path);
        if (!id || !need.has(id)) continue;
        added[id] = item;
        need.delete(id);
      }
      if (rows.length < page) break;
      offset += page;
    }
    if (Object.keys(added).length) {
      stripExtraItemsById.value = { ...stripExtraItemsById.value, ...added };
    }
  }

  async function rebuildSessionAssetIndex() {
    const sessions = canvasSessions.value || [];
    if (!sessions.length) {
      sessionAssetIdsBySession.value = {};
      return;
    }
    const entries = await Promise.all(
      sessions.map(async (s) => {
        try {
          const detail = await api.canvas.getSession(s.id);
          const ids: string[] = [];
          const seen = new Set<string>();
          for (const p of Object.keys(detail?.state?.items || {})) {
            if (!p.startsWith('asset:')) continue;
            const id = p.slice('asset:'.length);
            if (!id || seen.has(id)) continue;
            seen.add(id);
            ids.push(id);
          }
          for (const p of detail?.state?.shelf || []) {
            if (typeof p !== 'string' || !p.startsWith('asset:')) continue;
            const id = p.slice('asset:'.length);
            if (!id || seen.has(id)) continue;
            seen.add(id);
            ids.push(id);
          }
          return [s.id, ids] as const;
        } catch {
          return [s.id, s.preview_asset_ids || []] as const;
        }
      }),
    );
    sessionAssetIdsBySession.value = Object.fromEntries(entries);
    const allIds = [...new Set(entries.flatMap(([, ids]) => ids))];
    await hydrateStripAssetMeta(allIds);
  }

  const previewItems = computed(() => {
    if (stripPreviewOverride.value?.length) return stripPreviewOverride.value;
    if (viewMode.value !== 'canvas') return enrichedGalleryItems.value;
    const byPath = new Map(enrichedGalleryItems.value.map((g) => [g.path, g]));
    const paths = new Set<string>([
      ...Object.keys(store.items),
      ...unref(store.shelf),
    ]);
    const items = [...paths]
      .filter((p) => p.startsWith('asset:'))
      .map((path) => byPath.get(path) || galleryStubForAssetPath(path));
    return sortGalleryItemsByTime(items);
  });

  function resolvePreviewSourceSessionId(path?: string): string {
    if (previewSourceSessionId.value) return previewSourceSessionId.value;
    if (viewMode.value === 'canvas' && canvasSessionId.value) return canvasSessionId.value;
    const p = path || previewItems.value[selectedIndex.value]?.path || '';
    if (p) {
      const assetId = assetIdFromGalleryPath(p);
      for (const [sid, ids] of Object.entries(sessionAssetIdsBySession.value || {})) {
        if (assetId && ids.includes(assetId)) return sid;
      }
    }
    return canvasSessionId.value || '';
  }

  const previewTransferTargets = computed(() => {
    const source = resolvePreviewSourceSessionId();
    return (canvasSessions.value || [])
      .filter((s) => s.id && s.id !== source)
      .map((s) => ({ id: s.id, title: canvasTitleLabel(s.title || '') }));
  });

  async function enterCanvasDetail(
    sessionId: string,
    setViewMode: (mode: 'grid' | 'canvas') => void,
    opts?: { openComposer?: boolean },
  ) {
    if (sessionId && sessionId !== canvasSessionId.value) {
      await store.switchSession(sessionId);
      onComposerRestore?.({ ...store.composerSnapshot });
    }
    setViewMode('canvas');
    if (opts?.openComposer) {
      openCreateComposer();
    }
  }

  async function onStripOpen(
    sessionId: string,
    setViewMode: (mode: 'grid' | 'canvas') => void,
  ) {
    await enterCanvasDetail(sessionId, setViewMode, { openComposer: false });
  }

  async function onStripPreview(payload: { sessionId: string; assetId: string }) {
    const sessionId = String(payload?.sessionId || '').trim();
    const assetId = String(payload?.assetId || '').trim();
    if (!sessionId || !assetId) return;

    const byPath = new Map(enrichedGalleryItems.value.map((g) => [g.path, g]));
    const resolveItem = (path: string): GalleryItem =>
      byPath.get(path) || galleryStubForAssetPath(path);

    let paths: string[] = [];
    try {
      const detail = await api.canvas.getSession(sessionId);
      const items = detail?.state?.items || {};
      const board = Object.entries(items)
        .sort((a, b) => (Number(a[1]?.zIndex) || 0) - (Number(b[1]?.zIndex) || 0))
        .map(([path]) => path)
        .filter((p) => p.startsWith('asset:'));
      const boardSet = new Set(board);
      const shelf = (detail?.state?.shelf || []).filter(
        (p) => typeof p === 'string' && p.startsWith('asset:') && !boardSet.has(p),
      );
      paths = [...board, ...shelf];
    } catch (e) {
      toast.error($tt('studio.error', { msg: String(e) }));
      return;
    }

    if (!paths.length) paths = [`asset:${assetId}`];

    let list = sortGalleryItemsByTime(paths.map(resolveItem));
    const q = stripSearchText.value.trim().toLowerCase();
    if (q) {
      const matched = list.filter((g) => {
        const hay = [g.title, g.prompt, g.model, g.name]
          .filter(Boolean)
          .join(' ')
          .toLowerCase();
        return hay.includes(q);
      });
      if (matched.length) list = matched;
    }
    let idx = list.findIndex((g) => assetIdFromGalleryPath(g.path) === assetId);
    if (idx < 0) {
      list = [resolveItem(`asset:${assetId}`), ...list];
      idx = 0;
    }

    stripPreviewOverride.value = list;
    previewSourceSessionId.value = sessionId;
    selectedIndex.value = idx;
    previewVisible.value = true;
  }

  async function onStripCreate(sessionId: string) {
    if (sessionId && sessionId !== canvasSessionId.value) {
      await store.switchSession(sessionId);
      onComposerRestore?.({ ...store.composerSnapshot });
    }
    openCreateComposer();
  }

  function onBackToStripList(setViewMode: (mode: 'grid' | 'canvas') => void) {
    setViewMode('grid');
    void store.refreshSessions().then(() => rebuildSessionAssetIndex());
  }

  async function onNewCanvas(setViewMode: (mode: 'grid' | 'canvas') => void) {
    try {
      const created = await store.createSession();
      toast.success($tt('canvas.sessionCreated'));
      await rebuildSessionAssetIndex();
      await enterCanvasDetail(created.id, setViewMode, { openComposer: true });
    } catch (e) {
      toast.error($tt('studio.error', { msg: String(e) }));
    }
  }

  async function refreshStripList() {
    try {
      if (!unref(store.ready) || !unref(store.sessionId)) {
        await store.ensureDefaultAndAdoptOrphans();
      } else {
        await store.refreshSessions();
      }
      await reloadGallery?.();
      await rebuildSessionAssetIndex();
    } catch (e) {
      toast.error($tt('studio.error', { msg: String(e) }));
    }
  }

  function onStripRename(sessionId: string) {
    const s = canvasSessions.value.find((x) => x.id === sessionId);
    stripRenameId.value = sessionId;
    stripRenameDraft.value = s?.title || '';
    stripRenameOpen.value = true;
  }

  async function commitStripRename() {
    stripRenameOpen.value = false;
    const id = stripRenameId.value;
    const t = stripRenameDraft.value.trim();
    if (!id || !t) return;
    const prev = canvasSessionId.value;
    try {
      if (id !== prev) await store.switchSession(id);
      await store.renameSession(t);
      if (prev && prev !== id) await store.switchSession(prev);
      await store.refreshSessions();
      toast.success($tt('canvas.sessionRenamed'));
    } catch (e) {
      toast.error($tt('studio.error', { msg: String(e) }));
    }
  }

  async function onStripDelete(sessionId: string) {
    try {
      await store.deleteSession(sessionId);
      toast.success($tt('canvas.sessionDeleted'));
      await rebuildSessionAssetIndex();
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      if (msg.includes('cannot_delete_last_canvas') || msg.includes('last canvas')) {
        toast.error($tt('canvas.sessionCannotDeleteLast'));
        return;
      }
      toast.error($tt('studio.error', { msg }));
    }
  }

  function markPreviewSource(sessionId?: string) {
    previewSourceSessionId.value = sessionId || canvasSessionId.value || '';
  }

  function clearPreviewOverride() {
    stripPreviewOverride.value = null;
    previewSourceSessionId.value = '';
  }

  async function onPreviewTransfer(payload: {
    item: GalleryItem;
    targetSessionId: string;
  }) {
    const path = String(payload?.item?.path || '').trim();
    const toId = String(payload?.targetSessionId || '').trim();
    if (!path.startsWith('asset:') || !toId) return;

    const fromId = resolvePreviewSourceSessionId(path);
    if (!fromId) {
      toast.error($tt('studio.error', { msg: $tt('canvas.transferNeedSource') }));
      return;
    }
    if (fromId === toId) return;

    try {
      await store.flushPersistBarrier();

      const [fromDetail, toDetail] = await Promise.all([
        api.canvas.getSession(fromId),
        api.canvas.getSession(toId),
      ]);

      const fromState = stripAssetFromSessionState(fromDetail.state, path);
      const toItems = { ...(toDetail.state?.items || {}) };
      const toShelf = [...(toDetail.state?.shelf || [])].filter((p) => p !== path);
      if (!(path in toItems) && !toShelf.includes(path)) toShelf.push(path);
      const toState = {
        ...toDetail.state,
        items: toItems,
        shelf: toShelf,
      };

      await api.canvas.updateSession(fromId, { state: fromState });
      await api.canvas.updateSession(toId, { state: toState });

      const verifyFrom = await api.canvas.getSession(fromId);
      if (sessionHasAssetPath(verifyFrom.state, path)) {
        await api.canvas.updateSession(fromId, {
          state: stripAssetFromSessionState(verifyFrom.state, path),
        });
      }

      const liveId = unref(store.sessionId);
      if (liveId === fromId || liveId === toId) {
        store.cancelPendingPersist();
        const live = await api.canvas.getSession(liveId);
        store.applyState(live.state);
      }

      if (stripPreviewOverride.value?.length) {
        const next = stripPreviewOverride.value.filter((g) => g.path !== path);
        stripPreviewOverride.value = next.length ? next : null;
        if (!next.length) {
          previewVisible.value = false;
        } else {
          selectedIndex.value = Math.min(selectedIndex.value, next.length - 1);
        }
      }

      await store.refreshSessions();
      await rebuildSessionAssetIndex();

      const targetTitle =
        canvasTitleLabel(
          (canvasSessions.value || []).find((s) => s.id === toId)?.title || '',
        ) || toId;
      toast.success($tt('canvas.transferToCanvasDone', { title: targetTitle }));
    } catch (e) {
      toast.error($tt('studio.error', { msg: String(e) }));
    }
  }

  return {
    stripSearchText,
    stripRenameOpen,
    stripRenameDraft,
    stripRenameId,
    stripPreviewOverride,
    previewSourceSessionId,
    sessionAssetIdsBySession,
    canvasSessions,
    canvasSessionId,
    canvasSessionTitle,
    canvasReady,
    galleryByAssetId,
    enrichedGalleryItems,
    previewItems,
    previewTransferTargets,
    rebuildSessionAssetIndex,
    enterCanvasDetail,
    onStripOpen,
    onStripPreview,
    onStripCreate,
    onBackToStripList,
    onNewCanvas,
    refreshStripList,
    onStripRename,
    commitStripRename,
    onStripDelete,
    markPreviewSource,
    clearPreviewOverride,
    onPreviewTransfer,
    canvasTitleLabel,
  };
}
