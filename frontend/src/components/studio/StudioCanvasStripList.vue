<template>
  <div ref="rootEl" class="studio-canvas-strips">
    <div v-if="activeTasks.length" class="studio-canvas-strips__tasks">
      <div class="studio-canvas-strips__tasks-row">
        <ActiveTaskCard
          v-for="task in activeTasks"
          :key="task.id"
          :task="task"
          media="image"
        />
      </div>
    </div>

    <div v-if="loading && !sessions.length" class="studio-canvas-strips__empty">
      {{ $t('common.loading') }}
    </div>

    <div v-else-if="!filteredSessions.length" class="studio-canvas-strips__empty">
      <p class="studio-canvas-strips__empty-title">
        {{ searchText ? $t('canvas.stripSearchEmpty') : $t('canvas.stripEmpty') }}
      </p>
      <p v-if="!searchText" class="studio-canvas-strips__empty-hint">
        {{ $t('canvas.stripEmptyHint') }}
      </p>
    </div>

    <ul v-else class="studio-canvas-strips__list" role="list">
      <li
        v-for="session in filteredSessions"
        :key="session.id"
        class="studio-canvas-strip"
        :data-session-id="session.id"
      >
        <!-- Top: name + meta | actions -->
        <div class="studio-canvas-strip__header">
          <button
            type="button"
            class="studio-canvas-strip__identity"
            @click="$emit('open', session.id)"
          >
            <span class="studio-canvas-strip__title">{{ sessionDisplayTitle(session) }}</span>
            <span class="studio-canvas-strip__sub">
              {{ $t('canvas.stripAssetCount', { count: session.item_count || 0 }) }}
              <template v-if="relativeLabel(session.updated_at)">
                <span class="studio-canvas-strip__dot" aria-hidden="true">·</span>
                {{ relativeLabel(session.updated_at) }}
              </template>
            </span>
          </button>

          <div class="studio-canvas-strip__actions">
            <DqButton
              type="primary"
              size="sm"
              @click.stop="$emit('create', session.id)"
            >
              {{ $t('canvas.stripCreate') }}
            </DqButton>
            <DqDropdown
              trigger="click"
              size="sm"
              @command="(cmd: string) => onMenu(session.id, cmd)"
            >
              <DqIconButton
                type="text"
                size="sm"
                :label="$t('canvas.moreMenu')"
                @click.stop
              >
                <DqIcon :size="16"><MoreFilled /></DqIcon>
              </DqIconButton>
              <template #dropdown>
                <DqDropdownMenu>
                  <DqDropdownItem command="rename">{{ $t('canvas.renameSession') }}</DqDropdownItem>
                  <DqDropdownItem
                    command="delete"
                    :disabled="sessions.length <= 1"
                    divided
                  >
                    {{ $t('canvas.deleteSession') }}
                  </DqDropdownItem>
                </DqDropdownMenu>
              </template>
            </DqDropdown>
          </div>
        </div>

        <!-- Bottom: thumbs open preview; empty / +N / title enter canvas detail -->
        <div class="studio-canvas-strip__media">
          <template v-if="previewIds(session).length === 1">
            <div class="studio-canvas-strip__thumbs studio-canvas-strip__thumbs--solo">
              <button
                type="button"
                class="studio-canvas-strip__thumb-wrap"
                :aria-label="$t('canvas.stripPreviewAsset')"
                @click.stop="$emit('preview', { sessionId: session.id, assetId: previewIds(session)[0] })"
              >
                <img
                  class="studio-canvas-strip__thumb studio-canvas-strip__thumb--contain"
                  :src="thumbUrl(previewIds(session)[0])"
                  alt=""
                  loading="lazy"
                  draggable="false"
                />
              </button>
            </div>
          </template>
          <template v-else-if="previewIds(session).length">
            <div
              class="studio-canvas-strip__thumbs"
              :class="{ 'studio-canvas-strip__thumbs--fill': fillsRow(session) }"
            >
              <button
                v-for="(id, i) in previewIds(session)"
                :key="id"
                type="button"
                class="studio-canvas-strip__thumb-wrap"
                :aria-label="$t('canvas.stripPreviewAsset')"
                @click.stop="onThumbClick(session, id, i)"
              >
                <img
                  class="studio-canvas-strip__thumb"
                  :src="thumbUrl(id)"
                  alt=""
                  loading="lazy"
                  draggable="false"
                />
                <span
                  v-if="i === previewIds(session).length - 1 && overflowCount(session) > 0"
                  class="studio-canvas-strip__more"
                >
                  +{{ overflowCount(session) }}
                </span>
              </button>
            </div>
          </template>
          <button
            v-else
            type="button"
            class="studio-canvas-strip__media-empty"
            @click="$emit('open', session.id)"
          >
            <DqIcon :size="18"><Picture /></DqIcon>
            <span>{{ $t('canvas.stripEmptyCanvas') }}</span>
          </button>
        </div>
      </li>
    </ul>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { MoreFilled, Picture } from '@danqing/dq-shell';
import type { CanvasSessionSummary, Task } from '@/types';
import ActiveTaskCard from '@/components/studio/ActiveTaskCard.vue';

const STRIP_THUMB_PX = 140;
const STRIP_THUMB_GAP = 2;

const props = defineProps<{
  sessions: CanvasSessionSummary[];
  searchText?: string;
  /** Full asset-id membership per session (board + shelf). */
  sessionAssetIds?: Record<string, string[]>;
  loading?: boolean;
  activeTasks?: Task[];
  galleryByAssetId?: Record<
    string,
    { prompt?: string; model?: string; title?: string; name?: string; created_at?: string }
  >;
}>();

const emit = defineEmits<{
  (e: 'open', sessionId: string): void;
  (e: 'preview', payload: { sessionId: string; assetId: string }): void;
  (e: 'create', sessionId: string): void;
  (e: 'rename', sessionId: string): void;
  (e: 'delete', sessionId: string): void;
}>();

const { t: $t } = useI18n();
const activeTasks = computed(() => props.activeTasks || []);
const rootEl = ref<HTMLElement | null>(null);
const stripInnerWidth = ref(0);
let stripRo: ResizeObserver | null = null;

/** How many square thumbs fit in the media row. */
const visibleSlotCount = computed(() => {
  const w = stripInnerWidth.value;
  if (w <= 0) return 6;
  const cell = STRIP_THUMB_PX + STRIP_THUMB_GAP;
  return Math.max(1, Math.floor((w + STRIP_THUMB_GAP) / cell));
});

function searchQuery(): string {
  return (props.searchText || '').trim().toLowerCase();
}

function sortedAssetIds(session: CanvasSessionSummary): string[] {
  const ids = sessionAssetIdList(session);
  if (ids.length <= 1) return ids;
  return [...ids].sort((a, b) => {
    const ta = props.galleryByAssetId?.[a]?.created_at || '';
    const tb = props.galleryByAssetId?.[b]?.created_at || '';
    const c = String(tb).localeCompare(String(ta));
    if (c !== 0) return c;
    return String(b).localeCompare(String(a));
  });
}

/** When searching assets, strip thumbs show matches only — not the whole canvas. */
function stripSourceIds(session: CanvasSessionSummary): string[] {
  const q = searchQuery();
  const all = sortedAssetIds(session);
  if (!q) return all;
  const matched = all.filter((id) => assetMatchesQuery(id, q));
  return matched.length ? matched : all;
}

function previewIds(session: CanvasSessionSummary): string[] {
  const ids = stripSourceIds(session);
  if (ids.length <= 1) return ids;
  return ids.slice(0, visibleSlotCount.value);
}

function fillsRow(session: CanvasSessionSummary): boolean {
  return previewIds(session).length >= visibleSlotCount.value;
}

function overflowCount(session: CanvasSessionSummary): number {
  const source = stripSourceIds(session);
  const total = searchQuery()
    ? source.length
    : Math.max(sessionAssetIdList(session).length, session.item_count || 0);
  return Math.max(0, total - previewIds(session).length);
}

function measureStripWidth() {
  const el = rootEl.value?.querySelector(
    '.studio-canvas-strip__media',
  ) as HTMLElement | null;
  const w = el?.clientWidth || rootEl.value?.clientWidth || 0;
  // Account for strip card horizontal padding when measuring from root.
  stripInnerWidth.value = el ? w : Math.max(0, w - 28);
}

onMounted(() => {
  void nextTick(measureStripWidth);
  if (typeof ResizeObserver === 'undefined') return;
  stripRo = new ResizeObserver(() => measureStripWidth());
  if (rootEl.value) stripRo.observe(rootEl.value);
});

onBeforeUnmount(() => {
  stripRo?.disconnect();
  stripRo = null;
});

/** Last tile with +N enters canvas; otherwise open asset preview. */
function onThumbClick(session: CanvasSessionSummary, assetId: string, index: number) {
  const ids = previewIds(session);
  if (index === ids.length - 1 && overflowCount(session) > 0) {
    emit('open', session.id);
    return;
  }
  emit('preview', { sessionId: session.id, assetId });
}

function thumbUrl(assetId: string): string {
  return `/api/assets/${encodeURIComponent(assetId)}/thumbnail`;
}

function relativeLabel(iso: string): string {
  if (!iso) return '';
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return '';
  const diff = Date.now() - t;
  const min = Math.floor(diff / 60000);
  if (min < 1) return $t('canvas.stripUpdatedJustNow');
  if (min < 60) return $t('canvas.stripUpdatedMinutes', { n: min });
  const hr = Math.floor(min / 60);
  if (hr < 48) return $t('canvas.stripUpdatedHours', { n: hr });
  const day = Math.floor(hr / 24);
  return $t('canvas.stripUpdatedDays', { n: day });
}

function sessionAssetIdList(session: CanvasSessionSummary): string[] {
  const full = props.sessionAssetIds?.[session.id];
  if (full?.length) return full;
  return session.preview_asset_ids || [];
}

function assetMatchesQuery(assetId: string, q: string): boolean {
  const meta = props.galleryByAssetId?.[assetId];
  if (!meta) return false;
  const hay = [meta.prompt, meta.model, meta.title, meta.name]
    .filter(Boolean)
    .join(' ')
    .toLowerCase();
  return hay.includes(q);
}

function isDefaultCanvasTitle(title: string): boolean {
  const t = (title || '').trim().toLowerCase();
  return t === 'default canvas' || t === '默认画布';
}

function sessionDisplayTitle(session: CanvasSessionSummary): string {
  if (isDefaultCanvasTitle(session.title || '')) {
    return $t('canvas.defaultCanvasTitle');
  }
  return session.title || '';
}

/** Stable list order: named default first, then oldest→newest. */
function sortSessionsStable(list: CanvasSessionSummary[]): CanvasSessionSummary[] {
  return [...list].sort((a, b) => {
    const aDef = isDefaultCanvasTitle(a.title || '') ? 0 : 1;
    const bDef = isDefaultCanvasTitle(b.title || '') ? 0 : 1;
    if (aDef !== bDef) return aDef - bDef;
    const c = String(a.created_at || '').localeCompare(String(b.created_at || ''));
    if (c !== 0) return c;
    return String(a.id || '').localeCompare(String(b.id || ''));
  });
}

const filteredSessions = computed(() => {
  const q = (props.searchText || '').trim().toLowerCase();
  const list = sortSessionsStable(props.sessions);
  if (!q) return list;
  return list.filter((s) => {
    const title = `${s.title || ''} ${sessionDisplayTitle(s)}`.toLowerCase();
    if (title.includes(q)) return true;
    return sessionAssetIdList(s).some((id) => assetMatchesQuery(id, q));
  });
});

watch(
  () => filteredSessions.value.length,
  () => {
    void nextTick(measureStripWidth);
  },
);

function onMenu(sessionId: string, cmd: string) {
  if (cmd === 'rename') emit('rename', sessionId);
  else if (cmd === 'delete') emit('delete', sessionId);
}
</script>

<style scoped>
.studio-canvas-strips {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 8px 0 48px;
  width: 100%;
  min-width: 0;
  max-width: none;
  box-sizing: border-box;
}

.studio-canvas-strips__tasks-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.studio-canvas-strips__tasks-row :deep(.active-task-card) {
  width: 112px;
  flex: 0 0 112px;
}

.studio-canvas-strips__list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.studio-canvas-strips__empty {
  padding: 64px 20px;
  text-align: center;
}

.studio-canvas-strips__empty-title {
  margin: 0;
  font-size: var(--dq-font-size-body);
  font-weight: 600;
  color: var(--dq-label-secondary);
}

.studio-canvas-strips__empty-hint {
  margin: 8px 0 0;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
}

.studio-canvas-strip {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 12px 14px 14px;
  border: 1px solid var(--dq-border-subtle);
  border-radius: var(--dq-radius-group);
  background: var(--dq-surface);
  transition:
    border-color 150ms ease,
    background-color 150ms ease;
}

.studio-canvas-strip:hover,
.studio-canvas-strip:focus-within {
  border-color: var(--dq-border);
  background: color-mix(in srgb, var(--dq-surface) 92%, var(--dq-label) 4%);
}

.studio-canvas-strip__header {
  display: flex;
  align-items: center;
  gap: 12px;
  min-width: 0;
}

.studio-canvas-strip__identity {
  flex: 1 1 auto;
  min-width: 0;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 4px;
  margin: 0;
  padding: 0;
  border: 0;
  background: transparent;
  text-align: left;
  cursor: pointer;
  color: inherit;
}

.studio-canvas-strip__identity:hover .studio-canvas-strip__title,
.studio-canvas-strip__identity:focus-visible .studio-canvas-strip__title {
  color: var(--dq-accent);
}

.studio-canvas-strip__title {
  font-size: var(--dq-font-size-title);
  font-weight: 600;
  letter-spacing: -0.01em;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 100%;
  transition: color 140ms ease;
}

.studio-canvas-strip__sub {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
}

.studio-canvas-strip__dot {
  opacity: 0.55;
}

.studio-canvas-strip__actions {
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  gap: 4px;
}

.studio-canvas-strip__media {
  display: flex;
  flex-direction: column;
  width: 100%;
  min-width: 0;
  margin: 0;
  padding: 0;
  border: 0;
  border-radius: var(--dq-radius-control);
  background: var(--dq-surface-inset);
  overflow: hidden;
  text-align: left;
  color: inherit;
}

.studio-canvas-strip__thumbs {
  --strip-thumb-size: 140px;
  display: flex;
  width: 100%;
  min-width: 0;
  gap: 2px;
  height: var(--strip-thumb-size);
  justify-content: flex-start;
  align-items: stretch;
}

/* Enough assets to fill the row — equal columns, no trailing gap */
.studio-canvas-strip__thumbs--fill .studio-canvas-strip__thumb-wrap {
  flex: 1 1 0;
  width: auto;
  min-width: 0;
}

/* Solo: same fixed band height; image keeps aspect, left-aligned */
.studio-canvas-strip__thumbs--solo {
  height: var(--strip-thumb-size);
}

.studio-canvas-strip__thumb-wrap {
  position: relative;
  flex: 0 0 var(--strip-thumb-size);
  width: var(--strip-thumb-size);
  height: var(--strip-thumb-size);
  margin: 0;
  padding: 0;
  border: 0;
  border-radius: 0;
  overflow: hidden;
  background: var(--dq-surface-inset);
  cursor: pointer;
  color: inherit;
}

.studio-canvas-strip__thumb-wrap:focus-visible {
  outline: none;
  box-shadow: inset var(--dq-focus-ring);
  z-index: 1;
}

.studio-canvas-strip__thumbs--solo .studio-canvas-strip__thumb-wrap {
  flex: 1 1 auto;
  width: 100%;
  min-width: 0;
  height: var(--strip-thumb-size);
}

.studio-canvas-strip__thumb {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  object-position: center;
  display: block;
}

.studio-canvas-strip__thumb--contain {
  object-fit: contain;
  object-position: left center;
  background: transparent;
}

.studio-canvas-strip__more {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: color-mix(in srgb, var(--dq-overlay-card) 55%, transparent);
  color: var(--dq-label);
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  -webkit-backdrop-filter: blur(4px);
  backdrop-filter: blur(4px);
}

.studio-canvas-strip__media-empty {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  width: 100%;
  height: 72px;
  margin: 0;
  padding: 0;
  border: 0;
  background: transparent;
  cursor: pointer;
  color: var(--dq-label-tertiary);
  font-size: var(--dq-font-size-caption);
}

.studio-canvas-strip__media-empty:focus-visible {
  outline: none;
  box-shadow: inset var(--dq-focus-ring);
}

@media (prefers-reduced-motion: reduce) {
  .studio-canvas-strip,
  .studio-canvas-strip__title {
    transition: none;
  }
}
</style>
