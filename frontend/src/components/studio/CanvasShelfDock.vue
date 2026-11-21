<template>
  <div
    class="canvas-shelf-dock"
    :class="{ 'canvas-shelf-dock--collapsed': collapsed }"
  >
    <div class="canvas-shelf-dock__bar">
      <button
        type="button"
        class="canvas-shelf-dock__toggle"
        :aria-expanded="!collapsed"
        @click="collapsed = !collapsed"
      >
        <span class="canvas-shelf-dock__title">{{ $t('canvas.dockTitle') }}</span>
        <span class="canvas-shelf-dock__count">{{ allItems.length }}</span>
        <span class="canvas-shelf-dock__chevron" aria-hidden="true">{{
          collapsed ? '▲' : '▼'
        }}</span>
      </button>
      <div v-if="!collapsed" class="canvas-shelf-dock__tools">
        <DqInput
          v-model="query"
          size="sm"
          class="canvas-shelf-dock__search"
          :placeholder="$t('canvas.dockSearch')"
          clearable
        />
      </div>
    </div>

    <div v-if="!collapsed" class="canvas-shelf-dock__body">
      <div v-if="!dayGroups.length" class="canvas-shelf-dock__empty">
        {{ allItems.length ? $t('canvas.dockSearchEmpty') : $t('canvas.dockEmpty') }}
      </div>
      <div v-else class="canvas-shelf-dock__days">
        <section
          v-for="group in dayGroups"
          :key="group.key"
          class="canvas-shelf-dock__day"
        >
          <div class="canvas-shelf-dock__day-label">
            <span>{{ group.label }}</span>
            <span class="canvas-shelf-dock__day-count">{{ group.items.length }}</span>
          </div>
          <div class="canvas-shelf-dock__day-row">
            <div
              v-for="item in group.items"
              :key="item.path"
              class="canvas-shelf-dock__tile"
              :class="{
                'canvas-shelf-dock__tile--selected': item.path === selectedPath,
                'canvas-shelf-dock__tile--board': isOnBoard(item.path),
              }"
            >
              <button
                type="button"
                class="canvas-shelf-dock__hit"
                :title="item.title || item.name || item.path"
                :aria-label="item.title || item.name || item.path"
                @click="onTileClick(item)"
              >
                <img
                  class="canvas-shelf-dock__thumb"
                  :src="thumbUrl(item)"
                  alt=""
                  loading="lazy"
                  draggable="false"
                />
              </button>
              <div
                v-if="!isOnBoard(item.path)"
                class="canvas-shelf-dock__actions"
              >
                <ComposerIconTip :content="$t('canvas.dockToBoard')" placement="left">
                  <button
                    type="button"
                    class="canvas-shelf-dock__action"
                    :aria-label="$t('canvas.dockToBoard')"
                    @click.stop="$emit('promote', item.path)"
                  >
                    <DqIcon :size="14"><Upload /></DqIcon>
                  </button>
                </ComposerIconTip>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import { Upload } from '@danqing/dq-shell';
import type { GalleryItem } from '@/types';
import { galleryThumbnailUrl } from '@/utils/canvasAssets';
import ComposerIconTip from '@/components/studio/ComposerIconTip.vue';

const props = defineProps<{
  /** All canvas membership paths (board ∪ shelf). */
  paths: string[];
  /** Paths currently on the freeform board. */
  boardPaths?: string[];
  galleryItems: GalleryItem[];
  selectedPath?: string;
}>();

const emit = defineEmits<{
  (e: 'select', item: GalleryItem): void;
  (e: 'preview', item: GalleryItem): void;
  (e: 'promote', path: string): void;
}>();

function onTileClick(item: GalleryItem) {
  emit('select', item);
  emit('preview', item);
}

const { t: $t } = useI18n();
const collapsed = ref(false);
const query = ref('');

const boardSet = computed(() => new Set(props.boardPaths || []));

function isOnBoard(path: string): boolean {
  return boardSet.value.has(path);
}

const byPath = computed(() => {
  const map = new Map<string, GalleryItem>();
  for (const g of props.galleryItems) map.set(g.path, g);
  return map;
});

function resolveItem(path: string): GalleryItem {
  const hit = byPath.value.get(path);
  if (hit) return hit;
  return {
    path,
    name: path.startsWith('asset:') ? path.slice(6) : path,
    width: 0,
    height: 0,
    duration_seconds: null,
    created_at: '',
    title: '',
    prompt: '',
    model: '',
    thumbnail: path.startsWith('asset:')
      ? `/api/assets/${encodeURIComponent(path.slice(6))}/thumbnail`
      : '',
    metadata: {},
  };
}

const allItems = computed(() => {
  const seen = new Set<string>();
  const out: GalleryItem[] = [];
  for (const path of props.paths) {
    if (!path || seen.has(path)) continue;
    seen.add(path);
    out.push(resolveItem(path));
  }
  return out.sort((a, b) => {
    const c = String(b.created_at || '').localeCompare(String(a.created_at || ''));
    if (c !== 0) return c;
    return String(b.path || '').localeCompare(String(a.path || ''));
  });
});

const filteredItems = computed(() => {
  const q = query.value.trim().toLowerCase();
  if (!q) return allItems.value;
  return allItems.value.filter((it) => {
    const hay = [it.title, it.prompt, it.model, it.name].filter(Boolean).join(' ').toLowerCase();
    return hay.includes(q);
  });
});

function localDayKeyFromDate(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}

function localDayKey(iso: string): string {
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return 'unknown';
  return localDayKeyFromDate(new Date(t));
}

function todayKey(): string {
  return localDayKeyFromDate(new Date());
}

function yesterdayKey(): string {
  const d = new Date();
  d.setDate(d.getDate() - 1);
  return localDayKeyFromDate(d);
}

function dayLabel(key: string): string {
  if (key === 'unknown') return $t('canvas.dockDayUnknown');
  if (key === todayKey()) return $t('canvas.dockDayToday');
  if (key === yesterdayKey()) return $t('canvas.dockDayYesterday');
  return key;
}

type DayGroup = { key: string; label: string; items: GalleryItem[] };

const dayGroups = computed((): DayGroup[] => {
  const map = new Map<string, GalleryItem[]>();
  for (const item of filteredItems.value) {
    const key = localDayKey(item.created_at || '');
    const list = map.get(key);
    if (list) list.push(item);
    else map.set(key, [item]);
  }
  const keys = [...map.keys()].sort((a, b) => {
    if (a === 'unknown') return 1;
    if (b === 'unknown') return -1;
    return b.localeCompare(a);
  });
  return keys.map((key) => ({
    key,
    label: dayLabel(key),
    items: map.get(key) || [],
  }));
});

function thumbUrl(item: GalleryItem): string {
  return galleryThumbnailUrl(item) || item.thumbnail || '';
}
</script>

<style scoped>
.canvas-shelf-dock {
  flex: 0 0 auto;
  display: flex;
  flex-direction: column;
  border-top: 1px solid var(--dq-border-subtle);
  background: var(--dq-surface);
  min-height: 0;
  max-height: min(320px, 40vh);
}

.canvas-shelf-dock--collapsed {
  max-height: none;
}

.canvas-shelf-dock__bar {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 12px;
  min-height: 36px;
}

.canvas-shelf-dock__toggle {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin: 0;
  padding: 0;
  border: 0;
  background: transparent;
  color: inherit;
  cursor: pointer;
}

.canvas-shelf-dock__title {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  color: var(--dq-label-secondary);
}

.canvas-shelf-dock__count {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
  font-variant-numeric: tabular-nums;
}

.canvas-shelf-dock__chevron {
  font-size: 10px;
  color: var(--dq-label-tertiary);
}

.canvas-shelf-dock__tools {
  flex: 1 1 auto;
  display: flex;
  justify-content: flex-end;
  min-width: 0;
}

.canvas-shelf-dock__search {
  width: min(220px, 40vw);
}

.canvas-shelf-dock__body {
  padding: 0 12px 10px;
  overflow-y: auto;
  min-height: 0;
}

.canvas-shelf-dock__empty {
  padding: 16px 4px;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
}

.canvas-shelf-dock__days {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.canvas-shelf-dock__day-label {
  display: flex;
  align-items: baseline;
  gap: 8px;
  padding: 0 2px;
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  color: var(--dq-label-secondary);
}

.canvas-shelf-dock__day-count {
  font-weight: 500;
  color: var(--dq-label-tertiary);
  font-variant-numeric: tabular-nums;
}

.canvas-shelf-dock__day-row {
  display: flex;
  flex-wrap: nowrap;
  gap: 8px;
  overflow-x: auto;
  padding-bottom: 2px;
}

.canvas-shelf-dock__tile {
  position: relative;
  flex: 0 0 88px;
  width: 88px;
  height: 88px;
  margin: 0;
  padding: 0;
  border: 1px solid var(--dq-border-subtle);
  border-radius: var(--dq-radius-control);
  background: var(--dq-surface-inset);
  overflow: hidden;
}

.canvas-shelf-dock__tile--board {
  box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--dq-accent) 35%, transparent);
}

.canvas-shelf-dock__tile--selected,
.canvas-shelf-dock__tile:hover,
.canvas-shelf-dock__tile:focus-within {
  border-color: color-mix(in srgb, var(--dq-accent) 55%, var(--dq-border));
}

.canvas-shelf-dock__hit {
  display: block;
  width: 100%;
  height: 100%;
  margin: 0;
  padding: 0;
  border: 0;
  background: transparent;
  cursor: pointer;
  color: inherit;
}

.canvas-shelf-dock__hit:focus-visible {
  outline: none;
  box-shadow: inset var(--dq-focus-ring);
}

.canvas-shelf-dock__thumb {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

/* Top-right promote control */
.canvas-shelf-dock__actions {
  position: absolute;
  top: 4px;
  right: 4px;
  z-index: 2;
  opacity: 0;
  transform: translateY(-2px);
  transition:
    opacity 140ms ease,
    transform 140ms ease;
  pointer-events: none;
}

.canvas-shelf-dock__tile:hover .canvas-shelf-dock__actions,
.canvas-shelf-dock__tile:focus-within .canvas-shelf-dock__actions,
.canvas-shelf-dock__tile--selected .canvas-shelf-dock__actions {
  opacity: 1;
  transform: translateY(0);
  pointer-events: auto;
}

.canvas-shelf-dock__action {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  margin: 0;
  padding: 0;
  border: 0;
  border-radius: 6px;
  background: color-mix(in srgb, var(--dq-overlay-card) 82%, transparent);
  color: var(--dq-label);
  cursor: pointer;
  line-height: 0;
  box-shadow: 0 1px 2px color-mix(in srgb, #000 35%, transparent);
  backdrop-filter: blur(6px);
}

.canvas-shelf-dock__action:hover,
.canvas-shelf-dock__action:focus-visible {
  background: var(--dq-accent);
  color: var(--dq-accent-contrast, #fff);
}

.canvas-shelf-dock__action:focus-visible {
  outline: none;
  box-shadow: var(--dq-focus-ring);
}

@media (prefers-reduced-motion: reduce) {
  .canvas-shelf-dock__actions {
    transition: none;
  }
}
</style>
