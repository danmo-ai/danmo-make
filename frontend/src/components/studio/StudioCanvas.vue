<template>
  <div class="studio-canvas">
    <!-- Group navigation (only when browsing inside a project group) -->
    <div v-if="groupMode && selectedGroupId" class="studio-canvas__group-bar">
      <DqButton
        type="text"
        size="sm"
        class="studio-canvas__exit-group-btn"
        @click="$emit('exit-group')"
      >
        {{ $t('gallery.exitGroup') }}
      </DqButton>
    </div>

    <!-- Empty state -->
    <div v-if="displayItems.length === 0 && !loading" class="studio-canvas__empty">
      <DqEmpty :description="emptyMessage" />
      <DqButton v-if="!hasActiveFilters" type="primary" @click="$emit('open-composer')">
        {{ $t('gallery.emptyComposeCta') }}
      </DqButton>
      <DqButton v-if="hasActiveFilters" type="primary" @click="$emit('reset-filters')">
        {{ $t('gallery.clearFilters') }}
      </DqButton>
    </div>

    <!-- Active tasks (generating placeholders) -->
    <div v-if="activeTasks.length > 0" class="studio-canvas__section studio-canvas__section--active">
      <div class="studio-canvas__active-layout">
        <div class="studio-canvas__active-heading">
          <div class="studio-canvas__active-heading-main">
            <span class="studio-canvas__section-dot studio-canvas__section-dot--active" />
            {{ $tt('studio.running') }}
            <span v-if="activeProgressHint" class="studio-canvas__active-hint">
              {{ activeProgressHint }}
            </span>
          </div>
          <DqButton
            v-if="logTaskId"
            type="text"
            size="sm"
            class="studio-canvas__logs-btn"
            @click="openTaskLogs"
          >
            {{ $t('studio.viewLogs') }}
          </DqButton>
        </div>
        <div class="studio-canvas__active-cards">
          <ActiveTaskCard
            v-for="task in activeTasks"
            :key="task.id"
            :task="task"
            :media="media"
          />
        </div>
      </div>
    </div>

    <!-- Recent task logs (after completion / failure) -->
    <div
      v-if="activeTasks.length === 0 && logTaskId && hasPersistedLogs"
      class="studio-canvas__section studio-canvas__section--log-recall"
    >
      <DqButton type="text" size="sm" class="studio-canvas__logs-btn" @click="openTaskLogs">
        {{ $t('studio.viewLogs') }}
      </DqButton>
    </div>

    <GenTaskLogDialog
      v-model:open="showTaskLogs"
      :task-id="logTaskId"
    />

    <!-- Gallery grid: project groups and ungrouped assets share one timeline -->
    <div
      v-for="section in canvasTimeSections"
      :key="section.label"
      class="studio-canvas__section"
    >
      <div class="studio-canvas__section-header">
        {{ section.label }}
      </div>
      <div class="studio-canvas__grid">
        <template v-for="cell in section.cells" :key="cellKey(cell)">
          <StudioGroupCard
            v-if="cell.kind === 'group'"
            :group="cell.group"
            @click="$emit('enter-group', cell.group.id)"
          />
          <StudioCard
            v-else
            :item="cell.item"
            :media="media"
            :selection-mode="selectionMode"
            :selected="isItemSelected(cell.item)"
            :gallery-canvas-mode="media === 'image' || media === 'video' || media === 'audio'"
            @click="handleCardClick(cell.item, $event)"
            @toggle-select="$emit('toggle-select', cell.item)"
            @action="$emit('card-action', $event)"
          />
        </template>
      </div>
    </div>

    <!-- Loading indicator -->
    <div v-if="loading" class="studio-canvas__loading">
      <DqIcon class="is-loading" size="24"><Loading /></DqIcon>
    </div>

    <!-- Load more -->
    <div v-else-if="hasMore && displayItems.length > 0" class="studio-canvas__load-more">
      <DqButton size="sm" @click="$emit('load-more')">
        {{ $t('common.loadMore') }}
      </DqButton>
    </div>

    <!-- End of list -->
    <div v-if="!hasMore && displayItems.length > 0" class="studio-canvas__end">
      {{ $t('gallery.noMore') }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { Loading } from '@danqing/dq-shell';
import { $tt } from '@/utils/i18n';
import { useTasksStore } from '@/stores/tasks';
import {
  buildLogDisplayItems,
  latestProgressItem,
} from '@/utils/genTaskLog';
import type { GalleryGroup, GalleryItem, Task } from '@/types';
import StudioCard from './StudioCard.vue';
import StudioGroupCard from './StudioGroupCard.vue';
import ActiveTaskCard from './ActiveTaskCard.vue';
import GenTaskLogDialog from './GenTaskLogDialog.vue';

const props = defineProps<{
  items: GalleryItem[];
  groups?: GalleryGroup[];
  activeTasks: Task[];
  loading: boolean;
  hasMore: boolean;
  media: 'image' | 'video' | 'audio';
  groupMode?: boolean;
  selectedGroupId?: string | null;
  hasActiveFilters?: boolean;
  selectionMode?: boolean;
  selectedPaths?: Set<string>;
  allSelected?: boolean;
}>();

const emit = defineEmits<{
  (e: 'select', item: GalleryItem): void;
  (e: 'card-action', payload: { action: string; item: GalleryItem }): void;
  (e: 'reset-filters'): void;
  (e: 'load-more'): void;
  (e: 'toggle-select', item: GalleryItem): void;
  (e: 'select-all'): void;
  (e: 'batch-delete'): void;
  (e: 'clear-selection'): void;
  (e: 'open-composer'): void;
  (e: 'enter-group', groupId: string): void;
  (e: 'exit-group'): void;
}>();

const { t: $t } = useI18n();
const tasksStore = useTasksStore();
const showTaskLogs = ref(false);
/** Persists after task leaves queue so logs stay viewable on failure. */
const logTaskId = ref<string | null>(null);

const isGroupView = computed(() => props.groupMode && !props.selectedGroupId);
const displayItems = computed(() => (isGroupView.value ? [...(props.groups || []), ...props.items] : props.items));

const primaryRunningTaskId = computed(() => {
  const running = props.activeTasks.find((task) => task.status === 'running');
  if (running?.id) return running.id;
  const first = props.activeTasks[0];
  return first?.id || null;
});

const hasPersistedLogs = computed(() => {
  const id = logTaskId.value;
  if (!id) return false;
  return (tasksStore.taskLogs[id]?.length ?? 0) > 0;
});

watch(primaryRunningTaskId, (id) => {
  if (id) {
    logTaskId.value = id;
  }
});

watch(
  () => {
    const id = logTaskId.value;
    if (!id) return '';
    const logs = tasksStore.taskLogs[id] || [];
    const last = logs[logs.length - 1];
    if (!last) return '';
    return `${logs.length}:${last.level}:${last.message}`;
  },
  (signature, prevSignature) => {
    if (!signature || signature === prevSignature) return;
    const id = logTaskId.value;
    if (!id) return;
    const logs = tasksStore.taskLogs[id] || [];
    const last = logs[logs.length - 1];
    if (last?.level === 'error') {
      showTaskLogs.value = true;
    }
  },
);

function openTaskLogs() {
  const id = primaryRunningTaskId.value || logTaskId.value;
  if (!id) return;
  logTaskId.value = id;
  showTaskLogs.value = true;
}

const activeProgressHint = computed(() => {
  const id = primaryRunningTaskId.value;
  if (!id) return '';
  const logs = tasksStore.taskLogs[id] || [];
  const latest = latestProgressItem(buildLogDisplayItems(logs, false));
  return latest?.title || '';
});

const emptyMessage = computed(() => {
  if (props.hasActiveFilters) {
    return $t('gallery.emptyFiltered');
  }
  if (isGroupView.value) {
    return $t('gallery.empty');
  }
  return $t('gallery.empty');
});

interface GalleryGridCell {
  kind: 'group';
  group: GalleryGroup;
  sortAt: number;
}

interface GalleryItemCell {
  kind: 'item';
  item: GalleryItem;
  sortAt: number;
}

type CanvasGridCell = GalleryGridCell | GalleryItemCell;

interface CanvasTimeSection {
  label: string;
  cells: CanvasGridCell[];
}

function parseSortTime(iso: string | undefined): number {
  if (!iso) return 0;
  const t = new Date(iso).getTime();
  return Number.isFinite(t) ? t : 0;
}

function groupActivityTime(group: GalleryGroup): number {
  let best = parseSortTime(group.updated_at || group.created_at);
  for (const preview of group.preview_assets || []) {
    best = Math.max(best, parseSortTime(preview.created_at));
  }
  return best;
}

function bucketForTime(
  sortAt: number,
  today: Date,
  yesterday: Date,
  weekAgo: Date,
  monthAgo: Date,
): 'today' | 'yesterday' | 'week' | 'month' | 'earlier' {
  const date = new Date(sortAt);
  if (date >= today) return 'today';
  if (date >= yesterday) return 'yesterday';
  if (date >= weekAgo) return 'week';
  if (date >= monthAgo) return 'month';
  return 'earlier';
}

const canvasTimeSections = computed((): CanvasTimeSection[] => {
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today.getTime() - 24 * 60 * 60 * 1000);
  const weekAgo = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);
  const monthAgo = new Date(today.getTime() - 30 * 24 * 60 * 60 * 1000);

  const buckets: Record<string, CanvasGridCell[]> = {
    today: [],
    yesterday: [],
    week: [],
    month: [],
    earlier: [],
  };

  const cells: CanvasGridCell[] = [];

  if (isGroupView.value) {
    for (const group of props.groups || []) {
      cells.push({ kind: 'group', group, sortAt: groupActivityTime(group) });
    }
  }
  for (const item of props.items) {
    cells.push({ kind: 'item', item, sortAt: parseSortTime(item.created_at) });
  }

  cells.sort((a, b) => b.sortAt - a.sortAt);

  for (const cell of cells) {
    const bucket = bucketForTime(cell.sortAt, today, yesterday, weekAgo, monthAgo);
    buckets[bucket].push(cell);
  }

  const sections: CanvasTimeSection[] = [];
  if (buckets.today.length) sections.push({ label: $t('gallery.groupToday'), cells: buckets.today });
  if (buckets.yesterday.length) {
    sections.push({ label: $t('gallery.groupYesterday'), cells: buckets.yesterday });
  }
  if (buckets.week.length) sections.push({ label: $t('gallery.groupThisWeek'), cells: buckets.week });
  if (buckets.month.length) sections.push({ label: $t('gallery.groupThisMonth'), cells: buckets.month });
  if (buckets.earlier.length) sections.push({ label: $t('gallery.groupEarlier'), cells: buckets.earlier });
  return sections;
});

function cellKey(cell: CanvasGridCell): string {
  return cell.kind === 'group' ? `group:${cell.group.id}` : cell.item.path;
}

function isItemSelected(item: GalleryItem) {
  return props.selectedPaths?.has(item.path) ?? false;
}

function handleCardClick(item: GalleryItem, event: MouseEvent) {
  if (props.selectionMode) {
    emit('toggle-select', item);
    return;
  }
  emit('select', item);
}
</script>

<style scoped>
.studio-canvas {
  width: 100%;
}

.studio-canvas__group-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 16px;
  min-height: 28px;
}

.studio-canvas__group-hint {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
}

.studio-canvas__exit-group-btn {
  font-weight: 500;
}

.studio-canvas__empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 80px 20px;
  gap: 16px;
}

.studio-canvas__section {
  margin-bottom: 28px;
}

.studio-canvas__section-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  letter-spacing: var(--dq-tracking-wider);
  text-transform: uppercase;
  color: var(--dq-label-tertiary);
  margin-bottom: 14px;
  padding-left: 2px;
}

.studio-canvas__section-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--dq-accent);
  animation: studio-pulse 2s ease-in-out infinite;
}

.studio-canvas__section-dot--active {
  background: var(--dq-accent);
}

@keyframes studio-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}

.studio-canvas__active-layout {
  display: flex;
  flex-direction: column;
  gap: 14px;
  min-width: 0;
}

.studio-canvas__active-heading {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  min-height: 24px;
  padding-left: 2px;
}

.studio-canvas__active-heading-main {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  letter-spacing: var(--dq-tracking-wider);
  text-transform: uppercase;
  color: var(--dq-label-tertiary);
}

.studio-canvas__active-hint {
  font-size: var(--dq-font-size-caption);
  font-weight: 500;
  letter-spacing: normal;
  text-transform: none;
  color: var(--dq-label-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.studio-canvas__logs-btn {
  flex-shrink: 0;
  font-size: var(--dq-font-size-caption);
}

.studio-canvas__section--log-recall {
  display: flex;
  justify-content: flex-end;
  margin-bottom: 12px;
}

.studio-canvas__active-cards {
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
  max-width: 200px;
}

.studio-canvas__grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(168px, 1fr));
  gap: 14px 16px;
}

@media (min-width: 768px) {
  .studio-canvas__grid {
    grid-template-columns: repeat(auto-fill, minmax(188px, 1fr));
    gap: 18px 20px;
  }
}

@media (min-width: 1200px) {
  .studio-canvas__grid {
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  }
}

.studio-canvas__loading,
.studio-canvas__load-more {
  display: flex;
  justify-content: center;
  padding: 32px;
}

.studio-canvas__loading {
  color: var(--dq-label-secondary);
}

.studio-canvas__end {
  text-align: center;
  padding: 24px;
  font-size: var(--dq-font-size-body);
  color: var(--dq-label-tertiary);
}

.is-loading {
  animation: studio-spin 1s linear infinite;
}

@keyframes studio-spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
</style>
