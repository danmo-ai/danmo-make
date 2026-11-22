<template>
  <WorkspaceLayout class="tasks-page">
    <div class="tasks-page__main">
      <WorkspaceHeader
        :title="$t('studio.tasksPageTitle')"
        :description="$t('studio.tasksPageDescription')"
      >
        <template #icon><DqIcon><ListOrdered /></DqIcon></template>
        <template #title-meta>
          <span class="tasks-page__counts" aria-live="polite">
            <span>{{ runningCount }} {{ $t('studio.running') }}</span>
            <span>{{ queuedCount }} {{ $t('studio.queued') }}</span>
          </span>
        </template>
        <template #filters>
          <DqSegmented
            v-model="statusFilter"
            class="tasks-page__filter"
            size="sm"
            :options="statusFilterOptions"
          />
          <DqInput
            v-model="searchQuery"
            class="tasks-page__search"
            :placeholder="$t('studio.tasksSearchPlaceholder')"
          />
        </template>
        <template #actions>
          <span v-if="loadError" class="tasks-page__load-error" role="alert">{{ loadError }}</span>
          <DqIconButton
            type="text"
            size="sm"
            :label="$t('common.refresh')"
            :disabled="loading"
            @click="refresh()"
          >
            <DqIcon><Refresh /></DqIcon>
          </DqIconButton>
        </template>
      </WorkspaceHeader>

      <div class="tasks-page__layout">
        <section class="tasks-page__list" :aria-label="$t('studio.tasksPageTitle')">
          <div v-if="visibleTasks.length === 0" class="tasks-page__empty">
            <p class="tasks-page__empty-title">{{ $t('studio.queueEmpty') }}</p>
            <p class="tasks-page__empty-hint">{{ $t('studio.tasksEmptyHint') }}</p>
          </div>

          <ol v-else class="tasks-page__rows">
            <li
              v-for="task in visibleTasks"
              :key="task.id"
              class="tasks-page__row"
              :class="{
                'is-selected': task.id === selectedId,
                [`is-${task.status}`]: true,
              }"
            >
              <button
                type="button"
                class="tasks-page__row-main"
                :aria-current="task.id === selectedId ? 'true' : undefined"
                @click="selectTask(task.id)"
              >
                <span class="tasks-page__row-top">
                  <span class="tasks-page__status">
                    <span
                      class="dq-status-dot"
                      :class="`dq-status-dot--${statusDot(task.status)}`"
                      aria-hidden="true"
                    />
                    {{ statusLabel(task.status) }}
                  </span>
                  <DqTag
                    v-if="isHighPriority(task)"
                    type="warning"
                    effect="plain"
                    size="sm"
                  >
                    {{ $t('studio.queuePriorityHigh') }}
                  </DqTag>
                  <span class="tasks-page__kind">{{ taskOperationLabel(task) }}</span>
                  <time class="tasks-page__time">{{ formatTime(task) }}</time>
                </span>
                <span class="tasks-page__row-model">
                  {{ task.params?.model || $t('queue.unspecifiedModel') }}
                </span>
                <span class="tasks-page__row-prompt">{{ taskLabel(task) }}</span>
                <span class="tasks-page__row-foot">
                  <TaskIdBadge :task-id="task.id" compact />
                  <span v-if="isLive(task)" class="tasks-page__progress">
                    {{ Math.round((displayProgress(task) ?? 0) * 100) }}%
                  </span>
                </span>
              </button>

              <div class="tasks-page__row-actions">
                <DqButton
                  v-if="task.status === 'queued'"
                  type="text"
                  size="sm"
                  :disabled="busy === task.id || isHighPriority(task)"
                  @click="setPriority(task.id, 'high')"
                >
                  {{ $t('studio.queueSetHigh') }}
                </DqButton>
                <DqButton
                  v-if="task.status === 'queued'"
                  type="text"
                  size="sm"
                  :disabled="busy === task.id || !isHighPriority(task)"
                  @click="setPriority(task.id, 'normal')"
                >
                  {{ $t('studio.queueSetNormal') }}
                </DqButton>
                <DqIconButton
                  v-if="isLive(task)"
                  type="danger"
                  size="sm"
                  :label="$t('studio.cancelTask')"
                  :disabled="busy === task.id"
                  @click="requestCancel(task)"
                >
                  <DqIcon><Close /></DqIcon>
                </DqIconButton>
                <DqIconButton
                  v-else
                  type="danger"
                  size="sm"
                  :label="$t('studio.deleteTask')"
                  :disabled="busy === task.id"
                  @click="requestPurge(task)"
                >
                  <DqIcon><Delete /></DqIcon>
                </DqIconButton>
              </div>
            </li>
          </ol>
        </section>

        <section class="tasks-page__detail" :aria-label="$t('studio.tasksDetailTitle')">
          <div v-if="!selectedTask" class="tasks-page__detail-empty">
            <p class="tasks-page__empty-title">{{ $t('studio.tasksSelectPrompt') }}</p>
            <p class="tasks-page__empty-hint">{{ $t('studio.tasksSelectHint') }}</p>
          </div>

          <template v-else>
            <header class="tasks-page__detail-head">
              <div class="tasks-page__detail-title">
                <h3>{{ taskLabel(selectedTask) }}</h3>
                <div class="tasks-page__detail-meta">
                  <DqTag effect="plain" size="sm">{{ statusLabel(selectedTask.status) }}</DqTag>
                  <span>{{ taskOperationLabel(selectedTask) }}</span>
                  <span>{{ selectedTask.params?.model || $t('queue.unspecifiedModel') }}</span>
                  <TaskIdBadge :task-id="selectedTask.id" compact />
                </div>
              </div>
              <p v-if="selectedTask.error_message" class="tasks-page__detail-error">
                {{ selectedTask.error_message }}
              </p>
            </header>

            <section class="tasks-page__trace">
              <h4 class="tasks-page__section-title">{{ $t('studio.traceTitle') }}</h4>
              <div
                v-if="selectedPipelineNodes.length > 0"
                class="tasks-page__pipeline"
                :aria-label="$t('studio.pipelineTab')"
              >
                <TaskPipelineGraph
                  :nodes="selectedPipelineNodes"
                  :active-node="selectedPipelineActive"
                  compact
                />
              </div>
              <div
                v-if="selectedStepTitle && selectedTask.status === 'running'"
                class="tasks-page__step-progress"
                role="status"
              >
                <div class="tasks-page__step-progress-head">
                  <span class="dq-status-dot dq-status-dot--running" aria-hidden="true" />
                  <span>{{ selectedStepTitle }}</span>
                </div>
                <div
                  v-if="selectedStepPercent != null"
                  class="tasks-page__step-progress-bar"
                  role="progressbar"
                  :aria-valuenow="Math.round(selectedStepPercent)"
                  aria-valuemin="0"
                  aria-valuemax="100"
                >
                  <span
                    class="tasks-page__step-progress-fill"
                    :style="{ width: `${selectedStepPercent}%` }"
                  />
                </div>
              </div>
              <TaskTraceWaterfall
                :task-id="selectedTask.id"
                :spans="selectedTrace?.spans ?? []"
                :reason="selectedTrace?.reason ?? null"
                :error-message="selectedTraceError"
              />
            </section>
          </template>
        </section>
      </div>
    </div>
  </WorkspaceLayout>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { useRoute, useRouter } from 'vue-router';
import { Close, Delete, ListOrdered, Refresh } from '@danqing/dq-shell';
import WorkspaceLayout from '@/components/workspace/WorkspaceLayout.vue';
import WorkspaceHeader from '@/components/workspace/WorkspaceHeader.vue';
import TaskIdBadge from '@/components/studio/TaskIdBadge.vue';
import TaskTraceWaterfall from '@/components/studio/TaskTraceWaterfall.vue';
import TaskPipelineGraph from '@/components/studio/TaskPipelineGraph.vue';
import { useTasksStore } from '@/stores/tasks';
import { api } from '@/utils/api';
import { $tt } from '@/utils/i18n';
import { confirm, toast } from '@/utils/feedback';
import {
  formatStepProgressTitle,
  stepProgressFromSpans,
  stepProgressPercent,
} from '@/utils/traceProgress';
import type { Task, TraceStepProgress } from '@/types';

const route = useRoute();
const router = useRouter();
const tasksStore = useTasksStore();

interface TaskRow extends Task {
  error_message?: string;
  error?: string | null;
  created_at?: string;
  completed_at?: string | null;
}

const recentTasks = ref<TaskRow[]>([]);
const loading = ref(false);
const loadError = ref<string | null>(null);
const busy = ref<string | null>(null);
const statusFilter = ref('all');
const searchQuery = ref('');

const statusFilterOptions = computed(() => [
  { label: $tt('studio.tasksFilterAll'), value: 'all' },
  { label: $tt('studio.running'), value: 'active' },
  { label: $tt('studio.queued'), value: 'queued' },
  { label: $tt('studio.tasksFilterFinished'), value: 'finished' },
]);

const liveTasks = computed<TaskRow[]>(() => {
  const withLive = (task: Task): TaskRow => {
    const live = tasksStore.liveTaskProgress[task.id];
    return live ? { ...task, ...live } : task;
  };
  return [
    ...tasksStore.queueState.running.map((task) => ({ ...withLive(task), status: 'running' })),
    ...tasksStore.queueState.queued.map((task) => ({ ...withLive(task), status: 'queued' })),
  ];
});

const runningCount = computed(() => tasksStore.queueState.running.length);
const queuedCount = computed(() => tasksStore.queueState.queued.length);

const allTasks = computed<TaskRow[]>(() => {
  const liveIds = new Set(liveTasks.value.map((task) => task.id));
  return [...liveTasks.value, ...recentTasks.value.filter((task) => !liveIds.has(task.id))];
});

const visibleTasks = computed(() => {
  const query = searchQuery.value.trim().toLowerCase();
  return allTasks.value.filter((task) => {
    if (!matchesStatusFilter(task)) return false;
    if (!query) return true;
    const haystack = [
      task.id,
      task.kind,
      task.params?.model,
      task.params?.title,
      task.params?.prompt,
    ]
      .map((value) => String(value ?? '').toLowerCase())
      .join(' ');
    return haystack.includes(query);
  });
});

const selectedId = computed(() => {
  const fromRoute = route.query.id;
  return typeof fromRoute === 'string' ? fromRoute : '';
});

const selectedTask = computed(
  () => allTasks.value.find((task) => task.id === selectedId.value) ?? null,
);

const selectedTrace = computed(() =>
  selectedId.value ? tasksStore.taskTraces[selectedId.value] ?? null : null,
);

const selectedTraceError = computed(() =>
  selectedId.value ? tasksStore.taskTraceErrors[selectedId.value] ?? null : null,
);

const selectedPipeline = computed(() =>
  selectedId.value ? tasksStore.taskPipelineGraphs[selectedId.value] ?? null : null,
);

const selectedPipelineNodes = computed(() => selectedPipeline.value?.nodes ?? []);
const selectedPipelineActive = computed(() => selectedPipeline.value?.active_node ?? null);

const selectedStepTip = computed<TraceStepProgress | null>(() => {
  const id = selectedId.value;
  if (!id) return null;
  const live = tasksStore.liveTaskProgress[id];
  if (live?.step != null && live?.total != null && Number(live.total) > 0) {
    return {
      phase: String(live.phase || live.progressMessage || ''),
      step: Number(live.step),
      total: Number(live.total),
    };
  }
  return (
    selectedPipeline.value?.step_progress
    ?? selectedTrace.value?.step_progress
    ?? stepProgressFromSpans(selectedTrace.value?.spans)
    ?? null
  );
});

const selectedStepTitle = computed(() => {
  const fromTip = formatStepProgressTitle(selectedStepTip.value);
  if (fromTip) return fromTip;
  const id = selectedId.value;
  if (!id) return null;
  const live = tasksStore.liveTaskProgress[id];
  return live?.progressMessage ?? null;
});

const selectedStepPercent = computed(() => {
  const fromTip = stepProgressPercent(selectedStepTip.value);
  if (fromTip != null) return fromTip;
  const id = selectedId.value;
  if (!id) return null;
  const live = tasksStore.liveTaskProgress[id]?.progress;
  if (typeof live === 'number') return Math.min(100, Math.max(0, live * 100));
  const graph = selectedPipeline.value?.progress;
  if (typeof graph === 'number') return Math.min(100, Math.max(0, graph * 100));
  return null;
});

function matchesStatusFilter(task: TaskRow): boolean {
  switch (statusFilter.value) {
    case 'active':
      return task.status === 'running';
    case 'queued':
      return task.status === 'queued';
    case 'finished':
      return task.status === 'completed' || task.status === 'failed' || task.status === 'cancelled';
    default:
      return true;
  }
}

function isLive(task: TaskRow): boolean {
  return task.status === 'running' || task.status === 'queued';
}

function isHighPriority(task: TaskRow): boolean {
  return (task.priority ?? 100) <= 50;
}

function displayProgress(task: TaskRow): number | null {
  const live = tasksStore.liveTaskProgress[task.id]?.progress;
  const value = live ?? task.progress;
  return typeof value === 'number' ? value : null;
}

function statusDot(status: string): string {
  if (status === 'running') return 'running';
  if (status === 'completed') return 'success';
  if (status === 'failed') return 'danger';
  if (status === 'cancelled') return 'warning';
  return 'pending';
}

function statusLabel(status: string): string {
  const key = `studio.taskStatus.${status}`;
  const label = $tt(key);
  return label && label !== key ? label : status;
}

function taskKindLabel(kind?: string): string {
  if (!kind) return '';
  const key = 'taskKind.' + String(kind).replace(/\./g, '_');
  const label = $tt(key);
  return label && label !== key ? label : kind;
}

function taskOperationLabel(task: TaskRow): string {
  const request = task.params?.request as Record<string, unknown> | undefined;
  const operation = String(request?.operation || request?.task || request?.mode || '').trim();
  if (operation && (task.kind === 'audio.edit' || task.kind === 'audio.lm')) {
    const key = `action.audio.${operation === 'cover-nofsq' ? 'cover' : operation}`;
    const label = $tt(key);
    if (label !== key) return label;
  }
  return taskKindLabel(task.kind);
}

function taskLabel(task: TaskRow): string {
  const title = String(task.params?.title || '').trim();
  const prompt = String(task.params?.prompt || '').trim();
  return truncate(title || prompt || task.id, 60);
}

function truncate(text: string, length: number): string {
  return text.length > length ? `${text.slice(0, length)}...` : text;
}

function formatTime(task: TaskRow): string {
  const raw = task.completed_at || task.created_at;
  if (!raw) return '';
  const parsed = new Date(raw.replace(' ', 'T'));
  if (Number.isNaN(parsed.getTime())) return raw;
  return parsed.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
}

function selectTask(taskId: string): void {
  if (!taskId || taskId === selectedId.value) return;
  void router.replace({ query: { ...route.query, id: taskId } });
}

async function refresh(): Promise<void> {
  loading.value = true;
  loadError.value = null;
  try {
    await tasksStore.pollQueueOnce();
    const response = (await api.gen.listMediaTasks({ limit: 100 })) as {
      tasks?: TaskRow[];
    };
    recentTasks.value = response?.tasks ?? [];
  } catch (e: unknown) {
    loadError.value = errorMessage(e);
  } finally {
    loading.value = false;
  }
}

async function setPriority(taskId: string, priority: 'high' | 'normal'): Promise<void> {
  busy.value = taskId;
  try {
    await api.gen.patchMediaTaskPriority(taskId, { priority });
    await tasksStore.pollQueueOnce();
    toast.success($tt('studio.priorityUpdated'));
  } catch (e: unknown) {
    toast.error($tt('studio.error', { msg: errorMessage(e) }));
  } finally {
    busy.value = null;
  }
}

function requestCancel(task: TaskRow): void {
  confirm(
    $tt('studio.cancelTaskConfirm', { id: task.id }),
    $tt('studio.cancelTask'),
    {
      confirmButtonText: $tt('studio.cancelTask'),
      cancelButtonText: $tt('common.cancel'),
      type: 'warning',
    },
  )
    .then(() => cancelTask(task.id))
    .catch(() => {});
}

function requestPurge(task: TaskRow): void {
  confirm(
    $tt('studio.deleteTaskConfirm', { id: task.id }),
    $tt('studio.deleteTask'),
    {
      confirmButtonText: $tt('studio.deleteTask'),
      cancelButtonText: $tt('common.cancel'),
      type: 'warning',
    },
  )
    .then(() => purgeTask(task.id))
    .catch(() => {});
}

async function cancelTask(taskId: string): Promise<void> {
  busy.value = taskId;
  try {
    const result = await api.gen.cancelMediaTask(taskId);
    if (!result || result.ok !== true) {
      throw new Error($tt('studio.cancelTaskFailed'));
    }
    await tasksStore.pollQueueOnce();
    toast.success($tt('studio.cancelled'));
  } catch (e: unknown) {
    toast.error($tt('studio.error', { msg: errorMessage(e) }));
  } finally {
    busy.value = null;
  }
}

async function purgeTask(taskId: string): Promise<void> {
  busy.value = taskId;
  try {
    await api.gen.purgeMediaTask(taskId);
    tasksStore.clearTaskTrace(taskId);
    recentTasks.value = recentTasks.value.filter((task) => task.id !== taskId);
    if (selectedId.value === taskId) {
      void router.replace({ query: { ...route.query, id: undefined } });
    }
    toast.success($tt('studio.taskDeleted'));
    await refresh();
  } catch (e: unknown) {
    toast.error($tt('studio.error', { msg: errorMessage(e) }));
  } finally {
    busy.value = null;
  }
}

function errorMessage(e: unknown): string {
  if (typeof e === 'object' && e !== null && 'response' in e) {
    const err = e as { response?: { data?: { detail?: string } } };
    const detail = err.response?.data?.detail;
    if (detail) return detail;
  }
  if (e instanceof Error) return e.message;
  return String(e);
}

watch(
  () => selectedId.value,
  (taskId) => {
    if (!taskId) return;
    void tasksStore.loadTaskTrace(taskId);
  },
  { immediate: true },
);

let refreshTimer: ReturnType<typeof setInterval> | null = null;

onMounted(async () => {
  tasksStore.ensureQueuePoller();
  await refresh();
  refreshTimer = setInterval(() => {
    void refresh();
    const id = selectedId.value;
    const task = selectedTask.value;
    // Poll spans while running so closed phases appear without a log stream.
    if (id && task?.status === 'running') {
      void tasksStore.loadTaskTrace(id);
    }
  }, 2000);
});

onBeforeUnmount(() => {
  tasksStore.releaseQueuePoller();
  if (refreshTimer) {
    clearInterval(refreshTimer);
    refreshTimer = null;
  }
});
</script>

<style scoped>
.tasks-page {
  height: 100%;
}

.tasks-page__main {
  display: flex;
  flex-direction: column;
  gap: 16px;
  min-height: 0;
  height: 100%;
}

.tasks-page__counts {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  margin-left: 8px;
  color: var(--dq-label-tertiary);
  font-size: var(--dq-font-size-caption);
  font-variant-numeric: tabular-nums;
  font-weight: 500;
}

.tasks-page__filter {
  flex: 0 1 auto;
  min-width: 0;
}

.tasks-page__search {
  flex: 1 1 180px;
  min-width: 140px;
  max-width: 280px;
}

.tasks-page__load-error {
  color: var(--dq-danger, #ff453a);
  font-size: var(--dq-font-size-caption);
}

.tasks-page__layout {
  display: grid;
  grid-template-columns: minmax(280px, 0.9fr) minmax(360px, 1.6fr);
  gap: 16px;
  min-height: 0;
  flex: 1;
}

.tasks-page__list {
  min-height: 0;
  overflow-y: auto;
  padding-right: 4px;
}

.tasks-page__rows {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.tasks-page__row {
  display: flex;
  align-items: stretch;
  gap: 6px;
  border: 1px solid var(--dq-border-subtle);
  border-radius: var(--dq-radius-control-sm);
  background: var(--dq-bg-base);
  transition: border-color 140ms ease, background-color 140ms ease;
}

.tasks-page__row:hover {
  border-color: var(--dq-border-strong);
}

.tasks-page__row.is-selected {
  border-color: color-mix(in srgb, var(--dq-accent) 55%, var(--dq-border-subtle));
  background: var(--dq-fill-tertiary);
}

.tasks-page__row.is-failed {
  border-color: color-mix(in srgb, var(--dq-danger, #ff453a) 32%, var(--dq-border-subtle));
}

.tasks-page__row-main {
  display: flex;
  flex: 1;
  flex-direction: column;
  gap: 3px;
  min-width: 0;
  padding: 9px 10px;
  border: 0;
  background: transparent;
  color: inherit;
  cursor: pointer;
  font: inherit;
  text-align: left;
}

.tasks-page__row-top {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  color: var(--dq-label-tertiary);
  font-size: var(--dq-font-size-caption);
}

.tasks-page__status {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  color: var(--dq-label-secondary);
  font-weight: 500;
}

.tasks-page__kind {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.tasks-page__time {
  margin-left: auto;
  font-variant-numeric: tabular-nums;
}

.tasks-page__row-model {
  overflow: hidden;
  color: var(--dq-label-secondary);
  font-size: var(--dq-font-size-caption);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.tasks-page__row-prompt {
  overflow: hidden;
  color: var(--dq-label-primary);
  font-size: var(--dq-font-size-body);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.tasks-page__row-foot {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--dq-label-tertiary);
  font-size: var(--dq-font-size-caption);
}

.tasks-page__progress {
  font-variant-numeric: tabular-nums;
}

.tasks-page__row-actions {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  justify-content: center;
  gap: 2px;
  padding: 6px 8px 6px 0;
}

.tasks-page__detail {
  display: flex;
  flex-direction: column;
  gap: 16px;
  min-width: 0;
  min-height: 0;
  overflow-y: auto;
  padding: 14px 16px;
  border: 1px solid var(--dq-border-subtle);
  border-radius: var(--dq-radius-control-sm);
  background: var(--dq-bg-base);
}

.tasks-page__detail-head {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.tasks-page__detail-title h3 {
  margin: 0 0 4px;
  color: var(--dq-label-primary);
  font-size: var(--dq-font-size-title-3, 15px);
  font-weight: 600;
  line-height: 1.35;
  word-break: break-word;
}

.tasks-page__detail-meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  color: var(--dq-label-tertiary);
  font-size: var(--dq-font-size-caption);
}

.tasks-page__detail-error {
  margin: 0;
  padding: 8px 10px;
  border-radius: var(--dq-radius-control-sm);
  color: var(--dq-danger, #ff453a);
  background: color-mix(in srgb, var(--dq-danger, #ff453a) 8%, transparent);
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
  word-break: break-word;
}

.tasks-page__section-title {
  margin: 0 0 8px;
  color: var(--dq-label-tertiary);
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  letter-spacing: var(--dq-tracking-wider);
  text-transform: uppercase;
}

.tasks-page__trace {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-width: 0;
}

.tasks-page__pipeline {
  padding: 8px 10px;
  border-radius: var(--dq-radius-control);
  background: var(--dq-surface-inset);
  border: 0.5px solid var(--dq-glass-border);
}

.tasks-page__step-progress {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 8px 12px;
  border-radius: var(--dq-radius-control);
  background: color-mix(in srgb, var(--dq-accent) 12%, transparent);
  border: 0.5px solid color-mix(in srgb, var(--dq-accent) 28%, transparent);
}

.tasks-page__step-progress-head {
  display: flex;
  align-items: center;
  gap: 8px;
  color: var(--dq-label-primary);
  font-size: var(--dq-font-size-body);
  font-weight: 500;
}

.tasks-page__step-progress-bar {
  height: 4px;
  overflow: hidden;
  border-radius: 999px;
  background: color-mix(in srgb, var(--dq-accent) 18%, transparent);
}

.tasks-page__step-progress-fill {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: var(--dq-accent);
  transition: width 160ms ease;
}

.tasks-page__empty,
.tasks-page__detail-empty {
  padding: 32px 16px;
  text-align: center;
}

.tasks-page__empty-title {
  margin: 0 0 4px;
  color: var(--dq-label-secondary);
  font-size: var(--dq-font-size-body);
  font-weight: 500;
}

.tasks-page__empty-hint {
  margin: 0;
  color: var(--dq-label-tertiary);
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
}

@media (max-width: 1080px) {
  .tasks-page__layout {
    grid-template-columns: 1fr;
  }
}
</style>
