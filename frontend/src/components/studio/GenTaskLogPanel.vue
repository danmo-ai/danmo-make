<template>
  <div
    class="gen-task-log-panel"
    :class="{
      'gen-task-log-panel--compact': compact,
      'gen-task-log-panel--side': side,
      'gen-task-log-panel--dialog': inDialog,
    }"
  >
    <header class="gen-task-log-panel__header">
      <div class="gen-task-log-panel__lead">
        <span v-if="!inDialog" class="gen-task-log-panel__title">{{ $t('studio.logs') }}</span>
        <span v-if="logs.length > 0" class="gen-task-log-panel__count">{{ logs.length }}</span>
      </div>
      <div class="gen-task-log-panel__actions">
        <DqButton
          type="text"
          size="sm"
          class="gen-task-log-panel__toggle"
          @click="showTechnical = !showTechnical"
        >
          {{ showTechnical ? $t('studio.logsDetailOff') : $t('studio.logsDetailOn') }}
        </DqButton>
        <DqButton
          v-if="canDiagnose"
          type="text"
          size="sm"
          class="gen-task-log-panel__toggle"
          :disabled="diagnosing"
          @click="runDiagnose"
        >
          {{ diagnosing ? $t('studio.pipelineDiagnosing') : $t('studio.pipelineDiagnose') }}
        </DqButton>
        <DqIconButton
          type="text"
          size="sm"
          :label="$t('studio.clearLogs')"
          :disabled="logs.length === 0"
          @click="clearLogs"
        >
          <DqIcon><Delete /></DqIcon>
        </DqIconButton>
      </div>
    </header>

    <TaskIdBadge
      v-if="taskId"
      :task-id="taskId"
      compact
    />

    <section
      v-if="hasPipeline || !activeProgress"
      class="gen-task-log-panel__pipeline"
      :aria-label="$t('studio.pipelineTab')"
    >
      <TaskPipelineGraph
        :nodes="pipelineNodes"
        :active-node="pipelineActive"
        compact
      />
    </section>

    <div
      v-if="activeProgress"
      class="gen-task-log-panel__progress"
      role="status"
    >
      <div class="gen-task-log-panel__progress-head">
        <span class="dq-status-dot dq-status-dot--running" aria-hidden="true" />
        <span class="gen-task-log-panel__progress-text">{{ activeProgress.title }}</span>
        <time class="gen-task-log-panel__progress-time">{{ activeProgress.time }}</time>
      </div>
      <div
        v-if="progressPercent != null"
        class="gen-task-log-panel__progress-bar"
        role="progressbar"
        :aria-valuenow="Math.round(progressPercent)"
        aria-valuemin="0"
        aria-valuemax="100"
      >
        <span
          class="gen-task-log-panel__progress-fill"
          :style="{ width: `${progressPercent}%` }"
        />
      </div>
    </div>

    <p v-if="diagnosis" class="gen-task-log-panel__diagnosis">{{ diagnosis }}</p>

    <div
      v-if="showLogBody"
      ref="logContainerRef"
      class="gen-task-log-panel__body"
    >
      <div v-if="displayItems.length === 0 && !activeProgress" class="gen-task-log-panel__empty">
        <p class="gen-task-log-panel__empty-title">{{ $t('studio.logsEmpty') }}</p>
        <p class="gen-task-log-panel__empty-hint">{{ $t('studio.logsEmptyHint') }}</p>
      </div>

      <template v-else-if="displayItems.length > 0">
        <p class="gen-task-log-panel__section-label">{{ $t('studio.logRecords') }}</p>
        <ol class="gen-task-log-panel__timeline">
          <li
            v-for="(item, rowIndex) in displayItems"
            :key="item.index"
            class="gen-task-log-panel__entry"
            :class="`gen-task-log-panel__entry--${item.kind}`"
          >
            <div class="gen-task-log-panel__rail" aria-hidden="true">
              <span class="gen-task-log-panel__dot" />
              <span
                v-if="rowIndex < displayItems.length - 1"
                class="gen-task-log-panel__stem"
              />
            </div>
            <div class="gen-task-log-panel__content">
              <div class="gen-task-log-panel__row-head">
                <span class="gen-task-log-panel__label">{{ item.title }}</span>
                <time
                  v-if="showEntryTime(item)"
                  class="gen-task-log-panel__time"
                >
                  {{ item.time }}
                </time>
              </div>
              <div v-if="item.chips?.length" class="gen-task-log-panel__chips">
                <span
                  v-for="chip in item.chips"
                  :key="chip.key"
                  class="gen-task-log-panel__chip"
                >
                  <span class="gen-task-log-panel__chip-label">{{ chip.label }}</span>
                  <span class="gen-task-log-panel__chip-value">{{ chip.value }}</span>
                </span>
              </div>
              <p
                v-if="item.detail"
                class="gen-task-log-panel__detail"
              >
                {{ item.detail }}
              </p>
            </div>
          </li>
        </ol>
      </template>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch, nextTick } from 'vue';
import { Delete } from '@danqing/dq-shell';
import TaskIdBadge from './TaskIdBadge.vue';
import TaskPipelineGraph from './TaskPipelineGraph.vue';
import { useTasksStore } from '@/stores/tasks';
import { api } from '@/utils/api';
import {
  buildLogDisplayItems,
  filterLogTimelineItems,
  latestProgressItem,
  resolveDisplayProgressPercent,
  type LogDisplayItem,
} from '@/utils/genTaskLog';

const props = defineProps<{
  taskId?: string | null;
  compact?: boolean;
  side?: boolean;
  /** 弹窗内展示：更高可视区域，不占作品流版面 */
  inDialog?: boolean;
}>();

const tasksStore = useTasksStore();
const logContainerRef = ref<HTMLElement | null>(null);
const showTechnical = ref(false);
const diagnosing = ref(false);
const diagnosis = ref<string | null>(null);

const pipelineGraph = computed(() => {
  const id = props.taskId;
  if (!id) return null;
  return tasksStore.taskPipelineGraphs[id] ?? null;
});

const pipelineNodes = computed(() => pipelineGraph.value?.nodes ?? []);
const pipelineActive = computed(() => pipelineGraph.value?.active_node ?? null);
const hasPipeline = computed(() => pipelineNodes.value.length > 0);
const canDiagnose = computed(() => Boolean(props.taskId));

const liveProgress = computed(() => {
  const id = props.taskId;
  if (!id) return null;
  return tasksStore.liveTaskProgress[id] ?? null;
});

async function loadPipelineGraph() {
  if (props.taskId) {
    await tasksStore.loadTaskPipelineGraph(props.taskId);
  }
}

async function runDiagnose() {
  if (!props.taskId || diagnosing.value) return;
  diagnosing.value = true;
  diagnosis.value = null;
  try {
    const res = await api.tasks.diagnose(props.taskId);
    diagnosis.value = res.summary;
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    diagnosis.value = msg;
  } finally {
    diagnosing.value = false;
  }
}

const logs = computed(() => {
  const id = props.taskId;
  if (!id) return [];
  return tasksStore.taskLogs[id] || [];
});

const rawDisplayItems = computed(() =>
  buildLogDisplayItems(logs.value, showTechnical.value),
);

const activeProgress = computed(() => latestProgressItem(rawDisplayItems.value));

const progressPercent = computed(() =>
  resolveDisplayProgressPercent(
    pipelineGraph.value?.progress,
    liveProgress.value,
    activeProgress.value?.title,
  ),
);

const displayItems = computed(() =>
  filterLogTimelineItems(
    rawDisplayItems.value,
    showTechnical.value,
    activeProgress.value?.index ?? null,
    { hidePipelineMilestones: hasPipeline.value && !showTechnical.value },
  ),
);

const showLogBody = computed(
  () => displayItems.value.length > 0 || (logs.value.length === 0 && !activeProgress.value),
);

function showEntryTime(item: LogDisplayItem): boolean {
  if (showTechnical.value) return true;
  return item.kind === 'milestone' || item.kind === 'error' || item.kind === 'warning';
}

function clearLogs() {
  if (props.taskId) {
    tasksStore.clearTaskLogs(props.taskId);
  }
}

watch(
  () => props.taskId,
  (id) => {
    diagnosis.value = null;
    if (id) {
      void loadPipelineGraph();
    }
  },
  { immediate: true },
);

watch(
  () => logs.value.length,
  () => {
    nextTick(() => {
      const container = logContainerRef.value;
      if (container) {
        container.scrollTop = container.scrollHeight;
      }
    });
  },
);
</script>

<style scoped>
.gen-task-log-panel {
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-width: 0;
}

.gen-task-log-panel__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  min-height: 28px;
}

.gen-task-log-panel__lead {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}

.gen-task-log-panel__title {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  letter-spacing: var(--dq-tracking-wider);
  text-transform: uppercase;
  color: var(--dq-label-tertiary);
  flex-shrink: 0;
}

.gen-task-log-panel__count {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  line-height: 1;
  padding: 3px 7px;
  border-radius: 999px;
  background: var(--dq-fill-on-glass);
  color: var(--dq-label-secondary);
}

.gen-task-log-panel__actions {
  display: flex;
  align-items: center;
  gap: 4px;
  flex-shrink: 0;
}

.gen-task-log-panel__toggle {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
}

.gen-task-log-panel__pipeline {
  padding: 8px 10px;
  border-radius: var(--dq-radius-control);
  background: var(--dq-surface-inset);
  border: 0.5px solid var(--dq-glass-border);
}

.gen-task-log-panel__progress {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 8px 12px;
  border-radius: var(--dq-radius-control);
  background: color-mix(in srgb, var(--dq-accent) 12%, transparent);
  border: 0.5px solid color-mix(in srgb, var(--dq-accent) 28%, transparent);
}

.gen-task-log-panel__progress-head {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}

.gen-task-log-panel__progress-text {
  flex: 1;
  min-width: 0;
  font-size: var(--dq-font-size-body);
  font-weight: 500;
  color: var(--dq-label-primary);
}

.gen-task-log-panel__progress-time {
  font-size: var(--dq-font-size-caption);
  font-variant-numeric: tabular-nums;
  color: var(--dq-label-tertiary);
  flex-shrink: 0;
}

.gen-task-log-panel__progress-bar {
  height: 4px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--dq-accent) 16%, transparent);
  overflow: hidden;
}

.gen-task-log-panel__progress-fill {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: var(--dq-accent);
  transition: width 0.25s ease;
}

.gen-task-log-panel__diagnosis {
  margin: 0;
  padding: 10px 12px;
  border-radius: var(--dq-radius-control);
  font-size: var(--dq-font-size-caption);
  line-height: 1.5;
  color: var(--dq-label-secondary);
  white-space: pre-wrap;
  background: var(--dq-fill-on-glass);
  border: 0.5px solid var(--dq-glass-border);
}

.gen-task-log-panel__body {
  flex: 1 1 auto;
  min-height: 0;
  max-height: 220px;
  overflow-y: auto;
  padding: 4px 2px 6px;
  border-radius: var(--dq-radius-group);
  background: var(--dq-surface-inset);
  border: 0.5px solid var(--dq-glass-border);
  scrollbar-gutter: stable;
}

.gen-task-log-panel--compact .gen-task-log-panel__body {
  max-height: 160px;
}

.gen-task-log-panel--side {
  flex: 1 1 auto;
  min-height: 0;
  gap: 12px;
}

.gen-task-log-panel--side .gen-task-log-panel__body {
  flex: 1 1 auto;
  min-height: 0;
  max-height: none;
}

.gen-task-log-panel--dialog .gen-task-log-panel__body {
  flex: 1 1 auto;
  min-height: min(320px, 50vh);
  max-height: min(480px, 62vh);
}

@media (max-width: 640px) {
  .gen-task-log-panel--side {
    flex: none;
    gap: 10px;
  }

  .gen-task-log-panel--side .gen-task-log-panel__body {
    flex: none;
    max-height: 180px;
  }

  .gen-task-log-panel--dialog .gen-task-log-panel__body {
    min-height: min(260px, 42vh);
    max-height: min(380px, 55vh);
  }
}

.gen-task-log-panel__section-label {
  margin: 0;
  padding: 8px 12px 0;
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  letter-spacing: var(--dq-tracking-wider);
  text-transform: uppercase;
  color: var(--dq-label-tertiary);
}

.gen-task-log-panel__empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 28px 16px;
  text-align: center;
}

.gen-task-log-panel__empty-title {
  margin: 0;
  font-size: var(--dq-font-size-body);
  font-weight: 500;
  color: var(--dq-label-secondary);
}

.gen-task-log-panel__empty-hint {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
  color: var(--dq-label-tertiary);
  max-width: 28em;
}

.gen-task-log-panel__timeline {
  list-style: none;
  margin: 0;
  padding: 8px 12px 12px;
}

.gen-task-log-panel__entry {
  display: grid;
  grid-template-columns: 18px minmax(0, 1fr);
  gap: 10px;
  padding-bottom: 14px;
}

.gen-task-log-panel__entry:last-child {
  padding-bottom: 0;
}

.gen-task-log-panel__rail {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding-top: 5px;
}

.gen-task-log-panel__dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--dq-label-quaternary, var(--dq-label-tertiary));
  flex-shrink: 0;
  z-index: 1;
}

.gen-task-log-panel__entry--milestone .gen-task-log-panel__dot {
  background: var(--dq-accent);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--dq-accent) 22%, transparent);
}

.gen-task-log-panel__entry--progress .gen-task-log-panel__dot {
  background: var(--dq-accent);
}

.gen-task-log-panel__entry--error .gen-task-log-panel__dot {
  background: var(--dq-danger);
}

.gen-task-log-panel__entry--warning .gen-task-log-panel__dot {
  background: var(--dq-warning);
}

.gen-task-log-panel__entry--technical .gen-task-log-panel__dot {
  width: 6px;
  height: 6px;
  margin-top: 1px;
  background: var(--dq-label-tertiary);
  opacity: 0.55;
}

.gen-task-log-panel__stem {
  position: absolute;
  top: 16px;
  bottom: -6px;
  width: 1px;
  background: var(--dq-separator, var(--dq-glass-border));
}

.gen-task-log-panel__content {
  min-width: 0;
}

.gen-task-log-panel__row-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 10px;
}

.gen-task-log-panel__label {
  flex: 1;
  min-width: 0;
  font-size: var(--dq-font-size-body);
  line-height: 1.4;
  font-weight: 500;
  color: var(--dq-label-primary);
  word-break: break-word;
}

.gen-task-log-panel__entry--technical .gen-task-log-panel__label {
  font-weight: 400;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
}

.gen-task-log-panel__entry--error .gen-task-log-panel__label {
  color: var(--dq-danger);
}

.gen-task-log-panel__entry--warning .gen-task-log-panel__label {
  color: var(--dq-warning);
}

.gen-task-log-panel__time {
  flex-shrink: 0;
  font-size: var(--dq-font-size-caption);
  font-variant-numeric: tabular-nums;
  color: var(--dq-label-tertiary);
  padding-top: 2px;
}

.gen-task-log-panel__chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 8px;
}

.gen-task-log-panel__chip {
  display: inline-flex;
  align-items: baseline;
  gap: 4px;
  padding: 4px 8px;
  border-radius: var(--dq-radius-control);
  background: var(--dq-fill-on-glass);
  border: 0.5px solid var(--dq-glass-border);
  font-size: var(--dq-font-size-caption);
  line-height: 1.3;
}

.gen-task-log-panel__chip-label {
  color: var(--dq-label-tertiary);
}

.gen-task-log-panel__chip-value {
  color: var(--dq-label-primary);
  font-variant-numeric: tabular-nums;
}

.gen-task-log-panel__detail {
  margin: 6px 0 0;
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
  color: var(--dq-label-tertiary);
  font-family: var(--dq-font-mono);
  word-break: break-word;
}

</style>
