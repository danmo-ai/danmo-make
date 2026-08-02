<template>
  <div
    class="task-pipeline-graph"
    :class="{ 'task-pipeline-graph--compact': compact }"
    role="list"
  >
    <div v-if="!nodes.length" class="task-pipeline-graph__empty">
      {{ $t('studio.pipelineGraphEmpty') }}
    </div>
    <template v-else>
      <ol class="task-pipeline-graph__steps">
        <li
          v-for="(node, index) in nodes"
          :key="node.id"
          class="task-pipeline-graph__step"
          role="listitem"
        >
          <div class="task-pipeline-graph__track" aria-hidden="true">
            <span
              class="task-pipeline-graph__indicator"
              :class="`task-pipeline-graph__indicator--${nodeStatus(node)}`"
            />
            <span
              v-if="index < nodes.length - 1"
              class="task-pipeline-graph__connector"
              :class="{ 'task-pipeline-graph__connector--done': isStepDone(node) }"
            />
          </div>
          <div
            class="task-pipeline-graph__content"
            :class="{
              'task-pipeline-graph__content--active': node.id === activeNode,
              [`task-pipeline-graph__content--${nodeStatus(node)}`]: true,
            }"
            :aria-current="node.id === activeNode ? 'step' : undefined"
          >
            <span class="task-pipeline-graph__label">{{ node.label || node.id }}</span>
            <span v-if="!compact && node.duration_ms != null" class="task-pipeline-graph__dur">
              {{ formatMs(node.duration_ms) }}
            </span>
          </div>
        </li>
      </ol>

      <footer v-if="!compact" class="task-pipeline-graph__foot">
        <div v-if="displayProgress != null" class="task-pipeline-graph__progress-wrap">
          <div class="task-pipeline-graph__progress-meta">
            <span v-if="progressLabel" class="task-pipeline-graph__progress-label">
              {{ progressLabel }}
            </span>
            <span class="task-pipeline-graph__progress-value">
              {{ Math.round(displayProgress) }}%
            </span>
          </div>
          <div
            class="task-pipeline-graph__progress-bar"
            role="progressbar"
            :aria-valuenow="Math.round(displayProgress)"
            aria-valuemin="0"
            aria-valuemax="100"
          >
            <span
              class="task-pipeline-graph__progress-fill"
              :style="{ width: `${displayProgress}%` }"
            />
          </div>
        </div>
        <button
          v-if="showDiagnose"
          type="button"
          class="task-pipeline-graph__diagnose"
          :disabled="diagnosing"
          @click="$emit('diagnose')"
        >
          {{ diagnosing ? $t('studio.pipelineDiagnosing') : $t('studio.pipelineDiagnose') }}
        </button>
      </footer>
      <p v-if="!compact && diagnosis" class="task-pipeline-graph__diagnosis">{{ diagnosis }}</p>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';

export interface PipelineGraphNode {
  id: string;
  label?: string;
  status?: string;
  duration_ms?: number | null;
}

const props = defineProps<{
  nodes: PipelineGraphNode[];
  activeNode?: string | null;
  /** 0–100，来自实时去噪步或 SSE progress */
  progress?: number | null;
  progressLabel?: string | null;
  showDiagnose?: boolean;
  diagnosing?: boolean;
  diagnosis?: string | null;
  /** 顶部横向步骤条，不含进度/诊断区 */
  compact?: boolean;
}>();

defineEmits<{
  diagnose: [];
}>();

const displayProgress = computed(() => {
  if (typeof props.progress !== 'number' || !Number.isFinite(props.progress)) return null;
  return Math.min(100, Math.max(0, props.progress));
});

function nodeStatus(node: PipelineGraphNode): string {
  return node.status || 'pending';
}

function isStepDone(node: PipelineGraphNode): boolean {
  const status = nodeStatus(node);
  return status === 'ok' || status === 'failed' || status === 'cancelled';
}

function formatMs(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}
</script>

<style scoped>
.task-pipeline-graph {
  display: flex;
  flex-direction: column;
  gap: 14px;
  min-width: 0;
}

.task-pipeline-graph--compact {
  gap: 0;
}

.task-pipeline-graph__empty {
  color: var(--dq-label-secondary);
  font-size: var(--dq-font-size-caption);
  padding: 8px 4px;
}

.task-pipeline-graph__steps {
  list-style: none;
  margin: 0;
  padding: 4px 2px;
  display: flex;
  flex-direction: column;
  gap: 0;
}

.task-pipeline-graph--compact .task-pipeline-graph__steps {
  flex-direction: row;
  align-items: flex-start;
  gap: 0;
  padding: 2px 0;
  overflow-x: auto;
  scrollbar-width: thin;
}

.task-pipeline-graph__step {
  display: grid;
  grid-template-columns: 22px minmax(0, 1fr);
  gap: 10px;
  min-height: 42px;
}

.task-pipeline-graph--compact .task-pipeline-graph__step {
  display: flex;
  flex-direction: column;
  align-items: center;
  flex: 1 1 0;
  min-width: 52px;
  min-height: 0;
  gap: 6px;
}

.task-pipeline-graph__track {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding-top: 10px;
}

.task-pipeline-graph--compact .task-pipeline-graph__track {
  flex-direction: row;
  width: 100%;
  padding-top: 0;
  justify-content: center;
}

.task-pipeline-graph__indicator {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  border: 1.5px solid var(--dq-glass-border);
  background: var(--dq-surface-inset);
  flex-shrink: 0;
  z-index: 1;
}

.task-pipeline-graph--compact .task-pipeline-graph__indicator {
  width: 8px;
  height: 8px;
}

.task-pipeline-graph__indicator--running {
  border-color: var(--dq-accent);
  background: var(--dq-accent);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--dq-accent) 22%, transparent);
}

.task-pipeline-graph__indicator--ok {
  border-color: color-mix(in srgb, var(--dq-success) 55%, var(--dq-glass-border));
  background: color-mix(in srgb, var(--dq-success) 85%, transparent);
}

.task-pipeline-graph__indicator--failed,
.task-pipeline-graph__indicator--cancelled {
  border-color: var(--dq-danger);
  background: var(--dq-danger);
}

.task-pipeline-graph__indicator--pending {
  opacity: 0.55;
}

.task-pipeline-graph__connector {
  position: absolute;
  top: 22px;
  bottom: -4px;
  width: 1.5px;
  background: var(--dq-glass-border);
}

.task-pipeline-graph--compact .task-pipeline-graph__connector {
  position: static;
  flex: 1 1 auto;
  width: auto;
  height: 1.5px;
  min-width: 8px;
  margin-top: 3px;
  align-self: center;
}

.task-pipeline-graph__connector--done {
  background: color-mix(in srgb, var(--dq-success) 45%, var(--dq-glass-border));
}

.task-pipeline-graph__content {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  min-height: 36px;
  padding: 6px 10px;
  border-radius: var(--dq-radius-control);
  border: 0.5px solid transparent;
  background: transparent;
}

.task-pipeline-graph--compact .task-pipeline-graph__content {
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  min-height: 0;
  padding: 0 2px;
  text-align: center;
}

.task-pipeline-graph__content--active {
  border-color: color-mix(in srgb, var(--dq-accent) 35%, transparent);
  background: color-mix(in srgb, var(--dq-accent) 10%, transparent);
}

.task-pipeline-graph--compact .task-pipeline-graph__content--active {
  border-color: transparent;
  background: transparent;
}

.task-pipeline-graph--compact .task-pipeline-graph__content--active .task-pipeline-graph__label {
  color: var(--dq-accent);
  font-weight: 600;
}

.task-pipeline-graph__content--running .task-pipeline-graph__label {
  color: var(--dq-label-primary);
  font-weight: 600;
}

.task-pipeline-graph__content--pending .task-pipeline-graph__label {
  color: var(--dq-label-tertiary);
}

.task-pipeline-graph__label {
  font-size: var(--dq-font-size-body);
  line-height: 1.35;
  color: var(--dq-label-primary);
}

.task-pipeline-graph--compact .task-pipeline-graph__label {
  font-size: var(--dq-font-size-caption);
  line-height: 1.25;
  word-break: keep-all;
}

.task-pipeline-graph__dur {
  flex-shrink: 0;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
  font-variant-numeric: tabular-nums;
}

.task-pipeline-graph__foot {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding-top: 2px;
}

.task-pipeline-graph__progress-wrap {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.task-pipeline-graph__progress-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.task-pipeline-graph__progress-label {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
}

.task-pipeline-graph__progress-value {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
  font-variant-numeric: tabular-nums;
}

.task-pipeline-graph__progress-bar {
  height: 4px;
  border-radius: 999px;
  background: var(--dq-fill-on-glass);
  overflow: hidden;
}

.task-pipeline-graph__progress-fill {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: var(--dq-accent);
  transition: width 0.25s ease;
}

.task-pipeline-graph__diagnose {
  align-self: flex-end;
  font-size: var(--dq-font-size-caption);
  padding: 5px 12px;
  border-radius: var(--dq-radius-control);
  border: 0.5px solid var(--dq-glass-border);
  background: var(--dq-fill-on-glass);
  color: var(--dq-label-primary);
  cursor: pointer;
}

.task-pipeline-graph__diagnose:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.task-pipeline-graph__diagnosis {
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
</style>
