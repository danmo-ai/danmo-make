import { reactive } from 'vue';
import { defineStore } from 'pinia';
import { api } from '@/utils/api';
import type { QueueState, TaskTrace, TraceStepProgress } from '@/types';
import {
  buildInferenceResultLogMessage,
  formatGenLogMessage,
  formatLogTimestamp,
  isDuplicateDenoiseStepLog,
  progressPhaseLabel,
} from '@/utils/genTaskLog';
import { formatStepProgressTitle } from '@/utils/traceProgress';
import { $tt } from '@/utils/i18n';

export interface LiveTaskProgress {
  progress?: number;
  step?: number;
  total?: number;
  eta_seconds?: number;
  progressMessage?: string;
  phase?: string;
}

export interface TaskLogEntry {
  time: string;
  message: string;
  level: string;
}

export interface TaskPipelineGraphState {
  graph_id?: string;
  source?: 'otel' | 'heuristic';
  nodes: Array<{
    id: string;
    label?: string;
    status?: string;
    duration_ms?: number | null;
  }>;
  active_node?: string | null;
  progress?: number;
  step_progress?: TraceStepProgress | null;
  trace_id?: string;
  duration_ms?: number | null;
}

interface TaskLogPhaseState {
  lastPhase: string;
  lastStep: number;
}

export const useTasksStore = defineStore('tasks', () => {
  const queueState = reactive<QueueState>({
    running: [],
    queued: [],
  });

  const liveTaskProgress = reactive<Record<string, LiveTaskProgress>>({});
  const taskLogs = reactive<Record<string, TaskLogEntry[]>>({});
  const taskPipelineGraphs = reactive<Record<string, TaskPipelineGraphState>>({});
  const taskTraces = reactive<Record<string, TaskTrace>>({});
  const taskTraceErrors = reactive<Record<string, string>>({});
  const taskLogPhaseState = new Map<string, TaskLogPhaseState>();

  let pollTimer: ReturnType<typeof setInterval> | null = null;
  let pollRefCount = 0;

  function patchLiveTaskProgress(taskId: string, patch: LiveTaskProgress) {
    if (!taskId || !patch || Object.keys(patch).length === 0) return;
    const prev = liveTaskProgress[taskId] || {};
    const next = { ...prev };
    if (typeof patch.progress === 'number') next.progress = patch.progress;
    if (patch.step != null) next.step = patch.step;
    if (patch.total != null) next.total = patch.total;
    if (patch.eta_seconds != null) next.eta_seconds = patch.eta_seconds;
    if (Object.prototype.hasOwnProperty.call(patch, 'progressMessage')) {
      next.progressMessage = patch.progressMessage;
    }
    if (Object.prototype.hasOwnProperty.call(patch, 'phase')) {
      next.phase = patch.phase;
    }
    // Skip no-op writes so SSE chatter does not thrash Vue reactivity.
    if (
      prev.progress === next.progress &&
      prev.step === next.step &&
      prev.total === next.total &&
      prev.eta_seconds === next.eta_seconds &&
      prev.progressMessage === next.progressMessage &&
      prev.phase === next.phase
    ) {
      return;
    }
    liveTaskProgress[taskId] = next;
  }

  function applyStepProgressTip(
    taskId: string,
    tip: TraceStepProgress | null | undefined,
    progress?: number | null,
  ) {
    if (!taskId || !tip) return;
    const title = formatStepProgressTitle(tip);
    patchLiveTaskProgress(taskId, {
      phase: tip.phase,
      step: tip.step,
      total: tip.total,
      progress: typeof progress === 'number' ? progress : undefined,
      progressMessage: title ?? tip.phase,
    });
  }

  function clearLiveTaskProgress(taskId: string) {
    if (taskId && liveTaskProgress[taskId] != null) {
      delete liveTaskProgress[taskId];
    }
  }

  function appendTaskLog(taskId: string, message: string, level = 'info') {
    if (!taskId || !String(message || '').trim()) return;
    if (!taskLogs[taskId]) {
      taskLogs[taskId] = [];
    }
    const logs = taskLogs[taskId];
    if (isDuplicateDenoiseStepLog(logs, message)) {
      return;
    }
    logs.push({
      time: formatLogTimestamp(),
      message,
      level,
    });
    if (logs.length > 500) {
      taskLogs[taskId] = logs.slice(-500);
    }
  }

  function ingestTaskLog(taskId: string, logData: unknown) {
    const row = logData as Record<string, unknown>;
    const raw = String(row.message || '');
    const lvl = String(row.level || 'info');
    appendTaskLog(taskId, formatGenLogMessage(raw), lvl);
  }

  function ingestTaskProgressLog(taskId: string, data: Record<string, unknown>) {
    const state = taskLogPhaseState.get(taskId) || { lastPhase: '', lastStep: 0 };

    if (data.message === 'post') {
      if (state.lastPhase !== 'post') {
        state.lastPhase = 'post';
        appendTaskLog(taskId, $tt('studio.queuePostProcessHint'), 'info');
      }
    } else if (data.message === 'denoise') {
      state.lastPhase = 'denoise';
    } else if (data.phase && state.lastPhase !== String(data.phase)) {
      state.lastPhase = String(data.phase);
      const label = progressPhaseLabel(String(data.phase), '');
      if (label) {
        appendTaskLog(taskId, label, 'info');
      }
    }

    const nextStep = data.step != null ? Number(data.step) : state.lastStep;
    const nextTotal = data.total != null ? Number(data.total) : 0;
    if (nextTotal > 0 && nextStep > 0) {
      state.lastStep = nextStep;
    }
    taskLogPhaseState.set(taskId, state);
  }

  function clearTaskLogs(taskId: string) {
    if (taskId && taskLogs[taskId] != null) {
      delete taskLogs[taskId];
    }
    taskLogPhaseState.delete(taskId);
  }

  function patchTaskPipelineGraph(taskId: string, graph: TaskPipelineGraphState) {
    if (!taskId || !graph?.nodes) return;
    taskPipelineGraphs[taskId] = graph;
  }

  function ingestTaskPipelineTrace(taskId: string, data: unknown) {
    patchTaskPipelineGraph(taskId, data as TaskPipelineGraphState);
  }

  async function loadTaskPipelineGraph(taskId: string) {
    if (!taskId) return;
    try {
      const data = (await api.tasks.fetchGraph(taskId)) as unknown as TaskPipelineGraphState;
      patchTaskPipelineGraph(taskId, data);
    } catch {
      /* graph optional until task runs */
    }
  }

  function clearTaskPipelineGraph(taskId: string) {
    if (taskId && taskPipelineGraphs[taskId] != null) {
      delete taskPipelineGraphs[taskId];
    }
  }

  /** Spans + graph for one task. Failures are recorded, never swallowed into
   * an empty trace that looks like "nothing ran". */
  async function loadTaskTrace(taskId: string): Promise<void> {
    if (!taskId) return;
    delete taskTraceErrors[taskId];
    try {
      const trace = await api.tasks.fetchTrace(taskId);
      taskTraces[taskId] = trace;
      const graph = (await api.tasks.fetchGraph(taskId)) as unknown as TaskPipelineGraphState;
      patchTaskPipelineGraph(taskId, graph);
      const tip = graph.step_progress ?? trace.step_progress ?? null;
      applyStepProgressTip(taskId, tip, graph.progress ?? trace.progress);
    } catch (e: unknown) {
      taskTraceErrors[taskId] = traceErrorMessage(e);
    }
  }

  function clearTaskTrace(taskId: string) {
    if (taskId) {
      delete taskTraces[taskId];
      delete taskTraceErrors[taskId];
      clearTaskPipelineGraph(taskId);
    }
  }

  function traceErrorMessage(e: unknown): string {
    if (typeof e === 'object' && e !== null && 'response' in e) {
      const err = e as { response?: { data?: { detail?: string } } };
      const detail = err.response?.data?.detail;
      if (detail) return detail;
    }
    if (e instanceof Error) return e.message;
    return String(e);
  }

  const logStreams = new Map<string, EventSource>();
  /** Task ids whose SSE is owned by a create page (avoid duplicate empty store streams). */
  const pageOwnedStreams = new Set<string>();

  function closeTaskLogStream(taskId: string) {
    const es = logStreams.get(taskId);
    if (es) {
      try {
        es.close();
      } catch {
        /* ignore */
      }
      logStreams.delete(taskId);
    }
  }

  function registerPageOwnedStream(taskId: string) {
    if (!taskId) return;
    pageOwnedStreams.add(taskId);
    closeTaskLogStream(taskId);
  }

  function unregisterPageOwnedStream(taskId: string) {
    if (!taskId) return;
    pageOwnedStreams.delete(taskId);
    closeTaskLogStream(taskId);
  }

  function openTaskLogStream(
    taskId: string,
    callbacks: {
      onLog?: (data: unknown) => void;
      onStatus?: (data: unknown) => void;
      onProgress?: (data: unknown) => void;
      onResult?: (data: unknown) => void;
      onDone?: (data: unknown) => void;
      onError?: (event: Event) => void;
    }
  ): EventSource | null {
    if (pageOwnedStreams.has(taskId)) {
      return logStreams.get(taskId) ?? null;
    }
    closeTaskLogStream(taskId);
    const url = api.tasks.logStreamUrl(taskId);
    const eventSource = new EventSource(url);
    logStreams.set(taskId, eventSource);

    eventSource.addEventListener('log', (event) => {
      const data = JSON.parse(event.data);
      ingestTaskLog(taskId, data);
      callbacks.onLog?.(data);
    });

    eventSource.addEventListener('trace', (event) => {
      const data = JSON.parse(event.data);
      ingestTaskPipelineTrace(taskId, data);
      callbacks.onProgress?.(data);
    });

    eventSource.addEventListener('progress', (event) => {
      const data = JSON.parse(event.data);
      ingestTaskProgressLog(taskId, data as Record<string, unknown>);
      const patch: LiveTaskProgress = {};
      if (typeof data.progress === 'number') patch.progress = data.progress;
      if (data.step != null) patch.step = data.step;
      if (data.total != null) patch.total = data.total;
      if (data.eta_seconds != null) patch.eta_seconds = data.eta_seconds;
      if (data.phase != null) patch.phase = String(data.phase);
      if (data.step != null && data.total != null) {
        const tip = {
          phase: String(data.phase || data.message || ''),
          step: Number(data.step),
          total: Number(data.total),
        };
        patch.progressMessage = formatStepProgressTitle(tip) ?? String(data.message || tip.phase);
      } else if (Object.prototype.hasOwnProperty.call(data, 'message')) {
        patch.progressMessage = data.message;
      } else if (data.phase) {
        patch.progressMessage = String(data.phase);
      }
      patchLiveTaskProgress(taskId, patch);
      callbacks.onProgress?.(data);
      if (!callbacks.onProgress && callbacks.onStatus) {
        callbacks.onStatus?.({
          status: 'running',
          progress: data.progress,
          step: data.step,
          total: data.total,
          eta_seconds: data.eta_seconds,
        });
      }
    });

    eventSource.addEventListener('status', (event) => {
      const data = JSON.parse(event.data);
      if (typeof data.progress === 'number') {
        patchLiveTaskProgress(taskId, { progress: data.progress });
      }
      callbacks.onStatus?.(data);
    });

    eventSource.addEventListener('result', (event) => {
      const data = JSON.parse(event.data) as Record<string, unknown>;
      const meta =
        (data.metadata as Record<string, unknown> | undefined) ??
        ((data.result as Record<string, unknown> | undefined)?.metadata as
          | Record<string, unknown>
          | undefined);
      const inferenceMsg = buildInferenceResultLogMessage(meta);
      if (inferenceMsg) {
        appendTaskLog(taskId, inferenceMsg, 'success');
      }
      callbacks.onResult?.(data);
    });

    eventSource.addEventListener('done', (event) => {
      const data = JSON.parse(event.data);
      callbacks.onDone?.(data);
      clearLiveTaskProgress(taskId);
      closeTaskLogStream(taskId);
    });

    eventSource.addEventListener('error', (event) => {
      callbacks.onError?.(event);
      clearLiveTaskProgress(taskId);
      closeTaskLogStream(taskId);
    });

    return eventSource;
  }

  async function pollQueueOnce(): Promise<void> {
    try {
      const data = await api.gen.getQueue();
      queueState.running = data.running || [];
      queueState.queued = data.queued || [];
      for (const t of queueState.running || []) {
        if (t?.id && !logStreams.has(t.id) && !pageOwnedStreams.has(t.id)) {
          openTaskLogStream(t.id, {});
        }
      }
    } catch (e) {
      console.error('TasksStore: queue poll failed', e);
    }
  }

  function ensureQueuePoller(): void {
    pollRefCount += 1;
    if (!pollTimer) {
      pollQueueOnce();
      pollTimer = setInterval(pollQueueOnce, 2000);
    }
  }

  function releaseQueuePoller(): void {
    pollRefCount = Math.max(0, pollRefCount - 1);
    if (pollRefCount === 0 && pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  return {
    queueState,
    liveTaskProgress,
    taskLogs,
    taskPipelineGraphs,
    taskTraces,
    taskTraceErrors,
    patchLiveTaskProgress,
    clearLiveTaskProgress,
    applyStepProgressTip,
    appendTaskLog,
    ingestTaskLog,
    ingestTaskProgressLog,
    clearTaskLogs,
    patchTaskPipelineGraph,
    ingestTaskPipelineTrace,
    loadTaskPipelineGraph,
    clearTaskPipelineGraph,
    loadTaskTrace,
    clearTaskTrace,
    openTaskLogStream,
    closeTaskLogStream,
    registerPageOwnedStream,
    unregisterPageOwnedStream,
    pollQueueOnce,
    ensureQueuePoller,
    releaseQueuePoller,
  };
});