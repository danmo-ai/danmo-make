/**
 * Registry model install + SSE progress (shared with Models page).
 */
import { onBeforeUnmount, ref } from 'vue';
import { toast } from '@/utils/feedback';
import { api } from '@/utils/api';
import { $tt } from '@/utils/i18n';

export function useModelInstall(opts?: { onCompleted?: () => void }) {
  const downloading = ref<Record<string, boolean>>({});
  const progressByKey = ref<Record<string, number>>({});
  const sseConnections = ref<Record<string, EventSource>>({});
  const settledByKey = ref<Record<string, () => void>>({});

  function uiKey(modelId: string, versionKey: string): string {
    return `${modelId}-${versionKey}`;
  }

  function closeAllSse() {
    for (const id of Object.keys(sseConnections.value)) {
      sseConnections.value[id].close();
    }
    sseConnections.value = {};
  }

  function settle(key: string) {
    const cb = settledByKey.value[key];
    delete settledByKey.value[key];
    cb?.();
  }

  function connectProgressSSE(taskId: string, label: string, key: string) {
    if (sseConnections.value[taskId]) {
      sseConnections.value[taskId].close();
    }
    const eventSource = new EventSource(api.download.installProgressStreamUrl(taskId));
    sseConnections.value[taskId] = eventSource;

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        progressByKey.value[key] = Math.round((data.progress || 0) * 100);
        if (data.status === 'completed') {
          eventSource.close();
          delete sseConnections.value[taskId];
          delete downloading.value[key];
          setTimeout(() => {
            delete progressByKey.value[key];
          }, 2000);
          toast.success($tt('download.downloadComplete', { name: label }));
          opts?.onCompleted?.();
          settle(key);
        } else if (data.status === 'failed' || data.status === 'cancelled') {
          eventSource.close();
          delete sseConnections.value[taskId];
          delete downloading.value[key];
          delete progressByKey.value[key];
          if (data.status === 'failed') {
            toast.error($tt('download.downloadFailed', { name: label, msg: data.error_message || '' }));
          }
          settle(key);
        }
      } catch {
        /* ignore malformed SSE */
      }
    };

    eventSource.onerror = () => {
      eventSource.close();
      delete sseConnections.value[taskId];
      delete downloading.value[key];
      settle(key);
    };
  }

  /**
   * Start (or re-attach to) a registry install. `onSettled` fires once when the
   * download reaches completed / failed / cancelled, or immediately when no task
   * could be started — callers can await a batch without polling.
   */
  async function installModel(
    modelId: string,
    versionKey: string,
    label: string,
    onSettled?: () => void,
  ) {
    const key = uiKey(modelId, versionKey);
    if (onSettled) settledByKey.value[key] = onSettled;
    if (downloading.value[key]) return;
    downloading.value[key] = true;
    progressByKey.value[key] = 0;
    try {
      const tasks = (await api.download.listDownloads()) as Array<{
        id: string;
        status: string;
        model_name?: string;
        version?: string | null;
      }>;
      const active = tasks.find(
        (t) =>
          t.model_name === modelId &&
          (t.version ?? null) === (versionKey || null) &&
          (t.status === 'running' || t.status === 'paused' || t.status === 'failed')
      );
      if (active?.id) {
        if (active.status === 'failed') {
          await api.download.resume(active.id);
        }
        connectProgressSSE(active.id, label, key);
        return;
      }

      const data = (await api.models.install(modelId, { version: versionKey })) as {
        task_id?: string;
      };
      if (data?.task_id) {
        connectProgressSSE(data.task_id, label, key);
      } else {
        delete downloading.value[key];
        settle(key);
      }
    } catch (e: unknown) {
      delete downloading.value[key];
      delete progressByKey.value[key];
      const msg = e instanceof Error ? e.message : String(e);
      toast.error($tt('download.downloadFailed', { name: label, msg }));
      settle(key);
    }
  }

  onBeforeUnmount(() => {
    closeAllSse();
  });

  return {
    downloading,
    progressByKey,
    installModel,
    uiKey,
  };
}
