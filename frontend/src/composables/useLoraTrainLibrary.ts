import { computed, ref } from 'vue';
import { api } from '@/utils/api';

export type LoraTrainRun = Record<string, unknown>;
export type UserLoraRow = Record<string, unknown>;

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function scalarText(value: unknown): string {
  if (value == null) return '';
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return String(value).trim();
  }
  const obj = asRecord(value);
  return String(obj.id || obj.name || '').trim();
}

function runParams(run: LoraTrainRun): Record<string, unknown> {
  return asRecord(run.params);
}

function runRequest(run: LoraTrainRun): Record<string, unknown> {
  return asRecord(runParams(run).request);
}

function runResultMeta(run: LoraTrainRun): Record<string, unknown> {
  const result = asRecord(run.result);
  return asRecord(result.metadata);
}

export function runTaskId(run: LoraTrainRun): string {
  return String(run.id || '');
}

export function runBaseModel(run: LoraTrainRun): string {
  const params = runParams(run);
  const request = runRequest(run);
  const raw =
    scalarText(params.base_model) ||
    scalarText(request.base_model) ||
    scalarText(run.model_id) ||
    scalarText(run.model) ||
    '—';
  return raw.split(':', 1)[0] || '—';
}

export function runPresetId(run: LoraTrainRun): string {
  const params = runParams(run);
  const request = runRequest(run);
  return (
    scalarText(request.preset) ||
    scalarText(asRecord(params.preset).id) ||
    scalarText(params.preset) ||
    ''
  );
}

export function runOutputName(run: LoraTrainRun): string {
  const params = runParams(run);
  const request = runRequest(run);
  const meta = runResultMeta(run);
  const training = asRecord(meta.training);
  const fromParams =
    scalarText(params.output_name) ||
    scalarText(request.output_name) ||
    scalarText(training.output_name);
  if (fromParams) return fromParams;
  return runTaskId(run).slice(0, 12) || '—';
}

export function runUserLoraId(run: LoraTrainRun): string {
  const meta = runResultMeta(run);
  const training =
    meta.training && typeof meta.training === 'object'
      ? (meta.training as Record<string, unknown>)
      : {};
  return String(meta.user_lora_id || training.user_lora_id || '').trim();
}

export function useLoraTrainLibrary() {
  const runs = ref<LoraTrainRun[]>([]);
  const userLoras = ref<UserLoraRow[]>([]);
  const loading = ref(false);

  const loraById = computed(() => {
    const map = new Map<string, UserLoraRow>();
    for (const row of userLoras.value) {
      const id = String(row.id || '').trim();
      if (id) map.set(id, row);
    }
    return map;
  });

  const loraByTaskId = computed(() => {
    const map = new Map<string, UserLoraRow>();
    for (const row of userLoras.value) {
      const tid = String(row.task_id || '').trim();
      if (tid) map.set(tid, row);
    }
    return map;
  });

  function userLoraForRun(run: LoraTrainRun): UserLoraRow | undefined {
    const tid = runTaskId(run);
    if (tid && loraByTaskId.value.has(tid)) {
      return loraByTaskId.value.get(tid);
    }
    const lid = runUserLoraId(run);
    if (lid && loraById.value.has(lid)) {
      return loraById.value.get(lid);
    }
    return undefined;
  }

  async function refresh({ limit = 12 }: { limit?: number } = {}) {
    loading.value = true;
    try {
      const [tasksRes, adaptersRes] = await Promise.all([
        api.gen.listMediaTasks({ kind: 'lora.training', limit }),
        api.loras.listUserAdapters(),
      ]);
      const taskPayload = tasksRes as { tasks?: LoraTrainRun[] };
      const adapterPayload = adaptersRes as { items?: UserLoraRow[] };
      runs.value = taskPayload.tasks || [];
      userLoras.value = adapterPayload.items || [];
    } catch {
      runs.value = [];
      userLoras.value = [];
    } finally {
      loading.value = false;
    }
  }

  return {
    runs,
    userLoras,
    loading,
    refresh,
    userLoraForRun,
    runOutputName,
    runBaseModel,
    runPresetId,
    runTaskId,
    runUserLoraId,
  };
}
