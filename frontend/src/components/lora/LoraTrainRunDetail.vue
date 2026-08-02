<template>
  <div class="lora-run-detail">
    <div class="lora-run-detail__head">
      <DqButton size="sm" @click="$emit('back')">{{ $t('common.back') }}</DqButton>
      <div class="lora-run-detail__head-main">
        <h3 class="lora-run-detail__title">{{ $t('loraTrain.runTitle') }}</h3>
        <TaskIdBadge :task-id="taskId" />
      </div>
    </div>

    <div v-if="task" class="lora-run-detail__status">
      <DqTag :type="statusTagType">{{ statusLabel }}</DqTag>
      <DqProgress :percentage="Math.round((task.progress || 0) * 100)" />
      <p v-if="progressMessage" class="lora-run-detail__progress-msg">{{ progressMessage }}</p>
      <p v-if="task.step != null && task.total != null" class="lora-run-detail__step-hint">
        {{ task.step }} / {{ task.total }}
        <span v-if="task.eta_seconds != null">
          · {{ $t('loraTrain.etaSeconds', { s: Math.max(0, Math.round(task.eta_seconds)) }) }}
        </span>
      </p>
      <p v-if="task.status === 'failed' && task.error" class="lora-run-detail__error">
        {{ task.error }}
      </p>
    </div>

    <div v-if="task && !isTerminal" class="lora-run-detail__queue-hint">
      <p>{{ $t('loraTrain.runInBackgroundHint') }}</p>
      <DqButton size="sm" type="secondary" @click="openGlobalTaskQueue">
        {{ $t('loraTrain.openTaskQueue') }}
      </DqButton>
    </div>

    <div v-if="artifacts?.loss_history?.length" class="lora-run-detail__section">
      <h4 class="lora-run-detail__section-title">{{ $t('loraTrain.lossCurve') }}</h4>
      <div class="lora-run-loss-bars">
        <div
          v-for="pt in lossBars"
          :key="pt.step"
          class="lora-run-loss-bar"
          :style="{ height: `${pt.heightPct}%` }"
          :title="lossBarTitle(pt)"
        />
      </div>
    </div>

    <LoraQualityHints
      v-if="task?.status === 'completed' && artifacts?.quality"
      :report="artifacts.quality"
      mode="training"
      :vision-available="visionAvailable"
      :vlm-loading="trainingVlmLoading"
      class="lora-run-detail__quality"
      @run-vlm="runTrainingVlmAudit"
    />

    <div v-if="artifacts?.progress_images?.length" class="lora-run-detail__section">
      <h4 class="lora-run-detail__section-title">{{ $t('loraTrain.progressGallery') }}</h4>
      <div class="lora-run-gallery-grid">
        <a
          v-for="name in artifacts.progress_images"
          :key="name"
          :href="artifactUrl(name)"
          target="_blank"
          rel="noopener"
        >
          <img :src="artifactUrl(name)" :alt="name" loading="lazy" />
        </a>
      </div>
    </div>

    <div
      v-if="canResumeTraining"
      class="lora-run-detail__section lora-run-detail__resume"
    >
      <h4 class="lora-run-detail__section-title">{{ $t('loraTrain.resumeTraining') }}</h4>
      <p class="lora-run-detail__resume-hint">{{ $t('loraTrain.resumeHint') }}</p>
      <DqButton size="sm" type="secondary" @click="resumeTraining">
        {{ $t('loraTrain.resumeFromRun') }}
      </DqButton>
    </div>

    <div
      v-if="artifacts?.checkpoints?.length && task?.status === 'completed'"
      class="lora-run-detail__section"
    >
      <h4 class="lora-run-detail__section-title">{{ $t('loraTrain.checkpoints') }}</h4>
      <div class="lora-run-checkpoint-row">
        <DqSelect v-model="selectedCheckpoint" size="sm" class="lora-run-checkpoint-select">
          <DqOption
            v-for="ck in artifacts.checkpoints"
            :key="ck"
            :label="ck"
            :value="ck"
          />
        </DqSelect>
        <DqInput v-model="registerName" :placeholder="$t('loraTrain.registerName')" size="sm" />
        <DqButton size="sm" type="secondary" :loading="registering" @click="registerSelected">
          {{ $t('loraTrain.registerCheckpoint') }}
        </DqButton>
      </div>
      <p v-if="registeredLoraId" class="lora-run-registered">
        {{ $t('loraTrain.registeredAs') }}: {{ registeredLoraId }}
      </p>
    </div>

    <div v-if="task?.status === 'completed'" class="lora-run-detail__actions">
      <DqButton type="primary" @click="verifyGenerate">
        {{ $t('loraTrain.verifyGenerate') }}
      </DqButton>
      <DqButton
        v-if="registeredLoraId || verifyLoraId()"
        size="sm"
        type="secondary"
        @click="openModelsUserLorasPage()"
      >
        {{ $t('loraTrain.viewInModels') }}
      </DqButton>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { useRouter } from 'vue-router';
import { api } from '@/utils/api';
import { toast } from '@/utils/feedback';
import { openGlobalTaskQueue } from '@/utils/appEvents';
import { openModelsUserLoras } from '@/utils/loraTrainHandoff';
import TaskIdBadge from '@/components/studio/TaskIdBadge.vue';
import LoraQualityHints from '@/components/lora/LoraQualityHints.vue';
import type { LoraTrainingQualityReport } from '@/utils/loraQuality';

const props = defineProps<{ taskId: string }>();
const emit = defineEmits<{
  (e: 'back'): void;
  (e: 'verify', payload: { prompt: string; loraId: string; baseModel: string }): void;
}>();

const { t } = useI18n();
const router = useRouter();
const task = ref<any>(null);
const artifacts = ref<any>(null);
const progressMessage = ref('');
const selectedCheckpoint = ref('');
const registerName = ref('');
const registering = ref(false);
const registeredLoraId = ref('');
const visionAvailable = ref(false);
const trainingVlmLoading = ref(false);
let taskStream: EventSource | null = null;
let lastArtifactRefresh = 0;

const statusTagType = computed(() => {
  const s = task.value?.status;
  if (s === 'completed') return 'success';
  if (s === 'failed') return 'danger';
  if (s === 'running') return 'primary';
  return 'info';
});

const isTerminal = computed(() => {
  const s = task.value?.status;
  return s === 'completed' || s === 'failed' || s === 'cancelled';
});

const statusLabel = computed(() => {
  const s = task.value?.status;
  if (!s) return '';
  const key = `loraTrain.runStatus.${s}`;
  const translated = t(key);
  return translated !== key ? translated : s;
});

const lossBars = computed(() => {
  const hist = artifacts.value?.loss_history;
  if (!Array.isArray(hist) || !hist.length) return [];
  const maxLoss = Math.max(...hist.map((p: { loss?: number }) => Number(p.loss) || 0), 1e-6);
  return hist.map((pt: { step: number; loss: number; val_loss?: number }) => ({
    step: pt.step,
    loss: pt.loss,
    val_loss: pt.val_loss,
    heightPct: Math.max(4, (Number(pt.loss) / maxLoss) * 100),
  }));
});

const canResumeTraining = computed(() => {
  const s = task.value?.status;
  if (s !== 'failed' && s !== 'cancelled') return false;
  return Array.isArray(artifacts.value?.checkpoints) && artifacts.value.checkpoints.length > 0;
});

function lossBarTitle(pt: { step: number; loss: number; val_loss?: number }): string {
  const base = `step ${pt.step}: loss ${Number(pt.loss).toFixed(4)}`;
  if (pt.val_loss != null && !Number.isNaN(Number(pt.val_loss))) {
    return `${base}, val ${Number(pt.val_loss).toFixed(4)}`;
  }
  return base;
}

function resumeTraining() {
  const checkpoints = artifacts.value?.checkpoints as string[] | undefined;
  if (!checkpoints?.length) return;
  const finals = checkpoints.filter((c) => c.includes('final'));
  const ck = selectedCheckpoint.value || finals[finals.length - 1] || checkpoints[checkpoints.length - 1];
  router.push({
    name: 'lora_train',
    query: {
      resume_task: props.taskId,
      resume_checkpoint: ck,
    },
  });
}

function mergeProgressDetail(row: Record<string, unknown> | null): Record<string, unknown> | null {
  if (!row) return row;
  const detail = row.progress_detail;
  if (!detail || typeof detail !== 'object') return row;
  const d = detail as Record<string, unknown>;
  return {
    ...row,
    step: d.step ?? row.step,
    total: d.total ?? row.total,
    eta_seconds: d.eta_seconds ?? row.eta_seconds,
  };
}

function applyProgressPayload(data: Record<string, unknown>) {
  const patch: Record<string, unknown> = {};
  if (typeof data.progress === 'number') patch.progress = data.progress;
  if (data.step != null) patch.step = data.step;
  if (data.total != null) patch.total = data.total;
  if (data.eta_seconds != null) patch.eta_seconds = data.eta_seconds;
  if (Object.keys(patch).length) {
    patchTask({ status: 'running', ...patch });
  }
  const msg = data.message ?? data.phase;
  if (msg != null && String(msg).trim()) {
    progressMessage.value = String(msg);
  }
}

function artifactUrl(name: string): string {
  return api.loras.artifactFileUrl(props.taskId, name);
}

function patchTask(patch: Record<string, unknown>) {
  task.value = { ...(task.value || {}), ...patch };
}

async function refreshArtifacts(force = false) {
  const now = Date.now();
  if (!force && now - lastArtifactRefresh < 8000) return;
  lastArtifactRefresh = now;
  try {
    const next = await api.loras.trainingArtifacts(props.taskId);
    artifacts.value = next;
    if ((next as any)?.vision_available != null) {
      visionAvailable.value = Boolean((next as any).vision_available);
    }
    if (!selectedCheckpoint.value && (next as any)?.checkpoints?.length) {
      const finals = (next as any).checkpoints.filter((c: string) => c.includes('final'));
      selectedCheckpoint.value = finals[finals.length - 1] || (next as any).checkpoints.at(-1);
    }
  } catch {
    if (force) artifacts.value = null;
  }
}

async function refreshVisionAvailability() {
  try {
    const info = await api.gen.getVisionModelInfo();
    visionAvailable.value = Boolean(info?.available);
  } catch {
    visionAvailable.value = false;
  }
}

async function runTrainingVlmAudit() {
  if (trainingVlmLoading.value) return;
  trainingVlmLoading.value = true;
  try {
    const quality = (await api.loras.trainingQualityVlm(props.taskId)) as LoraTrainingQualityReport;
    artifacts.value = { ...(artifacts.value || {}), quality };
    toast.success(t('loraTrain.quality.vlmAuditDone'));
  } catch (e: unknown) {
    const err = e as { response?: { data?: { detail?: { message?: string } | string } }; message?: string };
    const detail = err?.response?.data?.detail;
    const msg =
      detail && typeof detail === 'object' && detail.message
        ? detail.message
        : typeof detail === 'string'
          ? detail
          : err?.message || String(e);
    toast.error(msg);
  } finally {
    trainingVlmLoading.value = false;
  }
}

async function refreshTask() {
  const row = (await api.gen.getMediaTask(props.taskId)) as Record<string, unknown>;
  task.value = mergeProgressDetail(row);
  const detail = row?.progress_detail as Record<string, unknown> | undefined;
  if (detail?.message != null) {
    progressMessage.value = String(detail.message);
  } else if (detail?.phase != null) {
    progressMessage.value = String(detail.phase);
  }
  await refreshArtifacts(true);
}

function closeStream() {
  if (taskStream) {
    try {
      taskStream.close();
    } catch {
      /* ignore */
    }
    taskStream = null;
  }
}

function startStream() {
  closeStream();
  taskStream = api.gen.streamMediaTask(props.taskId, {
    onProgress: (data: any) => {
      applyProgressPayload(data);
      void refreshArtifacts(false);
    },
    onStatus: (data: any) => {
      patchTask(data);
    },
    onDone: async (data: any) => {
      patchTask({ status: data.status });
      progressMessage.value = '';
      await refreshTask();
    },
    onError: () => {
      void refreshTask();
    },
  });
}

watch(
  () => task.value?.result?.metadata?.user_lora_id,
  (id) => {
    if (id && !registeredLoraId.value) registeredLoraId.value = String(id);
  },
  { immediate: true }
);

watch(
  () => props.taskId,
  async () => {
    progressMessage.value = '';
    await refreshTask();
    if (!isTerminal.value) startStream();
    else closeStream();
  }
);

function verifyLoraId(): string {
  return (
    registeredLoraId.value ||
    String(task.value?.result?.metadata?.user_lora_id || task.value?.result?.metadata?.training?.user_lora_id || '')
  );
}

function verifyGenerate() {
  const meta = task.value?.result?.metadata?.training || {};
  const params = task.value?.params || {};
  const caption = String(
    meta.training_caption ||
      params.progress_prompt ||
      meta.progress_prompt ||
      ''
  ).trim();
  emit('verify', {
    prompt: caption,
    loraId: verifyLoraId(),
    baseModel: String(params.base_model || 'flux1-dev'),
  });
}

async function registerSelected() {
  if (!selectedCheckpoint.value) return;
  registering.value = true;
  try {
    const res = (await api.loras.registerCheckpoint(props.taskId, {
      checkpoint: selectedCheckpoint.value,
      name: registerName.value.trim(),
    })) as { user_lora?: { id?: string } };
    registeredLoraId.value = String(res.user_lora?.id || '');
    toast.success(t('loraTrain.registerSuccess'));
  } catch (e: any) {
    const msg = e?.response?.data?.detail?.message || e?.message || String(e);
    toast.error(msg);
  } finally {
    registering.value = false;
  }
}

function openModelsUserLorasPage() {
  openModelsUserLoras(router);
}

onMounted(async () => {
  await Promise.all([refreshTask(), refreshVisionAvailability()]);
  if (!isTerminal.value) startStream();
});

onUnmounted(() => {
  closeStream();
});
</script>

<style scoped>
.lora-run-detail {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 20px 22px;
  border-radius: var(--dq-radius-group, 16px);
  border: 0.5px solid color-mix(in srgb, var(--dq-border) 50%, transparent);
  background: color-mix(in srgb, var(--dq-fill-secondary) 72%, var(--dq-bg-base));
  overflow: auto;
}

.lora-run-detail :deep(.dq-surface-card__body) {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.lora-run-detail__head {
  display: flex;
  align-items: flex-start;
  gap: 12px;
}

.lora-run-detail__head-main {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
  min-width: 0;
}

.lora-run-detail__title {
  margin: 0;
  font-size: var(--dq-font-size-title);
  font-weight: 600;
  color: var(--dq-label-primary);
}

.lora-run-detail__status {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.lora-run-detail__progress-msg {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
  line-height: 1.4;
}

.lora-run-detail__step-hint {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
}

.lora-run-detail__error {
  margin: 0;
  padding: 10px 12px;
  border-radius: var(--dq-radius-group);
  background: color-mix(in srgb, var(--dq-danger) 10%, var(--dq-fill-secondary));
  border: 0.5px solid color-mix(in srgb, var(--dq-danger) 30%, transparent);
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-primary);
  line-height: 1.45;
  word-break: break-word;
}

.lora-run-detail__queue-hint {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 8px;
  padding: 10px 12px;
  border-radius: var(--dq-radius-group);
  background: var(--dq-fill-secondary);
  border: 0.5px solid var(--dq-border-subtle);
}

.lora-run-detail__queue-hint p {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
  line-height: 1.45;
}

.lora-run-detail__section-title {
  margin: 0 0 10px;
  font-size: var(--dq-font-size-body);
  font-weight: 600;
  color: var(--dq-label-primary);
}

.lora-run-detail__quality {
  margin: 4px 0 8px;
}

.lora-run-detail__resume-hint {
  margin: 0 0 10px;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
  line-height: 1.45;
}

.lora-run-gallery-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
  gap: 10px;
}

.lora-run-gallery-grid img {
  width: 100%;
  border-radius: var(--dq-radius-group);
  border: 0.5px solid var(--dq-border);
}

.lora-run-loss-bars {
  display: flex;
  align-items: flex-end;
  gap: 2px;
  height: 80px;
  padding: 8px;
  border-radius: var(--dq-radius-group);
  background: var(--dq-fill-secondary);
  border: 0.5px solid var(--dq-border-subtle);
}

.lora-run-loss-bar {
  flex: 1;
  min-width: 4px;
  background: var(--dq-accent);
  opacity: 0.75;
  border-radius: 2px 2px 0 0;
}

.lora-run-checkpoint-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}

.lora-run-checkpoint-select {
  min-width: min(280px, 100%);
}

.lora-run-registered {
  margin: 8px 0 0;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
}

.lora-run-detail__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding-top: 4px;
  border-top: 0.5px solid var(--dq-border-subtle);
}
</style>
