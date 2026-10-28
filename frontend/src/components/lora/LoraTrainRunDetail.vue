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
      v-if="task?.status === 'completed'"
      class="lora-run-detail__section"
    >
      <LoraCheckpointComparison
        :task-id="taskId"
        :evaluation="artifacts?.candidate_evaluation || null"
        @registered="onCandidateRegistered"
        @retry="onCandidateRetry"
      />
    </div>

    <div v-if="task?.status === 'completed' && registeredLoraId" class="lora-run-detail__verification">
      <h4 class="lora-run-detail__section-title">{{ $t('loraTrain.verifyTitle') }}</h4>
      <p class="lora-run-detail__verify-hint">{{ $t('loraTrain.verifyHint') }}</p>
      <DqAlert type="success" :closable="false" :title="$t('loraTrain.verifyCheckpointBound', { checkpoint: selectedCheckpoint || 'final_adapters.safetensors' })" />
      <DqAlert
        v-if="verification.extra_adapters?.length"
        type="info"
        :closable="false"
        :title="$t('loraTrain.verifyExtraAdapters', { adapters: verification.extra_adapters.join(', ') })"
      />
      <div class="lora-run-detail__verify-grid">
        <label>{{ $t('loraTrain.progressPrompt') }}<DqInput v-model="verifyPrompt" type="textarea" :rows="3" /></label>
        <label>{{ $t('studio.seed') }}<DqInput v-model="verification.seed" /></label>
        <label>{{ $t('create.loraScale') }}<DqInput v-model.number="verification.lora_scale" type="number" min="0" max="2" step="0.05" /></label>
        <label>{{ $t('create.stepsLabel') }}<DqInput v-model.number="verification.steps" type="number" min="1" max="150" /></label>
        <label>{{ $t('create.guidanceLabel') }}<DqInput v-model.number="verification.guidance" type="number" min="0" max="30" step="0.1" /></label>
      </div>
      <p class="lora-run-detail__verify-recipe">{{ verificationSummary }}</p>
      <div class="lora-run-detail__actions">
        <DqButton type="primary" :disabled="!verifyLoraId() || !verifyPrompt.trim()" @click="verifyGenerate">
          {{ $t('loraTrain.verifyGenerate') }}
        </DqButton>
        <DqButton
          size="sm"
          type="secondary"
          @click="openModelsUserLorasPage()"
        >
          {{ $t('loraTrain.viewInModels') }}
        </DqButton>
      </div>
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
import LoraCheckpointComparison from '@/components/lora/LoraCheckpointComparison.vue';
import type { LoraTrainingQualityReport } from '@/utils/loraQuality';

const props = defineProps<{ taskId: string }>();
const emit = defineEmits<{
  (e: 'back'): void;
  (e: 'verify', payload: Record<string, unknown>): void;
  (e: 'resume', payload: { taskId: string; checkpoint: string }): void;
}>();

const { t } = useI18n();
const router = useRouter();
const task = ref<any>(null);
const artifacts = ref<any>(null);
const progressMessage = ref('');
const selectedCheckpoint = ref('');
const registeredLoraId = ref('');
const verifyPrompt = ref('');
const registeredRecipe = ref<Record<string, any> | null>(null);
const registeredCheckpoint = ref('');
const verification = ref<Record<string, any>>({ seed: '42', steps: 30, guidance: 4, lora_scale: 0.85, width: 1024, height: 1024, model: 'z-image-turbo', scheduler: 'linear' });
const visionAvailable = ref(false);
const trainingVlmLoading = ref(false);
let taskStream: EventSource | null = null;
let lastArtifactRefresh = 0;
let candidatePollTimer: ReturnType<typeof setInterval> | null = null;

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
  // Parent must apply: same-route navigation alone never re-runs onMounted, and
  // activeRunId would keep this detail panel open (resume looks like a no-op).
  emit('resume', { taskId: props.taskId, checkpoint: ck });
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
    const evalStatus = String((next as any)?.candidate_evaluation?.status || '');
    if (!selectedCheckpoint.value) {
      const selected = String((next as any)?.candidate_evaluation?.selected_checkpoint || '');
      if (selected) selectedCheckpoint.value = selected;
      else if ((next as any)?.checkpoints?.length) {
        const finals = (next as any).checkpoints.filter((c: string) => c.includes('final'));
        selectedCheckpoint.value = finals[finals.length - 1] || (next as any).checkpoints.at(-1);
      }
    }
    const registeredFromEval = String((next as any)?.candidate_evaluation?.user_lora_id || '');
    if (registeredFromEval && !registeredLoraId.value) {
      registeredLoraId.value = registeredFromEval;
      registeredCheckpoint.value = String((next as any)?.candidate_evaluation?.selected_checkpoint || '');
    }
    syncVerification();
    syncCandidatePolling(evalStatus);
  } catch {
    if (force) artifacts.value = null;
  }
}

function stopCandidatePolling() {
  if (candidatePollTimer) {
    clearInterval(candidatePollTimer);
    candidatePollTimer = null;
  }
}

function syncCandidatePolling(evalStatus: string) {
  const waiting = evalStatus === 'pending_evaluation' || evalStatus === 'evaluating';
  if (!waiting) {
    stopCandidatePolling();
    return;
  }
  if (candidatePollTimer) return;
  candidatePollTimer = setInterval(() => {
    void refreshArtifacts(true);
  }, 4000);
}

function onCandidateRegistered(payload: { user_lora_id: string; checkpoint: string }) {
  registeredLoraId.value = payload.user_lora_id;
  registeredCheckpoint.value = payload.checkpoint;
  selectedCheckpoint.value = payload.checkpoint;
  void refreshArtifacts(true);
}

function onCandidateRetry() {
  void refreshArtifacts(true);
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
  async (id) => {
    if (id && !registeredLoraId.value) registeredLoraId.value = String(id);
    const active = verifyLoraId();
    if (!active) return;
    try {
      const result = await api.loras.listUserAdapters() as { items?: Array<Record<string, any>> };
      registeredRecipe.value = result.items?.find((item) => item.id === active) || null;
      syncVerification();
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: { message?: string } | string } }; message?: string };
      const detail = err.response?.data?.detail;
      toast.error(typeof detail === 'object' && detail?.message ? detail.message : String(detail || err.message || e));
    }
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
  const active = selectedCheckpoint.value || 'final_adapters.safetensors';
  if (registeredCheckpoint.value && registeredCheckpoint.value !== active) return '';
  return (
    registeredLoraId.value ||
    String(task.value?.result?.metadata?.user_lora_id || task.value?.result?.metadata?.training?.user_lora_id || '')
  );
}

function verificationRecipe(): Record<string, any> {
  const params = task.value?.params || {};
  const registered = registeredCheckpoint.value === (selectedCheckpoint.value || 'final_adapters.safetensors');
  return (registered && registeredRecipe.value?.inference) || artifacts.value?.inference || params.metadata?.inference || {
    model: String(params.base_model || 'z-image'),
    steps: Number(params.progress_steps || 30),
    guidance: Number(params.guidance || 4),
    lora_weight: 1,
    scheduler: 'linear',
  };
}

const verificationSummary = computed(() => t('loraTrain.verifyRecipe', {
  model: verification.value.model,
  steps: verification.value.steps,
  guidance: verification.value.guidance,
  weight: verification.value.lora_scale,
  width: verification.value.width,
  height: verification.value.height,
}));

function syncVerification() {
  const meta = task.value?.result?.metadata?.training || {};
  const params = task.value?.params || {};
  if (!verifyPrompt.value) verifyPrompt.value = String(meta.training_caption || params.progress_prompt || '').trim();
  const recipe = verificationRecipe();
  verification.value = {
    ...verification.value,
    ...recipe,
    lora_scale: Number(recipe.lora_weight ?? verification.value.lora_scale),
    extra_adapters: Array.isArray(recipe.extra_adapters) ? recipe.extra_adapters.map(String) : [],
    seed: '42',
    width: Number(recipe.width || 1024),
    height: Number(recipe.height || 1024),
    model: String(recipe.model || params.base_model || 'z-image'),
  };
}

function verifyGenerate() {
  const checkpoint = selectedCheckpoint.value || 'final_adapters.safetensors';
  emit('verify', {
    ...verification.value,
    prompt: verifyPrompt.value.trim(),
    lora: verifyLoraId(),
    verify: '1',
    checkpoint,
  });
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
  stopCandidatePolling();
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

.lora-run-detail__verification {
  padding: 14px;
  border: 0.5px solid var(--dq-border-subtle);
  border-radius: var(--dq-radius-card);
  background: color-mix(in srgb, var(--dq-accent) 3%, var(--dq-fill-secondary));
}

.lora-run-detail__verify-hint,
.lora-run-detail__verify-recipe {
  margin: 0 0 12px;
  color: var(--dq-label-secondary);
  font-size: var(--dq-font-size-caption);
  line-height: 1.5;
}

.lora-run-detail__verify-recipe {
  margin: 12px 0 0;
  color: var(--dq-label-tertiary);
}

.lora-run-detail__verify-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.lora-run-detail__verify-grid label {
  display: flex;
  flex-direction: column;
  gap: 6px;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
}

.lora-run-detail__verify-grid label:first-child {
  grid-column: 1 / -1;
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
  align-items: center;
  gap: 10px;
  margin-top: 14px;
}
</style>
