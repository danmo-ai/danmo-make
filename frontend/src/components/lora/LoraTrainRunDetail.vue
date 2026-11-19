<template>
  <div class="lora-run-detail">
    <div class="lora-run-detail__head">
      <div class="lora-run-detail__head-main">
        <h3 class="lora-run-detail__title">{{ runDisplayName }}</h3>
        <DqTag v-if="task" :type="statusTagType">{{ statusLabel }}</DqTag>
        <TaskIdBadge :task-id="taskId" />
      </div>
    </div>

    <div v-if="task && !isTerminal" class="lora-run-detail__status">
      <DqProgress :percentage="Math.round((task.progress || 0) * 100)" />
      <p v-if="progressMessage" class="lora-run-detail__progress-msg">{{ progressMessage }}</p>
      <p v-if="task.step != null && task.total != null" class="lora-run-detail__step-hint">
        {{ task.step }} / {{ task.total }}
        <span v-if="task.eta_seconds != null">
          · {{ $t('loraTrain.etaSeconds', { s: Math.max(0, Math.round(task.eta_seconds)) }) }}
        </span>
      </p>
      <div class="lora-run-detail__queue-hint">
        <p>{{ $t('loraTrain.runInBackgroundHint') }}</p>
        <DqButton size="sm" type="secondary" @click="openGlobalTaskQueue">
          {{ $t('loraTrain.openTaskQueue') }}
        </DqButton>
      </div>
    </div>

    <p v-if="task?.status === 'failed' && task.error" class="lora-run-detail__error">
      {{ task.error }}
    </p>

    <ul v-if="task && isTerminal" class="lora-run-detail__summary">
      <li v-for="row in runSummaryRows" :key="row.key">
        <span>{{ row.label }}</span>
        <strong>{{ row.value }}</strong>
      </li>
    </ul>

    <div v-if="artifacts?.loss_history?.length" class="lora-run-detail__section">
      <h4 class="lora-run-detail__section-title">{{ $t('loraTrain.lossCurve') }}</h4>
      <p v-if="lossSummary" class="lora-run-detail__section-hint">{{ lossSummary }}</p>
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
      <h4 class="lora-run-detail__section-title">{{ $t('loraTrain.checkpoints') }}</h4>
      <p class="lora-run-detail__section-hint">{{ $t('loraTrain.registerHint') }}</p>
      <div v-if="checkpointOptions.length" class="lora-run-checkpoint-row">
        <DqSelect v-model="selectedCheckpoint" size="sm" class="lora-run-checkpoint-select">
          <DqOption
            v-for="ck in checkpointOptions"
            :key="ck"
            :label="ck"
            :value="ck"
          />
        </DqSelect>
        <DqInput v-model="registerName" :placeholder="$t('loraTrain.registerName')" size="sm" />
        <DqButton
          size="sm"
          type="primary"
          :loading="registering"
          :disabled="!selectedCheckpoint || Boolean(verifyLoraId())"
          @click="registerSelected"
        >
          {{ verifyLoraId() ? $t('loraTrain.registeredBadge') : $t('loraTrain.registerCheckpoint') }}
        </DqButton>
      </div>
      <DqAlert
        v-else
        type="warning"
        :closable="false"
        :title="$t('loraTrain.noCheckpointTitle')"
      >
        <p>{{ $t('loraTrain.noCheckpointBody') }}</p>
      </DqAlert>
      <p v-if="registeredLoraId" class="lora-run-registered">
        {{ $t('loraTrain.registeredAs') }}: {{ registeredLoraId }}
      </p>
    </div>

    <div v-if="task?.status === 'completed'" class="lora-run-detail__verification">
      <h4 class="lora-run-detail__section-title">{{ $t('loraTrain.verifyTitle') }}</h4>
      <p class="lora-run-detail__verify-hint">{{ $t('loraTrain.verifyHint') }}</p>
      <DqAlert v-if="!verifyLoraId()" type="warning" :closable="false" :title="$t('loraTrain.verifyNeedsRegistration')">
        <p>{{ $t('loraTrain.verifyCheckpointHint', { checkpoint: selectedCheckpoint || '—' }) }}</p>
      </DqAlert>
      <template v-else>
        <DqAlert type="success" :closable="false" :title="$t('loraTrain.verifyCheckpointBound', { checkpoint: selectedCheckpoint || 'final_adapters.safetensors' })" />
        <DqAlert
          v-if="verification.extra_adapters?.length"
          type="info"
          :closable="false"
          :title="$t('loraTrain.verifyExtraAdapters', { adapters: formatExtraAdapters(verification.extra_adapters) })"
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
          <DqButton type="primary" :disabled="!verifyPrompt.trim()" @click="verifyGenerate">
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
      </template>
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
import { stackedAdapterToken } from '@/utils/loraAdapterMeta';

type StackedAdapterItem = Parameters<typeof stackedAdapterToken>[0];

function formatExtraAdapters(adapters: unknown): string {
  if (!Array.isArray(adapters)) return '';
  return adapters
    .map((item) => stackedAdapterToken(item as StackedAdapterItem))
    .filter(Boolean)
    .join(', ');
}

const props = defineProps<{ taskId: string }>();
const emit = defineEmits<{
  (e: 'verify', payload: Record<string, unknown>): void;
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
const verifyPrompt = ref('');
const registeredRecipe = ref<Record<string, any> | null>(null);
const registeredCheckpoint = ref('');
const verification = ref<Record<string, any>>({ seed: '42', steps: 8, guidance: 0, lora_scale: 0.8, width: 1024, height: 1024, model: 'z-image-turbo', scheduler: 'linear' });
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

const runDisplayName = computed(() => {
  const result = task.value?.result || {};
  const params = task.value?.params || {};
  const name =
    result.output_name ||
    params.request?.output_name ||
    params.output_name ||
    '';
  return String(name).trim() || t('loraTrain.runTitle');
});

const checkpointOptions = computed(() => {
  const fromArtifacts = Array.isArray(artifacts.value?.checkpoints)
    ? artifacts.value.checkpoints.map((c: string) => checkpointBasename(c))
    : [];
  if (fromArtifacts.length) return [...new Set(fromArtifacts)];
  const fromResult = Array.isArray(task.value?.result?.checkpoints)
    ? task.value.result.checkpoints.map((c: string) => checkpointBasename(c))
    : [];
  if (fromResult.length) return [...new Set(fromResult)];
  if (task.value?.result?.adapter_path) {
    return [checkpointBasename(String(task.value.result.adapter_path))];
  }
  return [];
});

const runSummaryRows = computed(() => {
  const params = task.value?.params || {};
  const request = params.request || {};
  const preset = params.preset || {};
  const result = task.value?.result || {};
  const base = String(request.base_model || params.base_model || result.inference?.model || '—');
  const presetId = String(request.preset || preset.id || '—');
  const presetLabelKey = `loraTrain.preset.${presetId}`;
  const presetLabel = t(presetLabelKey);
  const rows = [
    { key: 'base', label: t('loraTrain.summaryBase'), value: base },
    {
      key: 'preset',
      label: t('loraTrain.summaryPreset'),
      value: presetLabel !== presetLabelKey ? presetLabel : presetId,
    },
    {
      key: 'dataset',
      label: t('loraTrain.summaryDataset'),
      value: String(request.dataset_id || params.dataset_id || '—'),
    },
    {
      key: 'images',
      label: t('loraTrain.summaryImages'),
      value: String(params.n_images ?? '—'),
    },
    {
      key: 'caption',
      label: t('loraTrain.summaryPrompt'),
      value: String(params.training_caption || request.default_prompt || '—'),
    },
  ];
  const res = Array.isArray(preset.resolution) ? preset.resolution.join('×') : '';
  if (res) {
    rows.push({ key: 'resolution', label: t('loraTrain.summaryResolution'), value: res });
  }
  if (preset.iterations != null) {
    rows.push({
      key: 'iterations',
      label: t('loraTrain.summaryIterations'),
      value: String(preset.iterations),
    });
  }
  const out = String(result.output_name || request.output_name || '').trim();
  if (out) {
    rows.push({ key: 'output', label: t('loraTrain.summaryOutput'), value: out });
  }
  return rows;
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

const lossSummary = computed(() => {
  const hist = artifacts.value?.loss_history;
  if (!Array.isArray(hist) || hist.length < 2) return '';
  const first = Number(hist[0]?.loss);
  const last = Number(hist[hist.length - 1]?.loss);
  if (!Number.isFinite(first) || !Number.isFinite(last)) return '';
  return t('loraTrain.lossSummary', {
    steps: hist.length,
    first: first.toFixed(3),
    last: last.toFixed(3),
  });
});

const canResumeTraining = computed(() => {
  const s = task.value?.status;
  if (s !== 'failed' && s !== 'cancelled') return false;
  return checkpointOptions.value.length > 0;
});

function checkpointBasename(path: string): string {
  const s = String(path || '');
  const parts = s.split(/[/\\]/);
  return parts[parts.length - 1] || s;
}

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

function hydrateArtifactsFromTaskResult() {
  const result = task.value?.result;
  if (!result || typeof result !== 'object') return;
  const checkpoints = Array.isArray(result.checkpoints)
    ? result.checkpoints.map((c: string) => checkpointBasename(c))
    : result.adapter_path
      ? [checkpointBasename(String(result.adapter_path))]
      : [];
  artifacts.value = {
    ...(artifacts.value || {}),
    loss_history: Array.isArray(result.loss_history)
      ? result.loss_history
      : artifacts.value?.loss_history,
    checkpoints: checkpoints.length ? checkpoints : artifacts.value?.checkpoints,
    inference: result.inference || artifacts.value?.inference,
    adapter_path: result.adapter_path || artifacts.value?.adapter_path,
    output_name: result.output_name || artifacts.value?.output_name,
    progress_images: artifacts.value?.progress_images || [],
  };
  if (result.user_lora_id && !registeredLoraId.value) {
    registeredLoraId.value = String(result.user_lora_id);
  }
  if (!selectedCheckpoint.value && checkpoints.length) {
    const finals = checkpoints.filter((c: string) => c.includes('final'));
    selectedCheckpoint.value = finals[finals.length - 1] || checkpoints[checkpoints.length - 1];
  }
  if (!registerName.value.trim()) {
    registerName.value = String(result.output_name || task.value?.params?.request?.output_name || '').trim();
  }
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
    if ((next as any)?.checkpoints?.length) {
      const names = (next as any).checkpoints.map((c: string) => checkpointBasename(c));
      artifacts.value = { ...(next as any), checkpoints: names };
      if (!selectedCheckpoint.value) {
        const finals = names.filter((c: string) => c.includes('final'));
        selectedCheckpoint.value = finals[finals.length - 1] || names.at(-1);
      }
    }
    syncVerification();
  } catch {
    hydrateArtifactsFromTaskResult();
    syncVerification();
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
  hydrateArtifactsFromTaskResult();
  await refreshArtifacts(true);
  syncVerification();
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
    String(
      task.value?.result?.user_lora_id ||
        task.value?.result?.metadata?.user_lora_id ||
        task.value?.result?.metadata?.training?.user_lora_id ||
        ''
    )
  );
}

function verificationRecipe(): Record<string, any> {
  const params = task.value?.params || {};
  const request = params.request || {};
  const registered = registeredCheckpoint.value === (selectedCheckpoint.value || 'final_adapters.safetensors');
  const recipe =
    (registered && registeredRecipe.value?.inference) ||
    artifacts.value?.inference ||
    task.value?.result?.inference ||
    params.preset?.inference ||
    params.metadata?.inference ||
    null;
  if (recipe && typeof recipe === 'object') return recipe;
  return {
    model: String(request.base_model || params.base_model || 'z-image-turbo'),
    steps: 8,
    guidance: 0,
    lora_weight: 0.8,
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
  const request = params.request || {};
  if (!verifyPrompt.value) {
    verifyPrompt.value = String(
      meta.training_caption || params.training_caption || ''
    ).trim();
  }
  const recipe = verificationRecipe();
  const res = Array.isArray(params.preset?.resolution) ? params.preset.resolution : null;
  verification.value = {
    ...verification.value,
    ...recipe,
    lora_scale: Number(recipe.lora_weight ?? verification.value.lora_scale ?? 0.8),
    extra_adapters: Array.isArray(recipe.extra_adapters)
      ? recipe.extra_adapters.map((item) => stackedAdapterToken(item as StackedAdapterItem)).filter(Boolean)
      : [],
    seed: String(verification.value.seed || '42'),
    width: Number(recipe.width || res?.[0] || 1024),
    height: Number(recipe.height || res?.[1] || 1024),
    model: String(recipe.model || request.base_model || params.base_model || 'z-image-turbo'),
    steps: Number(recipe.steps ?? verification.value.steps ?? 8),
    guidance: Number(recipe.guidance ?? verification.value.guidance ?? 0),
    scheduler: String(recipe.scheduler || 'linear'),
  };
}

function verifyGenerate() {
  const checkpoint = selectedCheckpoint.value || 'final_adapters.safetensors';
  // Hand off primary LoRA + compose knobs only. Companion adapters are read from
  // the registered LoRA's inference metadata at bind time — keep inference neutral.
  emit('verify', {
    prompt: verifyPrompt.value.trim(),
    lora: verifyLoraId(),
    model: verification.value.model,
    steps: verification.value.steps,
    guidance: verification.value.guidance,
    lora_scale: verification.value.lora_scale,
    scheduler: verification.value.scheduler,
    width: verification.value.width,
    height: verification.value.height,
    seed: verification.value.seed,
    verify: '1',
    checkpoint,
  });
}

async function registerSelected() {
  if (!selectedCheckpoint.value) return;
  registering.value = true;
  try {
    const res = (await api.loras.registerCheckpoint(props.taskId, {
      checkpoint: selectedCheckpoint.value,
      name: registerName.value.trim(),
    })) as { user_lora?: Record<string, any> };
    registeredLoraId.value = String(res.user_lora?.id || '');
    registeredCheckpoint.value = selectedCheckpoint.value;
    registeredRecipe.value = res.user_lora || null;
    syncVerification();
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
  gap: 18px;
  overflow: auto;
}

.lora-run-detail__head {
  display: flex;
  align-items: flex-start;
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

.lora-run-detail__summary {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
}

.lora-run-detail__summary li {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  padding: 10px 0;
  border-bottom: 0.5px solid var(--dq-border-subtle);
  font-size: var(--dq-font-size-body);
}

.lora-run-detail__summary li:last-child {
  border-bottom: none;
}

.lora-run-detail__summary span {
  color: var(--dq-label-tertiary);
  flex-shrink: 0;
}

.lora-run-detail__summary strong {
  color: var(--dq-label-primary);
  font-weight: 600;
  text-align: right;
  word-break: break-word;
  min-width: 0;
}

.lora-run-detail__section-hint {
  margin: 0 0 10px;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
  line-height: 1.45;
}

.lora-run-detail__verification {
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding-top: 4px;
  border-top: 0.5px solid var(--dq-border-subtle);
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
  flex-wrap: wrap;
  align-items: center;
  gap: 10px 14px;
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
  padding: 4px 0;
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
