<template>
  <WorkspaceLayout class="copilot-page lora-train-page">
    <template #sidebar>
      <WorkspaceSidebar :title="$t('loraTrain.recentRuns')">
        <template #icon><DqIcon><MagicStick /></DqIcon></template>
        <DqButton
          type="primary"
          block
          class="lv-sidebar-card__new-btn"
          :disabled="trainingUnavailable"
          @click="startNewTraining"
        >
          + {{ $t('loraTrain.newTraining') }}
        </DqButton>
        <div class="lora-train-page__sidebar-body">
          <LoraTrainHistory
            ref="historyRef"
            :active-id="activeRunId"
            variant="sidebar"
            hide-header
            hide-refresh
            @select="openRun"
          />
        </div>
      </WorkspaceSidebar>
    </template>

    <div class="copilot-page__main">
      <DqSurfaceCard
        v-if="trainingUnavailable && !activeRunId"
        class="copilot-page__workspace-card studio-surface-card lora-train-page__unavailable"
      >
        <h2>{{ $t('loraTrain.unavailableTitle') }}</h2>
        <p>{{ $t('loraTrain.unavailableBody') }}</p>
        <p v-if="requirements?.detail" class="lora-train-page__unavailable-detail">
          {{ requirements.detail }}
        </p>
      </DqSurfaceCard>

      <DqSurfaceCard
        v-else-if="activeRunId"
        class="copilot-page__workspace-card studio-surface-card"
      >
        <LoraTrainRunDetail
          :task-id="activeRunId"
          @verify="onVerifyGenerate"
          @deleted="onRunDeleted"
        />
      </DqSurfaceCard>

      <template v-else>
        <div class="lv-editor-tabs-wrapper">
          <div class="lv-editor-tabs lora-train-page__stepper" role="tablist">
            <template v-for="(item, i) in stepNavItems" :key="item.key">
              <span v-if="i > 0" class="lv-editor-tabs__connector" aria-hidden="true" />
              <button
                type="button"
                role="tab"
                class="lv-editor-tabs__tab"
                :class="{
                  'lv-editor-tabs__tab--active': step === i,
                  'lv-editor-tabs__tab--done': step > i || stepDone(i),
                  'lv-editor-tabs__tab--locked': i > maxReachableStep,
                }"
                :aria-selected="step === i"
                @click="goToStep(i)"
              >
                <span class="lv-editor-tabs__icon" :class="{ 'lv-editor-tabs__icon--done': step > i || stepDone(i) }" aria-hidden="true">
                  <span v-if="step > i || stepDone(i)" class="lv-editor-tabs__icon-mark">✓</span>
                  <span v-else class="lv-editor-tabs__icon-mark">{{ i + 1 }}</span>
                </span>
                <span class="lv-editor-tabs__text">
                  <span class="lv-editor-tabs__label">{{ item.label }}</span>
                </span>
              </button>
            </template>
          </div>
        </div>

        <DqSurfaceCard class="copilot-page__workspace-card studio-surface-card">
          <!-- Step 1: base model -->
          <div v-show="step === 0" class="lora-train-page__panel lora-train-page__panel--base">
            <div class="lora-train-page__face-intro">
              <h2>{{ $t('loraTrain.faceWorkflowTitle') }}</h2>
              <p>{{ $t('loraTrain.faceWorkflowBody') }}</p>
            </div>
            <div class="lora-train-page__base-grid">
              <button
                v-for="m in trainableModels"
                :key="m.id"
                type="button"
                class="lora-train-page__base-tile"
                :class="{
                  'is-selected': form.base_model === m.id,
                  'is-ready': m.trainable && m.ready,
                  [`is-family-${modelFamilyKey(m.id)}`]: true,
                }"
                :disabled="!m.trainable || !m.ready"
                :aria-pressed="form.base_model === m.id"
                @click="selectBase(m)"
              >
                <span class="lora-train-page__base-tile-icon" aria-hidden="true">
                  {{ modelInitials(m) }}
                </span>
                <span class="lora-train-page__base-tile-body">
                  <span class="lora-train-page__base-tile-name">
                    {{ modelDisplayName(m) }}
                    <em v-if="m.id === 'z-image-turbo'" class="lora-train-page__preset-badge">{{ $t('loraTrain.presetRecommended') }}</em>
                  </span>
                  <span v-if="modelShortDesc(m)" class="lora-train-page__base-tile-desc">
                    {{ modelShortDesc(m) }}
                  </span>
                </span>
                <span
                  v-if="form.base_model === m.id"
                  class="lora-train-page__base-tile-check"
                  aria-hidden="true"
                />
              </button>
            </div>
          </div>
          <div v-if="step === 0" class="lora-train-page__nav-bar">
            <span />
            <DqButton
              type="primary"
              :disabled="!canNext(0)"
              @click="goToStep(1)"
            >
              {{ $t('common.next') }}
            </DqButton>
          </div>

          <!-- Step 2: dataset -->
          <div v-show="step === 1" class="lora-train-page__panel lora-train-page__panel--flush">
            <LoraDatasetPanel
              ref="datasetPanelRef"
              :selected-id="form.dataset_id"
              :datasets="datasets"
              :default-prompt="form.default_prompt"
              :dataset-health="datasetHealth"
              :dataset-health-stale="datasetHealthStale || datasetHealth?.faces?.resolution?.join('x') !== activeResolutionKey"
              :dataset-health-loading="datasetHealthLoading"
              :vision-available="visionAvailable"
              :dataset-vlm-loading="datasetVlmLoading"
              @update:selected-id="form.dataset_id = $event"
              @update:default-prompt="form.default_prompt = $event"
              @datasets-changed="onDatasetsChanged"
              @preflight-requested="onFacePreflightRequested"
              @run-dataset-vlm="runDatasetVlmAudit"
            />
          </div>
          <div v-if="step === 1" class="lora-train-page__nav-bar">
            <DqButton size="sm" @click="goToStep(0)">
              {{ $t('common.back') }}
            </DqButton>
            <DqButton
              type="primary"
              :loading="datasetHealthLoading"
              :disabled="!faceDatasetReady"
              @click="goToNextFromDataset"
            >
              {{ $t('common.next') }}
            </DqButton>
          </div>

          <!-- Step 3: config -->
          <div v-show="step === 2" class="lora-train-page__panel">
            <DqAlert
              v-if="hasResume"
              type="warning"
              :closable="false"
              class="lora-train-page__resume-banner"
              :title="$t('loraTrain.resumeActiveTitle')"
            >
              <p>{{ $t('loraTrain.resumeActiveBody', {
                task: form.resume_task_id,
                checkpoint: form.resume_checkpoint,
                base: form.base_model,
              }) }}</p>
              <DqButton size="sm" type="secondary" @click="clearResume">
                {{ $t('loraTrain.clearResume') }}
              </DqButton>
            </DqAlert>
            <div class="lora-train-page__preset-grid">
              <button
                v-for="p in presetKeys"
                :key="p"
                type="button"
                class="lora-train-page__preset-card"
                :class="{
                  'is-selected': form.preset === p,
                  'is-recommended': p === recommendedPreset,
                }"
                @click="form.preset = p"
              >
                <span class="lora-train-page__preset-name">
                  {{ $t(`loraTrain.preset.${p}`) }}
                  <em
                    v-if="p === recommendedPreset"
                    class="lora-train-page__preset-badge"
                  >{{ $t('loraTrain.presetRecommended') }}</em>
                </span>
                <span class="lora-train-page__preset-desc">{{ $t(`loraTrain.presetDesc.${p}`) }}</span>
                <span v-if="p !== 'custom' && presetStats(p)" class="lora-train-page__preset-stats">
                  {{ presetStats(p) }}
                  <template v-if="presetCardResolution(p)"> · {{ presetCardResolution(p) }}</template>
                </span>
              </button>
            </div>
            <p
              v-if="form.base_model === 'z-image-turbo' && form.preset === 'turbo-sft'"
              class="lora-train-page__recipe-note"
            >
              {{ $t('loraTrain.turboSftInferHint') }}
            </p>

            <div class="lora-train-page__essentials">
            <div class="lora-train-page__field">
              <label class="lora-train-page__label">{{ $t('loraTrain.outputName') }}</label>
              <DqInput v-model="form.output_name" :placeholder="$t('loraTrain.outputNamePlaceholder')" />
              <p class="lora-train-page__field-hint">{{ $t('loraTrain.outputNameDesc') }}</p>
            </div>
            </div>
          </div>
          <div v-if="step === 2" class="lora-train-page__nav-bar">
            <DqButton size="sm" @click="goToStep(1)">
              {{ $t('common.back') }}
            </DqButton>
            <DqButton
              type="primary"
              :disabled="!canNext(2)"
              @click="goToStep(3)"
            >
              {{ $t('common.next') }}
            </DqButton>
          </div>

          <!-- Step 4: confirm -->
          <div v-show="step === 3" class="lora-train-page__panel">
            <DqAlert
              v-if="hasResume"
              type="warning"
              :closable="false"
              class="lora-train-page__resume-banner"
              :title="$t('loraTrain.resumeConfirmWarning')"
            >
              <p>{{ $t('loraTrain.resumeActiveBody', {
                task: form.resume_task_id,
                checkpoint: form.resume_checkpoint,
                base: form.base_model,
              }) }}</p>
              <DqButton size="sm" type="secondary" @click="clearResume">
                {{ $t('loraTrain.clearResume') }}
              </DqButton>
            </DqAlert>
            <p class="lora-train-page__confirm-intro">{{ $t('loraTrain.confirmIntro') }}</p>
            <div class="lora-train-page__review" :class="{ 'has-preview': confirmPreviewImages.length }">
            <div class="lora-train-page__review-main">
            <ul v-if="isIdentityDataset" class="lora-train-page__identity-checklist">
              <li
                v-for="item in identityConfirmChecks"
                :key="item.key"
                :class="item.ok ? 'is-ok' : 'is-bad'"
              >
                <span class="lora-train-page__identity-check-mark" aria-hidden="true">{{ item.ok ? '✓' : '!' }}</span>
                <span>{{ item.label }}</span>
              </li>
            </ul>
            <LoraQualityHints
              v-if="datasetHealth"
              :report="datasetHealth"
              mode="dataset"
              fold
              :vision-available="visionAvailable"
              :vlm-loading="datasetVlmLoading"
              :dataset-audit-kind="selectedDataset?.kind === 'style' ? 'style' : 'concept'"
              class="lora-train-page__confirm-health"
              @run-vlm="runDatasetVlmAudit"
            />
            <ul class="lora-train-page__summary">
              <li><span>{{ $t('loraTrain.summaryBase') }}</span><strong>{{ form.base_model }}</strong></li>
              <li>
                <span>{{ $t('loraTrain.summaryDataset') }}</span>
                <strong>{{ selectedDataset?.name }} ({{ selectedDataset?.training_image_count ?? trainingImages.length }})</strong>
              </li>
              <li><span>{{ $t('loraTrain.summaryPreset') }}</span><strong>{{ $t(`loraTrain.preset.${form.preset}`) }}</strong></li>
              <li v-if="effectiveDatasetCaption">
                <span>{{ $t('loraTrain.summaryPrompt') }}</span>
                <strong>{{ effectiveDatasetCaption }}</strong>
              </li>
              <li v-if="datasetHealth?.faces?.resolution">
                <span>{{ $t('loraTrain.facePreflightSummary') }}</span>
                <strong>{{ datasetHealth.faces.resolution.join('×') }}</strong>
              </li>
              <li v-if="form.output_name.trim()">
                <span>{{ $t('loraTrain.summaryOutput') }}</span><strong>{{ form.output_name }}</strong>
              </li>
              <li v-if="hasResume">
                <span>{{ $t('loraTrain.summaryResume') }}</span>
                <strong>{{ form.resume_task_id }} / {{ form.resume_checkpoint }}</strong>
              </li>
            </ul>
            <p
              v-if="form.base_model === 'z-image-turbo' && form.preset === 'turbo-sft'"
              class="lora-train-page__recipe-note"
            >
              {{ $t('loraTrain.turboSftInferHint') }}
            </p>
            </div>
            <div v-if="confirmPreviewImages.length" class="lora-train-page__confirm-preview">
              <span class="lora-train-page__confirm-preview-label">{{ $t('loraTrain.confirmPreview') }}</span>
              <div class="lora-train-page__confirm-preview-grid">
                <img
                  v-for="img in confirmPreviewImages"
                  :key="img.file"
                  :src="confirmImageUrl(img.file)"
                  :alt="img.file"
                  loading="lazy"
                />
                <span
                  v-if="trainingImages.length > confirmPreviewImages.length"
                  class="lora-train-page__confirm-preview-more"
                >
                  +{{ trainingImages.length - confirmPreviewImages.length }}
                </span>
              </div>
            </div>
            </div>
          </div>
          <div v-if="step === 3" class="lora-train-page__nav-bar">
            <DqButton size="sm" @click="goToStep(2)">
              {{ $t('common.back') }}
            </DqButton>
            <DqButton
              type="primary"
              :loading="submitting"
              :disabled="!canSubmit"
              @click="submitTraining"
            >
              {{ $t('loraTrain.startTraining') }}
            </DqButton>
          </div>
        </DqSurfaceCard>
      </template>
    </div>
  </WorkspaceLayout>
</template>

<script setup lang="ts">
import { computed, nextTick, onMounted, reactive, ref, watch } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import { useI18n } from 'vue-i18n';
import { MagicStick } from '@danqing/dq-shell';
import { api, taskIdFromSubmitResponse } from '@/utils/api';
import { toast } from '@/utils/feedback';
import { openGlobalTaskQueue } from '@/utils/appEvents';
import { getItem, setItem, DQ_STORAGE } from '@/utils/storage';
import LoraTrainRunDetail from '@/components/lora/LoraTrainRunDetail.vue';
import LoraDatasetPanel from '@/components/lora/LoraDatasetPanel.vue';
import LoraTrainHistory from '@/components/lora/LoraTrainHistory.vue';
import LoraQualityHints from '@/components/lora/LoraQualityHints.vue';
import WorkspaceLayout from '@/components/workspace/WorkspaceLayout.vue';
import WorkspaceSidebar from '@/components/workspace/WorkspaceSidebar.vue';
import { useLoraTrainLibrary } from '@/composables/useLoraTrainLibrary';
import { useRegistryStore } from '@/stores/registry';
import type { LoraDatasetHealthReport } from '@/utils/loraQuality';
import '@/styles/editor-tabs.css';

const { t, te, locale } = useI18n();
const router = useRouter();
const route = useRoute();
const registryStore = useRegistryStore();

const step = ref(0);
const submitting = ref(false);
const activeRunId = ref('');
const resumeIntent = ref(false);
const applyingResumeSync = ref(false);
const trainableModels = ref<any[]>([]);
const datasets = ref<any[]>([]);
const datasetHealth = ref<LoraDatasetHealthReport | null>(null);
const datasetHealthStale = ref(true);
const datasetHealthLoading = ref(false);
let datasetHealthRequestSeq = 0;
const datasetVlmLoading = ref(false);
const visionAvailable = ref(false);
const requirements = ref<any>(null);
const presetsByModel = ref<Record<string, Record<string, Record<string, unknown>>>>({});
const datasetPanelRef = ref<InstanceType<typeof LoraDatasetPanel> | null>(null);
const historyRef = ref<InstanceType<typeof LoraTrainHistory> | null>(null);
const { loading: historyLoading, refresh: refreshHistory } = useLoraTrainLibrary();
/** Gate on GET /api/loras/training/requirements.ready (MLX build). */
const trainingUnavailable = computed(
  () => requirements.value == null || requirements.value.ready === false,
);
function recommendedPresetFor(model: string) {
  if (model === 'z-image-turbo') return 'turbo-sft';
  if (model === 'z-image') return 'base-face';
  return 'turbo-sft';
}

const recommendedPreset = computed(() => recommendedPresetFor(form.base_model));

const NAMED_PRESETS = new Set(['turbo-sft', 'turbo-diff', 'base-face']);

const presetKeys = computed(() => {
  const fromApi = Object.keys(modelPresets.value || {}).filter((k) => NAMED_PRESETS.has(k));
  const keys = fromApi.length
    ? [
        ...fromApi.filter((k) => k === recommendedPreset.value),
        ...fromApi.filter((k) => k !== recommendedPreset.value),
      ]
    : [recommendedPreset.value];
  return keys;
});

function coerceVisiblePreset() {
  if (!presetKeys.value.includes(form.preset)) form.preset = recommendedPreset.value;
}

function presetCardResolution(preset: string): string {
  const cfg = modelPresets.value[preset] as Record<string, unknown> | undefined;
  return cfg ? presetResolution(cfg) : '';
}

const stepNavItems = computed(() => [
  { key: 'base', label: t('loraTrain.faceStepBase') },
  { key: 'dataset', label: t('loraTrain.faceStepDataset') },
  { key: 'config', label: t('loraTrain.faceStepConfig') },
  { key: 'confirm', label: t('loraTrain.faceStepConfirm') },
]);

const form = reactive({
  base_model: 'z-image-turbo',
  dataset_id: '',
  preset: 'turbo-sft' as string,
  default_prompt: '',
  output_name: '',
  resume_task_id: '',
  resume_checkpoint: '',
});

const selectedDataset = computed(() =>
  datasets.value.find((d) => d.id === form.dataset_id)
);
const isIdentityDataset = computed(() => String(selectedDataset.value?.kind || 'concept') !== 'style');
const skippedTrainingFiles = computed(() => new Set<string>((selectedDataset.value?.skipped_images || []).map(String)));
const trainingImages = computed(() =>
  (selectedDataset.value?.images || []).filter((img: { file: string }) => !skippedTrainingFiles.value.has(img.file))
);

function faceAuditRow(file: string) {
  const rows = datasetHealth.value?.faces?.rows || [];
  const base = file.split('/').pop() || file;
  return rows.find((row: { file: string }) => row.file === file || row.file === base || file.endsWith(`/${row.file}`));
}

function faceRowBlocksTraining(row: { action?: string; multi?: boolean } | undefined): boolean {
  if (!row) return true;
  return row.action === 'tiny' || row.action === 'none' || row.action === 'error' || row.action === 'multi' || !!row.multi;
}

const structuralHealthOk = computed(() => {
  if (!datasetHealth.value || datasetHealthStale.value) return false;
  const level = String(datasetHealth.value.level || '');
  if (level === 'error') return false;
  const faces = datasetHealth.value.faces;
  // When face detector is not wired, accept structural health only.
  if (!faces?.available) return true;
  if ((faces.resolution || []).join('x') !== activeResolutionKey.value) return false;
  return !trainingImages.value.some((img: { file: string }) => faceRowBlocksTraining(faceAuditRow(img.file)));
});

const faceDatasetReady = computed(() => {
  if (trainingImages.value.length < 3) return false;
  if (!isIdentityDataset.value) return true;
  const trigger = String(selectedDataset.value?.trigger_word || '').trim();
  if (!trigger) return false;
  if (!structuralHealthOk.value) return false;
  return triggerBound.value;
});

const triggerBound = computed(() => {
  const trigger = String(selectedDataset.value?.trigger_word || '').trim();
  if (!trigger || trainingImages.value.length < 1) return false;
  return effectiveDatasetCaption.value.includes(trigger);
});

function activeTrainingResolution(): [number, number] {
  const res = activePresetDetail.value?.resolution;
  if (Array.isArray(res) && res.length >= 2) return [Number(res[0]), Number(res[1])];
  return [1024, 1024];
}
const activeResolutionKey = computed(() => activeTrainingResolution().join('x'));

const modelPresets = computed(
  () => presetsByModel.value[form.base_model] || {}
);

const activePresetDetail = computed(() => {
  return (modelPresets.value[form.preset] || null) as Record<string, unknown> | null;
});

const confirmPreviewImages = computed(() => trainingImages.value.slice(0, 8));

const effectiveDatasetCaption = computed(() => {
  // form.default_prompt is the live editor value; datasets[] may lag until blur save.
  if (form.dataset_id) {
    const fromForm = form.default_prompt.trim();
    if (fromForm) return fromForm;
  }
  return String(selectedDataset.value?.default_prompt || '').trim();
});

const wizardSummary = computed(() => {
  const rows: Array<{ key: string; label: string; value: string }> = [];
  if (form.base_model) {
    const m = trainableModels.value.find((x) => x.id === form.base_model);
    rows.push({ key: 'base', label: t('loraTrain.summaryBase'), value: modelDisplayName(m || { id: form.base_model }) });
  }
  if (selectedDataset.value) {
    rows.push({
      key: 'dataset',
      label: t('loraTrain.summaryDataset'),
      value: `${selectedDataset.value.name} (${selectedDataset.value.training_image_count ?? trainingImages.value.length ?? selectedDataset.value.image_count ?? 0})`,
    });
  }
  if (form.preset && step.value >= 2) {
    rows.push({
      key: 'preset',
      label: t('loraTrain.summaryPreset'),
      value: t(`loraTrain.preset.${form.preset}`),
    });
  }
  return rows;
});

function stepDone(i: number): boolean {
  if (i === 0) return !!form.base_model && step.value > 0;
  if (i === 1) return faceDatasetReady.value && step.value > 1;
  if (i === 2) {
    return (
      !!form.preset
      && !!effectiveDatasetCaption.value
      && (!isIdentityDataset.value || triggerBound.value)
      && step.value > 2
    );
  }
  return false;
}

const identityConfirmChecks = computed(() => {
  const faces = datasetHealth.value?.faces;
  const trigger = String(selectedDataset.value?.trigger_word || '').trim();
  const imageCount = trainingImages.value.length;
  const usable = trainingImages.value.filter((img: { file: string }) => {
    const row = faceAuditRow(img.file);
    return !!row && !faceRowBlocksTraining(row);
  }).length;
  const blocked = imageCount - usable;
  return [
    {
      key: 'images',
      ok: imageCount >= 3,
      label: t('loraTrain.identityCheckImages', { count: imageCount }),
    },
    {
      key: 'trigger',
      ok: triggerBound.value,
      label: trigger
        ? (triggerBound.value
          ? t('loraTrain.identityCheckTrigger', { trigger })
          : t('loraTrain.triggerCoverageMissing', { trigger }))
        : t('loraTrain.identityCheckTriggerMissing'),
    },
    {
      key: 'faces',
      ok: faces?.available
        ? blocked === 0 && imageCount >= 3 && !datasetHealthStale.value
        : structuralHealthOk.value && imageCount >= 3,
      label: faces?.available
        ? t('loraTrain.identityCheckFaces', {
            ok: usable,
            total: imageCount,
          })
        : t('loraTrain.identityCheckStructural'),
    },
    {
      key: 'recipe',
      ok: !!form.preset && !!effectiveDatasetCaption.value && (!trigger || triggerBound.value),
      label: t('loraTrain.identityCheckRecipe', {
        preset: t(`loraTrain.preset.${form.preset}`),
        resolution: activeResolutionKey.value.replace('x', '×'),
      }),
    },
  ];
});

function presetStats(p: string): string {
  const cfg = modelPresets.value[p] as Record<string, unknown> | undefined;
  if (!cfg) return '';
  const it = cfg.iterations;
  const rank = cfg.lora_rank;
  if (it == null || rank == null) return '';
  return t('loraTrain.presetStatsShort', { iterations: it, rank });
}

function presetResolution(cfg: Record<string, unknown> | null): string {
  const res = cfg?.resolution;
  if (!Array.isArray(res) || res.length < 2) return '';
  return `${res[0]}×${res[1]}`;
}

function confirmImageUrl(file: string): string {
  if (!form.dataset_id || !file) return '';
  return api.loras.datasetImageUrl(form.dataset_id, file);
}

const maxReachableStep = computed(() => {
  let max = 0;
  if (form.base_model) max = 1;
  if (faceDatasetReady.value) max = Math.max(max, 2);
  if (form.preset && effectiveDatasetCaption.value) max = Math.max(max, 3);
  return max;
});

function canNext(stepIndex: number): boolean {
  if (stepIndex === 0) return !!form.base_model;
  if (stepIndex === 1) return faceDatasetReady.value;
  if (stepIndex === 2) {
    return (
      !!form.preset
      && !!effectiveDatasetCaption.value
      && (!isIdentityDataset.value || triggerBound.value)
    );
  }
  return true;
}

const canSubmit = computed(() => {
  if (!faceDatasetReady.value || !canNext(2) || requirements.value?.can_submit === false) return false;
  if (isIdentityDataset.value && identityConfirmChecks.value.some((item) => !item.ok)) return false;
  return true;
});

const hasResume = computed(
  () => !!form.resume_task_id.trim() && !!form.resume_checkpoint.trim()
);

function clearResume() {
  form.resume_task_id = '';
  form.resume_checkpoint = '';
  resumeIntent.value = false;
  persistDraft();
}

function goToStep(i: number) {
  if (i > maxReachableStep.value) return;
  step.value = i;
}

function openRun(taskId: string) {
  if (!taskId) return;
  activeRunId.value = taskId;
}

function onRunDeleted(taskId: string) {
  if (activeRunId.value === taskId) activeRunId.value = '';
  historyRef.value?.refresh();
}

function modelDisplayName(m: any): string {
  const n = m.name;
  if (n && typeof n === 'object') return n.en || n.zh || m.id;
  return m.id;
}

function modelInitials(m: any): string {
  const name = modelDisplayName(m);
  const parts = name.replace(/[\[\]()]/g, ' ').split(/\s+/).filter(Boolean);
  if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
  return name.slice(0, 2).toUpperCase();
}

function modelFamilyKey(id: string): string {
  if (id.startsWith('flux')) return 'flux';
  if (id.startsWith('z-image')) return 'zimage';
  if (id.startsWith('qwen')) return 'qwen';
  return 'default';
}

function modelShortDesc(m: { id?: string }): string {
  const id = m.id || '';
  const key = `loraTrain.baseDesc.${id}`;
  if (id && te(key)) return t(key);
  const reg = registryStore.registry?.models?.[id];
  const desc = reg?.description;
  if (!desc || typeof desc !== 'object') return '';
  const loc = String(locale.value || 'en').startsWith('zh') ? 'zh' : 'en';
  const text = String((desc as Record<string, string>)[loc] || desc.en || desc.zh || '').trim();
  if (!text) return '';
  return text.length > 72 ? `${text.slice(0, 69)}…` : text;
}

function apiErrorMessage(e: unknown): string {
  const err = e as { response?: { data?: { detail?: { message?: string } | string } }; message?: string };
  const detail = err?.response?.data?.detail;
  if (detail && typeof detail === 'object' && detail.message) return detail.message;
  if (typeof detail === 'string') return detail;
  return err?.message || String(e);
}

async function refreshDatasetHealth(id?: string) {
  const dsId = (id || form.dataset_id || '').trim();
  const resolution = activeTrainingResolution();
  const requestSeq = ++datasetHealthRequestSeq;
  if (!dsId) {
    datasetHealth.value = null;
    datasetHealthStale.value = true;
    datasetHealthLoading.value = false;
    return;
  }
  datasetHealthLoading.value = true;
  try {
    const report = (await api.loras.datasetHealth(dsId, resolution)) as LoraDatasetHealthReport;
    if (requestSeq !== datasetHealthRequestSeq || dsId !== form.dataset_id) return;
    datasetHealth.value = report;
    datasetHealthStale.value = report.faces?.resolution?.join('x') !== activeResolutionKey.value;
    if (datasetHealth.value?.vision_available != null) {
      visionAvailable.value = Boolean(datasetHealth.value.vision_available);
    }
  } catch (e: unknown) {
    if (requestSeq !== datasetHealthRequestSeq) return;
    datasetHealth.value = null;
    datasetHealthStale.value = true;
    toast.error(apiErrorMessage(e));
  } finally {
    if (requestSeq === datasetHealthRequestSeq) datasetHealthLoading.value = false;
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

async function runDatasetVlmAudit() {
  if (!form.dataset_id || datasetVlmLoading.value) return;
  datasetVlmLoading.value = true;
  try {
    const kind = selectedDataset.value?.kind === 'style' ? 'style' : 'concept';
    const [width, height] = activeTrainingResolution();
    datasetHealth.value = (await api.loras.datasetHealthVlm(form.dataset_id, {
      auditKind: kind,
      resolution: [width, height],
    })) as LoraDatasetHealthReport;
    datasetHealthStale.value = datasetHealth.value.faces?.resolution?.join('x') !== activeResolutionKey.value;
    toast.success(t('loraTrain.quality.vlmAuditDone'));
  } catch (e: unknown) {
    toast.error(apiErrorMessage(e));
  } finally {
    datasetVlmLoading.value = false;
  }
}

function onDatasetsChanged(next: any[]) {
  const before = selectedDataset.value?.images || [];
  const afterDs = next.find((d) => d.id === form.dataset_id);
  const after = afterDs?.images || [];
  const changedImages = before.length !== after.length || before.some((img: { file: string; prompt?: string }, index: number) =>
    img.file !== after[index]?.file || String(img.prompt || '') !== String(after[index]?.prompt || '')
  );
  const skipChanged = JSON.stringify(selectedDataset.value?.skipped_images || []) !== JSON.stringify(afterDs?.skipped_images || []);
  datasets.value = next;
  // Renames/default-template edits do not invalidate pixels, but image/caption edits do.
  // Skip only changes which stored files are trained; reuse the face audit and reapply the overlay.
  if (changedImages) datasetHealthStale.value = true;
  else if (skipChanged) void refreshDatasetHealth(form.dataset_id);
}

async function onFacePreflightRequested() {
  await refreshDatasetHealth(form.dataset_id);
  if (!faceDatasetReady.value) toast.warning(t('loraTrain.faceDatasetGate'));
}

function selectBase(m: any) {
  if (!m.trainable || !m.ready) return;
  form.base_model = m.id;
}

async function loadMeta() {
  const [modelsRes, dsRes, reqRes] = await Promise.all([
    api.loras.trainableModels(),
    api.loras.listDatasets(),
    api.loras.trainingRequirements(form.base_model),
  ]);
  trainableModels.value = (modelsRes as any).items || [];
  presetsByModel.value = (modelsRes as any).presets_by_model || {};
  datasets.value = (dsRes as any).items || [];
  requirements.value = reqRes;
  coerceVisiblePreset();
}

async function refreshRequirements() {
  try {
    requirements.value = await api.loras.trainingRequirements(form.base_model);
  } catch {
    // ignore
  }
}

function buildTrainingBody(): Record<string, unknown> {
  const body: Record<string, unknown> = {
    base_model: form.base_model,
    dataset_id: form.dataset_id,
    preset: form.preset,
    output_name: form.output_name,
    auto_register: true,
    face_crop: isIdentityDataset.value ? 'auto' : 'off',
  };

  const resumeTask = form.resume_task_id.trim();
  const resumeCk = form.resume_checkpoint.trim();
  if (resumeTask && resumeCk) {
    body.resume_task_id = resumeTask;
    body.resume_checkpoint = resumeCk;
  }
  return body;
}

function restoreDraft() {
  try {
    const raw = getItem(DQ_STORAGE.LORA_TRAIN_DRAFT);
    if (!raw) return;
    const draft = JSON.parse(raw) as Partial<typeof form> & { step?: number };
    if (draft.base_model) form.base_model = draft.base_model;
    if (draft.dataset_id) form.dataset_id = draft.dataset_id;
    if (draft.preset) form.preset = draft.preset;
    if (draft.default_prompt) form.default_prompt = draft.default_prompt;
    if (draft.output_name) form.output_name = draft.output_name;
    // Never restore resume fields from draft — they leak across unrelated new runs.
    form.resume_task_id = '';
    form.resume_checkpoint = '';
    resumeIntent.value = false;
    coerceVisiblePreset();
    if (typeof draft.step === 'number' && draft.step >= 0 && draft.step <= 3) {
      step.value = Math.min(draft.step, maxReachableStep.value);
    }
  } catch {
    // ignore corrupt draft
  }
}

function persistDraft() {
  setItem(
    DQ_STORAGE.LORA_TRAIN_DRAFT,
    JSON.stringify({
      base_model: form.base_model,
      dataset_id: form.dataset_id,
      preset: form.preset,
      default_prompt: form.default_prompt,
      output_name: form.output_name,
      step: step.value,
    })
  );
}

watch(
  () => form.dataset_id,
  async (id) => {
    datasetHealthStale.value = true;
    if (!id) {
      datasetHealth.value = null;
      return;
    }
    void refreshDatasetHealth(id);
    try {
      const ds = await api.loras.getDataset(id);
      datasets.value = datasets.value.map((d) => (d.id === id ? { ...d, ...(ds as Record<string, unknown>) } : d));
      const metaPrompt = String((ds as Record<string, unknown>).default_prompt || '').trim();
      if (metaPrompt) {
        form.default_prompt = metaPrompt;
      }
    } catch (e: unknown) {
      toast.error(apiErrorMessage(e));
    }
  }
);

async function importFromRouteQuery() {
  const raw = route.query.import;
  const ids = (Array.isArray(raw) ? raw.join(',') : String(raw || ''))
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  if (!ids.length) return;

  const explicitDatasetId = String(route.query.dataset_id || '').trim();
  const datasetName =
    String(route.query.dataset_name || '').trim() || t('loraTrain.canvasImportDataset');

  if (explicitDatasetId) {
    form.dataset_id = explicitDatasetId;
    if (!datasets.value.some((d) => d.id === explicitDatasetId)) {
      try {
        const ds = await api.loras.getDataset(explicitDatasetId);
        datasets.value = [ds as Record<string, unknown>, ...datasets.value];
      } catch (e: unknown) {
        toast.error(apiErrorMessage(e));
        return;
      }
    }
  } else if (datasetPanelRef.value) {
    await datasetPanelRef.value.createDatasetWithName(datasetName);
  } else {
    const ds = (await api.loras.createDataset({
      name: datasetName,
      default_prompt: datasetName,
    })) as Record<string, any>;
    form.default_prompt = datasetName;
    datasets.value.unshift(ds);
    form.dataset_id = ds.id;
  }

  if (!form.dataset_id) return;
  try {
    const ds = await api.loras.importAssets(form.dataset_id, ids, form.default_prompt);
    datasets.value = datasets.value.map((d) =>
      d.id === form.dataset_id ? { ...d, ...(ds as object) } : d
    );
    step.value = 1;
    toast.success(t('loraTrain.galleryImported'));
    await refreshDatasetHealth(form.dataset_id);
    const q = { ...route.query };
    delete q.import;
    delete q.dataset_name;
    delete q.dataset_id;
    router.replace({ query: q });
  } catch (e: any) {
    toast.error(e?.message || String(e));
  }
}

function openRunFromQuery() {
  const runId = String(route.query.run || '').trim();
  if (!runId) return;
  activeRunId.value = runId;
  const q = { ...route.query };
  delete q.run;
  router.replace({ query: q });
}

function resumeFromRouteQuery() {
  const taskId = String(route.query.resume_task || '').trim();
  const checkpoint = String(route.query.resume_checkpoint || '').trim();
  if (!taskId || !checkpoint) return Promise.resolve();
  applyingResumeSync.value = true;
  form.resume_task_id = taskId;
  form.resume_checkpoint = checkpoint;
  resumeIntent.value = true;
  step.value = Math.max(step.value, 2);
  const q = { ...route.query };
  delete q.resume_task;
  delete q.resume_checkpoint;
  router.replace({ query: q });
  return api.gen.getMediaTask(taskId).then((row: any) => {
    const rec = row as Record<string, unknown>;
    const params = (rec.params || {}) as Record<string, unknown>;
    const sourceBase = String(params.base_model || rec.model_id || '').split(':')[0].trim();
    if (sourceBase) form.base_model = sourceBase;
    const sourcePreset = String(params.preset || '').trim();
    if (sourcePreset) form.preset = sourcePreset;
    coerceVisiblePreset();
    void refreshRequirements();
    toast.success(t('loraTrain.resumeTraining'));
  }).catch(() => {
    toast.success(t('loraTrain.resumeTraining'));
  }).finally(() => {
    applyingResumeSync.value = false;
  });
}

watch(step, () => {
  persistDraft();
});

watch(
  () => [
    form.base_model,
    form.dataset_id,
    form.preset,
    form.default_prompt,
    form.output_name,
  ],
  () => persistDraft()
);

watch(() => form.base_model, (mid) => {
  if (applyingResumeSync.value) return;
  if (hasResume.value) {
    const hadIntent = resumeIntent.value;
    clearResume();
    if (hadIntent) {
      toast.info(t('loraTrain.resumeClearedOnBaseChange'));
    }
  }
  if (!presetKeys.value.includes(form.preset)) {
    form.preset = recommendedPresetFor(mid);
  }
  coerceVisiblePreset();
  void refreshRequirements();
});

watch(activeResolutionKey, () => {
  datasetHealthStale.value = true;
  if (form.dataset_id && isIdentityDataset.value) void refreshDatasetHealth();
});

watch(() => form.preset, () => {
  coerceVisiblePreset();
});

async function submitTraining() {
  submitting.value = true;
  try {
    await datasetPanelRef.value?.saveDraft();
    if (form.dataset_id) await refreshDatasetHealth();
    if (!canSubmit.value) {
      toast.warning(t('loraTrain.faceDatasetGate'));
      return;
    }
    const body = buildTrainingBody();
    const res = await api.loras.submitTraining(body);
    const tid = taskIdFromSubmitResponse(res);
    toast.success(t('loraTrain.submitted'));
    openGlobalTaskQueue();
    if (tid) {
      activeRunId.value = tid;
      historyRef.value?.refresh();
    }
    clearResume();
  } catch (e: any) {
    const msg = e?.response?.data?.detail?.message || e?.message || String(e);
    toast.error(msg);
  } finally {
    submitting.value = false;
  }
}

async function goToNextFromDataset() {
  await datasetPanelRef.value?.saveDraft();
  if (form.dataset_id && (datasetHealthStale.value || datasetHealth.value?.faces?.resolution?.join('x') !== activeResolutionKey.value)) {
    await refreshDatasetHealth();
  }
  if (!canNext(1)) {
    toast.warning(t('loraTrain.faceDatasetGate'));
    return;
  }
  goToStep(2);
}

function startNewTraining() {
  activeRunId.value = '';
  form.output_name = '';
  form.preset = recommendedPreset.value;
  clearResume();
  coerceVisiblePreset();
  if (step.value !== 0) step.value = 0;
  else persistDraft();
  toast.success(t('loraTrain.newTrainingOpened'));
  void nextTick(() => {
    document.querySelector('.lora-train-page__panel--base')?.scrollIntoView({ block: 'nearest' });
  });
}

function onVerifyGenerate(payload: Record<string, unknown>) {
  router.push({
    name: 'image_create',
    query: Object.fromEntries(Object.entries(payload).filter(([, value]) => value != null).map(([key, value]) => [key, Array.isArray(value) ? value.join(',') : String(value)])),
  });
}

onMounted(async () => {
  try {
    await Promise.all([loadMeta(), refreshVisionAvailability(), registryStore.load()]);
    restoreDraft();
    if (!form.dataset_id && datasets.value.length === 1) {
      form.dataset_id = datasets.value[0].id;
    }
    coerceVisiblePreset();
    openRunFromQuery();
    await resumeFromRouteQuery();
    await importFromRouteQuery();
    if (form.dataset_id) {
      await refreshDatasetHealth(form.dataset_id);
    }
  } catch (e: unknown) {
    toast.error(apiErrorMessage(e));
  }
});
</script>

<style scoped>
.lora-train-page__unavailable {
  max-width: 40rem;
  margin: 2rem auto;
  padding: 1.5rem 1.75rem;
}
.lora-train-page__unavailable h2 {
  margin: 0 0 0.75rem;
  font-size: 1.25rem;
}
.lora-train-page__unavailable p {
  margin: 0 0 0.75rem;
  line-height: 1.5;
  opacity: 0.9;
}
.lora-train-page__unavailable-detail {
  font-size: 0.875rem;
  opacity: 0.7;
}
.lora-train-page__sidebar-card {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.lora-train-page__sidebar-card :deep(.dq-surface-card__body) {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.lora-train-page .lv-sidebar-card__new-btn {
  margin-bottom: 14px;
  flex-shrink: 0;
}

.lora-train-page__sidebar-body {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
  overflow: hidden;
}

.lora-train-page__sidebar-body .lora-train-history {
  flex: 1;
  min-height: 0;
  overflow: hidden;
}

.lora-train-page__sidebar-body .lora-train-history__list {
  max-height: none;
  flex: 1;
  min-height: 0;
  overflow-y: auto;
}

.lora-train-page .dq-download-menu__label {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* ── Stepper (reuses lv-editor-tabs visual) ── */
.lora-train-page__stepper {
  flex-shrink: 0;
}

.lora-train-page__stepper .lv-editor-tabs__tab--locked {
  cursor: not-allowed;
  opacity: 0.45;
}

/* ── Nav bar (panel bottom) ── */
.lora-train-page__nav-bar {
 display: flex;
 align-items: center;
 justify-content: space-between;
 padding: 10px 0 0;
 border-top: 0.5px solid var(--dq-border-subtle);
 flex-shrink: 0;
 margin-top: auto;
 width: 100%;
 box-sizing: border-box;
}

.lora-train-page .copilot-page__workspace-card > :deep(.dq-surface-card__body) {
 flex: 1;
 min-height: 0;
 overflow: hidden;
 display: flex;
 flex-direction: column;
 padding: 12px 16px 16px;
 width: 100%;
 max-width: 100%;
 box-sizing: border-box;
}

.lora-train-page__panel {
  width: 100%;
  min-width: 0;
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
  overflow: hidden;
}

.lora-train-page__panel:not(.lora-train-page__panel--base) {
  overflow-y: auto;
  overflow-x: hidden;
}

.lora-train-page__panel--base {
  width: 100%;
  overflow-y: auto;
  overflow-x: hidden;
}

.lora-train-page__panel--flush {
  max-width: none;
  width: 100%;
}

.lora-train-page__face-intro {
  margin-bottom: 22px;
}

.lora-train-page__face-intro h2 {
  margin: 0 0 8px;
  font-size: var(--dq-font-size-headline);
  color: var(--dq-label-primary);
}

.lora-train-page__face-intro p {
  max-width: 760px;
  margin: 0;
  font-size: var(--dq-font-size-body);
  line-height: 1.6;
  color: var(--dq-label-secondary);
}

.lora-train-page__base-grid {
 display: grid;
 grid-template-columns: repeat(2, minmax(0, 1fr));
 gap: 16px;
 width: 100%;
 align-items: stretch;
}

.lora-train-page__base-tile {
  position: relative;
  display: flex;
  align-items: flex-start;
  gap: 16px;
  width: 100%;
  min-width: 0;
  height: 100%;
  box-sizing: border-box;
  padding: 20px;
  border: 1.5px solid var(--dq-border-subtle);
  border-radius: 16px;
  background: var(--dq-bg-base);
  cursor: pointer;
  text-align: left;
  transition:
    border-color 0.2s ease,
    background 0.2s ease,
    box-shadow 0.2s ease,
    transform 0.15s ease;
}

.lora-train-page__base-tile:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.lora-train-page__base-tile:not(:disabled):hover {
  border-color: color-mix(in srgb, var(--dq-accent) 35%, var(--dq-border));
  background: color-mix(in srgb, var(--dq-accent) 3%, var(--dq-bg-base));
  transform: translateY(-2px);
  box-shadow: 0 8px 24px color-mix(in srgb, var(--dq-accent) 8%, transparent);
}

.lora-train-page__base-tile.is-selected {
  border-color: var(--dq-accent);
  background: color-mix(in srgb, var(--dq-accent) 6%, var(--dq-bg-base));
  box-shadow:
    inset 0 0 0 1.5px var(--dq-accent),
    0 12px 32px color-mix(in srgb, var(--dq-accent) 12%, transparent);
}

.lora-train-page__base-tile-icon {
  flex-shrink: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 56px;
  height: 56px;
  border-radius: 14px;
  font-size: var(--dq-font-size-title);
  font-weight: 700;
  letter-spacing: var(--dq-tracking-tighter);
  color: var(--dq-label-primary);
  background: var(--dq-fill-secondary);
  border: 1px solid var(--dq-border-subtle);
  transition: color 140ms ease, background-color 140ms ease, border-color 140ms ease;
}

.lora-train-page__base-tile.is-family-flux .lora-train-page__base-tile-icon,
.lora-train-page__base-tile.is-family-zimage .lora-train-page__base-tile-icon,
.lora-train-page__base-tile.is-family-qwen .lora-train-page__base-tile-icon {
  color: var(--dq-label-secondary);
  background: var(--dq-fill-secondary);
  border-color: var(--dq-border-subtle);
}

.lora-train-page__base-tile.is-selected .lora-train-page__base-tile-icon {
  transform: scale(1.05);
  box-shadow: 0 4px 12px color-mix(in srgb, var(--dq-accent) 20%, transparent);
}

.lora-train-page__base-tile-body {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
  padding-top: 4px;
}

.lora-train-page__base-tile-name {
  font-size: var(--dq-font-size-body);
  font-weight: 600;
  letter-spacing: var(--dq-tracking-tight);
  color: var(--dq-label-primary);
  line-height: 1.3;
}

.lora-train-page__base-tile-desc {
  font-size: var(--dq-font-size-caption);
  line-height: 1.5;
  color: var(--dq-label-tertiary);
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.lora-train-page__base-tile-check {
  position: absolute;
  right: 16px;
  bottom: 16px;
  width: 20px;
  height: 20px;
  border-radius: 999px;
  background: var(--dq-accent);
  box-shadow: 0 0 0 4px color-mix(in srgb, var(--dq-accent) 15%, transparent);
}

.lora-train-page__base-tile-check::after {
  content: '';
  position: absolute;
  left: 7px;
  top: 3px;
  width: 6px;
  height: 10px;
  border: solid var(--dq-on-accent);
  border-width: 0 2px 2px 0;
  transform: rotate(45deg);
}

.lora-train-page__panel--flush > :deep(.lora-dataset-panel) {
}

.lora-train-page__preset-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
  margin-bottom: 12px;
  width: 100%;
}

.lora-train-page__preset-card {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 8px;
  min-height: 132px;
  padding: 18px 18px 16px;
  border: 1.5px solid var(--dq-border-subtle);
  border-radius: 16px;
  background: var(--dq-bg-base);
  cursor: pointer;
  text-align: left;
  transition: border-color 0.15s ease, background 0.15s ease, box-shadow 0.15s ease;
}

.lora-train-page__preset-card:hover {
  border-color: var(--dq-border);
}

.lora-train-page__preset-card.is-recommended {
  border-color: var(--dq-accent-border-hover);
}

.lora-train-page__preset-badge {
  margin-left: 0.35rem;
  font-size: var(--dq-font-size-caption);
  font-style: normal;
  font-weight: 600;
  color: var(--dq-accent);
}

.lora-train-page__recipe-note {
  margin: 0 0 16px;
  padding: 10px 14px;
  border-radius: 12px;
  background: color-mix(in srgb, var(--dq-bg-base) 55%, var(--dq-fill-secondary));
  border: 0.5px solid var(--dq-border-subtle);
  font-size: var(--dq-font-size-caption);
  line-height: 1.55;
  color: var(--dq-label-secondary);
}

.lora-train-page__essentials .lora-train-page__field-hint {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.lora-train-page__essentials {
  display: grid;
  grid-template-columns: minmax(0, 1.4fr) minmax(200px, 0.7fr);
  gap: 8px 20px;
  align-items: start;
}

.lora-train-page__review.has-preview {
  display: grid;
  grid-template-columns: minmax(0, 1.2fr) minmax(220px, 0.8fr);
  gap: 20px;
  align-items: start;
}

.lora-train-page__review-main {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.lora-train-page__preset-card.is-selected {
  border-color: color-mix(in srgb, var(--dq-accent) 55%, var(--dq-border));
  background: color-mix(in srgb, var(--dq-accent) 9%, var(--dq-fill-secondary));
  box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--dq-accent) 22%, transparent);
}

.lora-train-page__preset-name {
  font-size: var(--dq-font-size-body);
  font-weight: 600;
  color: var(--dq-label-primary);
}

.lora-train-page__preset-desc {
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
  color: var(--dq-label-tertiary);
}

.lora-train-page__preset-stats {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  color: var(--dq-accent);
  margin-top: 2px;
}

.lora-train-page__preset-detail {
  margin-bottom: 16px;
  padding: 12px 14px;
  border-radius: 12px;
  background: color-mix(in srgb, var(--dq-bg-base) 50%, var(--dq-fill-secondary));
  border: 0.5px solid var(--dq-border-subtle);
}

.lora-train-page__custom-block {
  margin-bottom: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.lora-train-page__custom-guide-text {
  margin: 0 0 8px;
  font-size: var(--dq-font-size-caption);
  line-height: 1.5;
  color: var(--dq-label-secondary);
}

.lora-train-page__custom-guide-list {
  margin: 0;
  padding-left: 1.1rem;
  font-size: var(--dq-font-size-caption);
  line-height: 1.5;
  color: var(--dq-label-secondary);
}

.lora-train-page__custom-guide-list li {
  margin-bottom: 4px;
}

.lora-train-page__prompt-warn {
  margin-top: 10px;
}

.lora-train-page__prompt-warn p {
  margin: 0 0 8px;
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
}

.lora-train-page__preset-detail-label {
  display: block;
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  color: var(--dq-label-secondary);
  margin-bottom: 8px;
}

.lora-train-page__preset-detail-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 18px;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-primary);
}

.lora-train-page__preset-crop-hint {
  font-style: normal;
  color: var(--dq-label-tertiary);
}

.lora-train-page__field-hint {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
  line-height: 1.4;
}

.lora-train-page__confirm-intro {
  margin: 0 0 14px;
  font-size: var(--dq-font-size-body);
  color: var(--dq-label-secondary);
  line-height: 1.5;
}

.lora-train-page__identity-checklist {
  list-style: none;
  margin: 0 0 16px;
  padding: 0;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
}

.lora-train-page__identity-checklist li {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 4px 0;
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
  color: var(--dq-label-primary);
}

.lora-train-page__identity-checklist li.is-ok .lora-train-page__identity-check-mark {
  color: var(--dq-success);
}

.lora-train-page__identity-checklist li.is-bad .lora-train-page__identity-check-mark {
  color: var(--dq-danger);
}

.lora-train-page__identity-check-mark {
  flex-shrink: 0;
  width: 1.25rem;
  font-weight: 700;
}

.lora-train-page__confirm-health {
  margin-bottom: 14px;
  width: 100%;
}

.lora-train-page__field {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-bottom: 14px;
  min-width: 0;
}

.lora-train-page__label {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  letter-spacing: var(--dq-tracking-wide);
  color: var(--dq-label-secondary);
}

.lora-train-page__optimization {
  margin: 16px 0;
  padding: 14px 16px;
  border-radius: 12px;
  border: 0.5px solid var(--dq-border-subtle);
  background: color-mix(in srgb, var(--dq-bg-base) 55%, var(--dq-fill-secondary));
  max-width: 100%;
}

.lora-train-page__optimization-head {
  margin-bottom: 12px;
}

.lora-train-page__field-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 12px 16px;
  max-width: 100%;
}

.lora-train-page__field--wide {
  grid-column: 1 / -1;
}

.lora-train-page__field--switch {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.lora-train-page__resume-banner {
  margin-bottom: 16px;
}

.lora-train-page__resume-banner p {
  margin: 0 0 10px;
}

.lora-train-page__advanced {
  margin-top: 4px;
  font-size: var(--dq-font-size-body);
  color: var(--dq-label-secondary);
}

.lora-train-page__advanced summary {
  cursor: pointer;
  margin-bottom: 12px;
}

.lora-train-page__summary {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: 100%;
  max-width: 100%;
  min-width: 0;
}

.lora-train-page__summary li {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  padding: 10px 0;
  border-bottom: 0.5px solid var(--dq-border-subtle);
  font-size: var(--dq-font-size-body);
  width: 100%;
  max-width: 100%;
  min-width: 0;
  box-sizing: border-box;
}

.lora-train-page__summary li:last-child {
  border-bottom: none;
}

.lora-train-page__summary span {
  color: var(--dq-label-tertiary);
  flex-shrink: 0;
}

.lora-train-page__summary strong {
  color: var(--dq-label-primary);
  font-weight: 600;
  text-align: right;
  word-break: break-word;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  flex-shrink: 1;
}

.lora-train-page__confirm-preview {
  margin-bottom: 16px;
  width: 100%;
}

.lora-train-page__confirm-preview-label {
  display: block;
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  color: var(--dq-label-secondary);
  margin-bottom: 8px;
}

.lora-train-page__confirm-preview-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(92px, 1fr));
  gap: 8px;
}

.lora-train-page__confirm-preview-grid img {
  width: 100%;
  height: 108px;
  object-fit: cover;
  border-radius: 12px;
  border: 0.5px solid var(--dq-border);
}

.lora-train-page__confirm-preview-more {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 108px;
  border-radius: 10px;
  background: var(--dq-fill-tertiary);
  border: 0.5px dashed var(--dq-border);
  font-size: var(--dq-font-size-body);
  font-weight: 600;
  color: var(--dq-label-tertiary);
}

@media (max-width: 960px) {
  .lora-train-page.copilot-page {
    flex-direction: column;
    overflow: auto;
  }

  .lora-train-page__sidebar-history {
    flex: 0 0 auto;
    max-height: 220px;
  }

  .lora-train-page__memory-panel {
    margin-top: 0;
  }

  .lora-train-page__base-grid,
  .lora-train-page__preset-grid,
  .lora-train-page__essentials,
  .lora-train-page__review.has-preview,
  .lora-train-page__identity-checklist {
    grid-template-columns: 1fr;
  }

  .lora-train-page__stepper .lv-editor-tabs__desc {
    display: none;
  }
}
</style>
