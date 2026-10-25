<template>
  <div class="lora-dataset-panel">
    <!-- Left: persistent dataset library -->
    <aside class="lora-dataset-panel__list">
      <div class="lora-dataset-panel__list-head">
        <span class="lora-dataset-panel__list-title">{{ $t('loraTrain.datasetLibrary') }}</span>
        <DqButton type="primary" @click="openCreateDialog">
          {{ $t('loraTrain.newDataset') }}
        </DqButton>
      </div>

      <p class="lora-dataset-panel__persist-note">{{ $t('loraTrain.datasetPersistNote') }}</p>

      <DqInput
        v-if="datasets.length > 3"
        v-model="datasetSearch"
        size="sm"
        class="lora-dataset-panel__search"
        :placeholder="$t('loraTrain.searchDatasets')"
        clearable
      />

      <div v-if="filteredDatasets.length" class="lora-dataset-panel__list-scroll">
        <button
          v-for="d in filteredDatasets"
          :key="d.id"
          type="button"
          class="lora-dataset-panel__list-item"
          :class="{ 'is-active': d.id === selectedId, 'is-ready': (d.image_count || 0) >= minImages }"
          @click="selectDataset(d.id)"
        >
          <img
            v-if="datasetCoverUrl(d)"
            :src="datasetCoverUrl(d)"
            alt=""
            class="lora-dataset-panel__list-thumb"
            loading="lazy"
          />
          <span v-else class="lora-dataset-panel__list-thumb lora-dataset-panel__list-thumb--empty" aria-hidden="true">
            <DqIcon :size="16"><PictureFilled /></DqIcon>
          </span>
          <span class="lora-dataset-panel__list-body">
            <span class="lora-dataset-panel__list-name">{{ d.name }}</span>
            <span class="lora-dataset-panel__list-meta">
              {{ $t('loraTrain.imageCount', { count: d.image_count || 0 }) }}
              <DqTag
                v-if="(d.image_count || 0) >= minImages"
                size="sm"
                type="success"
                effect="plain"
                class="lora-dataset-panel__ready-tag"
              >
                {{ $t('loraTrain.datasetReady') }}
              </DqTag>
            </span>
          </span>
          <DqIconButton
            type="text"
            size="sm"
            class="lora-dataset-panel__list-delete"
            :label="$t('loraTrain.deleteDataset')"
            :disabled="deletingDataset"
            @click.stop="deleteDataset(d.id, d.name)"
          >
            <DqIcon :size="14"><Delete /></DqIcon>
          </DqIconButton>
        </button>
      </div>
      <DqEmpty
        v-else-if="datasets.length"
        :description="$t('loraTrain.noSearchResults')"
        class="lora-dataset-panel__list-empty"
      />
      <DqEmpty v-else :description="$t('loraTrain.noDatasets')" class="lora-dataset-panel__list-empty" />

      <div class="lora-dataset-panel__list-foot">
        <DqButton size="sm" type="secondary" :loading="importingDog6" block @click="importDog6">
          {{ $t('loraTrain.importDog6') }}
        </DqButton>
      </div>
    </aside>

    <!-- Right: active dataset workspace -->
    <div class="lora-dataset-panel__workspace">
      <template v-if="selectedId && selectedDataset">
        <div class="lora-dataset-panel__workspace-head">
          <div class="lora-dataset-panel__name-row">
            <label class="lora-dataset-panel__label">{{ $t('loraTrain.datasetName') }}</label>
            <DqInput
              v-model="datasetNameEdit"
              size="sm"
              class="lora-dataset-panel__name-input"
              @blur="saveDatasetName"
            />
          </div>
          <div class="lora-dataset-panel__workspace-head-actions">
            <DqTag
              size="sm"
              :type="trainingCount >= minImages ? 'success' : 'warning'"
              effect="plain"
            >
              {{ skippedCount
                ? $t('loraTrain.datasetProgressTraining', { training: trainingCount, total: imageCount })
                : $t('loraTrain.datasetProgress', { current: imageCount, min: minImages }) }}
            </DqTag>
            <DqButton
              size="sm"
              type="danger"
              :loading="deletingDataset"
              @click="deleteDataset(selectedId, selectedDataset?.name || selectedId)"
            >
              {{ $t('loraTrain.deleteDataset') }}
            </DqButton>
          </div>
        </div>

        <div class="lora-dataset-panel__meta">
        <div class="lora-dataset-panel__kind-row">
          <label class="lora-dataset-panel__label">{{ $t('loraTrain.datasetKind') }}</label>
          <DqSelect v-model="datasetKindEdit" size="sm" class="lora-dataset-panel__kind-select" @change="saveDatasetKind">
            <DqOption :label="$t('loraTrain.datasetKindConcept')" value="concept" />
            <DqOption :label="$t('loraTrain.datasetKindStyle')" value="style" />
          </DqSelect>
        </div>

        <div v-if="datasetKindEdit === 'concept'" class="lora-dataset-panel__field">
          <label class="lora-dataset-panel__label">{{ $t('loraTrain.triggerWord') }}</label>
          <div class="lora-dataset-panel__trigger-row">
            <DqInput
              v-model="triggerWordEdit"
              size="sm"
              :placeholder="$t('loraTrain.triggerWordHint')"
              @blur="saveTriggerWord"
            />
            <DqButton
              v-if="triggerWordEdit.trim() && !captionHasTrigger"
              size="sm"
              type="secondary"
              @click="writeTriggerIntoCaption"
            >
              {{ $t('loraTrain.bindTriggerToCaption') }}
            </DqButton>
          </div>
          <p v-if="datasetHealthLoading" class="lora-dataset-panel__trigger-status is-warning">
            {{ $t('loraTrain.facePreflightRunning') }}
          </p>
          <p v-else-if="datasetHealth" class="lora-dataset-panel__trigger-status" :class="datasetHealthStale ? 'is-warning' : 'is-ready'">
            {{ datasetHealthStale
              ? $t('loraTrain.facePreflightStale')
              : $t('loraTrain.facePreflightFresh', { resolution: (datasetHealth.faces?.resolution || []).join('×') }) }}
          </p>
          <p
            v-if="triggerWordEdit.trim()"
            class="lora-dataset-panel__trigger-status"
            :class="captionHasTrigger ? 'is-ready' : 'is-warning'"
          >
            {{ captionHasTrigger
              ? $t('loraTrain.triggerCoverageReady', { trigger: triggerWordEdit.trim() })
              : $t('loraTrain.triggerCoverageMissing', { trigger: triggerWordEdit.trim() }) }}
          </p>
          <p v-else class="lora-dataset-panel__trigger-status is-warning">
            {{ $t('loraTrain.identityCheckTriggerMissing') }}
          </p>
        </div>

        <div class="lora-dataset-panel__field">
          <label class="lora-dataset-panel__label">{{ $t('loraTrain.defaultPrompt') }}</label>
          <DqInput
            :model-value="defaultPrompt"
            type="textarea"
            :rows="2"
            :placeholder="$t('loraTrain.defaultPromptHint')"
            @update:model-value="(v: string) => emit('update:defaultPrompt', v)"
            @blur="patchDefaultPrompt"
          />
          <p class="lora-dataset-panel__field-hint">{{ $t('loraTrain.defaultPromptDesc') }}</p>
        </div>
        </div>

        <details v-if="datasetKindEdit === 'concept'" class="lora-dataset-panel__guide" :open="imageCount < 3">
          <summary>{{ $t('loraTrain.identityGuideTitle') }}</summary>
          <p>{{ $t('loraTrain.identityGuideBody') }}</p>
        </details>

        <div class="lora-dataset-panel__toolbar">
        <div
          class="lora-dataset-panel__dropzone"
          :class="{
            'is-dragover': dragOver,
            'is-disabled': !selectedId,
            'is-uploading': uploading,
            'is-compact': imageCount > 0,
          }"
          @dragover.prevent="dragOver = true"
          @dragleave.prevent="dragOver = false"
          @drop.prevent="onDropFiles"
          @click="openUploadPicker"
        >
          <DqIcon class="lora-dataset-panel__dropzone-icon" aria-hidden="true"><Upload /></DqIcon>
          <p class="lora-dataset-panel__dropzone-title">
            {{ uploading ? $t('loraTrain.uploading') : $t('loraTrain.dropzoneTitle') }}
          </p>
          <p v-if="!uploading && !imageCount" class="lora-dataset-panel__dropzone-hint">{{ $t('loraTrain.dropzoneHint') }}</p>
          <input
            ref="uploadInputRef"
            type="file"
            multiple
            accept="image/*"
            hidden
            @change="onUploadImages"
          />
        </div>

        <div class="lora-dataset-panel__actions">
          <DqButton size="sm" type="primary" :loading="preflightLoading || datasetHealthLoading" :disabled="!imageCount" @click="runPreflight">
            {{ $t('loraTrain.runFacePreflight') }}
          </DqButton>
          <DqButton size="sm" type="secondary" @click="showGalleryImport = true">
            {{ $t('loraTrain.importGallery') }}
          </DqButton>
        </div>
        </div>

        <div v-if="hasVlmMarks" class="lora-dataset-panel__vlm-legend">
          <span class="lora-dataset-panel__vlm-legend-label">{{ $t('loraTrain.quality.vlmMarkLegend') }}</span>
          <span class="lora-dataset-panel__vlm-legend-item is-good">{{ $t('loraTrain.quality.vlmMarkGood') }}</span>
          <span class="lora-dataset-panel__vlm-legend-item is-fair">{{ $t('loraTrain.quality.vlmMarkFair') }}</span>
          <span class="lora-dataset-panel__vlm-legend-item is-poor">{{ $t('loraTrain.quality.vlmMarkPoor') }}</span>
        </div>

        <div v-if="hasFaceMarks" class="lora-dataset-panel__vlm-legend">
          <span class="lora-dataset-panel__vlm-legend-label">{{ $t('loraTrain.quality.faceMarkLegend') }}</span>
          <span class="lora-dataset-panel__vlm-legend-item is-face-crop">{{ $t('loraTrain.quality.faceMarkCrop') }}</span>
          <span class="lora-dataset-panel__vlm-legend-item is-face-none">{{ $t('loraTrain.quality.faceMarkNone') }}</span>
          <span class="lora-dataset-panel__vlm-legend-item is-face-multi">{{ $t('loraTrain.quality.faceMarkMulti') }}</span>
        </div>

        <div v-if="showGateBar" class="lora-dataset-panel__gate-bar">
          <p v-if="blockedFiles.length" class="lora-dataset-panel__gate-copy">
            {{ $t('loraTrain.gateBlockedBar', { count: blockedFiles.length }) }}
          </p>
          <p v-else-if="skippedCount" class="lora-dataset-panel__gate-copy">
            {{ $t('loraTrain.gateSkippedBar', { count: skippedCount }) }}
          </p>
          <div class="lora-dataset-panel__gate-bar-actions">
            <DqButton size="sm" type="secondary" @click="showBlockedOnly = !showBlockedOnly">
              {{ showBlockedOnly ? $t('loraTrain.gateShowAll') : $t('loraTrain.gateShowBlocked') }}
            </DqButton>
            <DqButton
              v-if="blockedFiles.length"
              size="sm"
              type="secondary"
              :loading="skippingImages"
              @click="skipFiles(blockedFiles)"
            >
              {{ $t('loraTrain.gateSkipAll') }}
            </DqButton>
            <DqButton
              v-if="blockedFiles.length"
              size="sm"
              type="danger"
              :loading="deletingBlocked"
              @click="deleteBlocked"
            >
              {{ $t('loraTrain.gateDeleteAll') }}
            </DqButton>
          </div>
        </div>

        <div v-if="selectedDataset?.images?.length" class="lora-dataset-panel__grid">
          <div
            v-for="img in displayImages"
            :key="img.file"
            class="lora-dataset-panel__cell"
            :class="{
              'is-gate-blocked': isGateFailure(img.file) && !isSkipped(img.file),
              'is-skipped': isSkipped(img.file),
            }"
          >
            <div
              class="lora-dataset-panel__cell-media"
              :class="vlmMediaClass(img.file)"
            >
              <img
                v-if="datasetImageUrl(img.file)"
                :src="datasetImageUrl(img.file)"
                :alt="img.file"
                loading="lazy"
              />
              <span
                v-if="vlmMark(img.file)"
                class="lora-dataset-panel__vlm-mark"
                :class="`is-${vlmMark(img.file)!.level}`"
                :title="vlmMarkTitle(img.file)"
              >
                {{ vlmMark(img.file)!.scoreText }}
              </span>
              <span
                v-if="faceCropBox(img.file)"
                class="lora-dataset-panel__face-window"
                :style="faceCropBox(img.file)!"
                aria-hidden="true"
              />
              <span
                v-if="faceMark(img.file)"
                class="lora-dataset-panel__face-mark"
                :class="`is-${faceMark(img.file)!.kind}`"
                :title="faceMark(img.file)!.title"
              >
                {{ faceMark(img.file)!.label }}
              </span>
              <DqIconButton
                type="text"
                size="sm"
                class="lora-dataset-panel__cell-remove"
                :label="$t('loraTrain.removeImage')"
                @click.stop="removeImage(img.file)"
              >
                <DqIcon :size="14"><Close /></DqIcon>
              </DqIconButton>
            </div>
            <div v-if="isGateFailure(img.file) && !isSkipped(img.file)" class="lora-dataset-panel__gate-actions">
              <DqButton size="sm" type="secondary" :disabled="skippingImages" @click="skipFiles([img.file])">
                {{ $t('loraTrain.gateSkip') }}
              </DqButton>
              <DqButton size="sm" type="danger" @click="removeImage(img.file)">
                {{ $t('loraTrain.gateDeleteBlocked') }}
              </DqButton>
            </div>
            <div v-else-if="isSkipped(img.file)" class="lora-dataset-panel__gate-actions">
              <span class="lora-dataset-panel__gate-skipped">{{ $t('loraTrain.gateSkipped') }}</span>
              <DqButton size="sm" type="secondary" :disabled="skippingImages" @click="restoreFile(img.file)">
                {{ $t('loraTrain.gateRestore') }}
              </DqButton>
            </div>
          </div>
        </div>
        <DqEmpty v-else :description="$t('loraTrain.noImages')" />

        <LoraQualityHints
          v-if="datasetHealth"
          :report="datasetHealth"
          mode="dataset"
          fold
          :vision-available="visionAvailable"
          :vlm-loading="datasetVlmLoading"
          :dataset-audit-kind="datasetKindEdit"
          hide-vlm-desc
          class="lora-dataset-panel__health"
          @run-vlm="$emit('run-dataset-vlm')"
        />
      </template>

      <DqEmpty v-else :description="$t('loraTrain.selectOrCreateDataset')" />
    </div>

    <DqDialog
      v-model:open="showCreateDialog"
      :title="$t('loraTrain.newDataset')"
      width="min(420px, 92vw)"
      destroy-on-close
    >
      <div class="lora-dataset-panel__dialog-field">
        <label class="lora-dataset-panel__label">{{ $t('loraTrain.newDatasetName') }}</label>
        <DqInput v-model="newDatasetName" :placeholder="$t('loraTrain.newDatasetDefault')" @keyup.enter="createDataset" />
      </div>
      <div class="lora-dataset-panel__dialog-field">
        <label class="lora-dataset-panel__label">{{ $t('loraTrain.triggerWord') }}</label>
        <DqInput v-model="newTriggerWord" :placeholder="$t('loraTrain.triggerWordHint')" @keyup.enter="createDataset" />
        <p class="lora-dataset-panel__field-hint">{{ $t('loraTrain.triggerWordDesc') }}</p>
      </div>
      <template #footer>
        <DqButton size="sm" @click="showCreateDialog = false">{{ $t('common.cancel') }}</DqButton>
        <DqButton type="primary" :disabled="!newDatasetName.trim()" @click="createDataset">
          {{ $t('common.confirm') }}
        </DqButton>
      </template>
    </DqDialog>

    <DqDialog
      v-model:open="showGalleryImport"
      :title="$t('loraTrain.importGalleryTitle')"
      width="min(560px, 92vw)"
      destroy-on-close
    >
      <p class="lora-dataset-panel__import-hint">{{ $t('loraTrain.importGalleryHint') }}</p>
      <AssetPicker accept-kind="image" @pick="onGalleryPick" />
      <div v-if="pendingGalleryAssets.length" class="lora-dataset-panel__pending-tags">
        <span
          v-for="a in pendingGalleryAssets"
          :key="a"
          class="lora-dataset-panel__pending-tag"
        >
          <span class="lora-dataset-panel__pending-tag-label">{{ a }}</span>
          <DqIconButton
            type="text"
            size="sm"
            :label="$t('common.delete')"
            @click="removePendingAsset(a)"
          >
            <DqIcon :size="12"><Close /></DqIcon>
          </DqIconButton>
        </span>
      </div>
      <template #footer>
        <DqButton size="sm" @click="showGalleryImport = false">{{ $t('common.cancel') }}</DqButton>
        <DqButton
          size="sm"
          type="primary"
          :disabled="!pendingGalleryAssets.length"
          :loading="importingGallery"
          @click="confirmGalleryImport"
        >
          {{ $t('loraTrain.importGalleryConfirm', { count: pendingGalleryAssets.length }) }}
        </DqButton>
      </template>
    </DqDialog>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { Close, Upload, PictureFilled, Delete } from '@danqing/dq-shell';
import { api } from '@/utils/api';
import { toast, confirm } from '@/utils/feedback';
import AssetPicker from '@/components/asset/AssetPicker.vue';
import LoraQualityHints from '@/components/lora/LoraQualityHints.vue';
import type { FaceAuditRow, LoraDatasetHealthReport, VlmImageSample } from '@/utils/loraQuality';
import {
  buildFaceRowMap,
  buildVlmSampleMap,
  faceCropOverlay,
  lookupVlmSample,
  vlmScoreLevel,
} from '@/utils/loraQuality';

const MIN_IMAGES = 3;

const props = defineProps<{
  selectedId: string;
  datasets: Array<Record<string, any>>;
  defaultPrompt: string;
  datasetHealth?: LoraDatasetHealthReport | null;
  datasetHealthStale?: boolean;
  datasetHealthLoading?: boolean;
  visionAvailable?: boolean;
  datasetVlmLoading?: boolean;
}>();

const emit = defineEmits<{
  (e: 'update:selectedId', id: string): void;
  (e: 'update:defaultPrompt', value: string): void;
  (e: 'datasets-changed', datasets: Array<Record<string, any>>): void;
  (e: 'preflight-requested'): Promise<void> | void;
  (e: 'run-dataset-vlm'): void;
}>();

const { t } = useI18n();

const minImages = MIN_IMAGES;
const dragOver = ref(false);
const uploading = ref(false);
const importingDog6 = ref(false);
const importingGallery = ref(false);
const preflightLoading = ref(false);
const deletingDataset = ref(false);
const deletingBlocked = ref(false);
const skippingImages = ref(false);
const showBlockedOnly = ref(false);
const showCreateDialog = ref(false);
const showGalleryImport = ref(false);
const newDatasetName = ref('');
const newTriggerWord = ref('');
const datasetSearch = ref('');
const pendingGalleryAssets = ref<string[]>([]);
const uploadInputRef = ref<HTMLInputElement | null>(null);
const datasetNameEdit = ref('');
const triggerWordEdit = ref('');
const datasetKindEdit = ref<'concept' | 'style'>('concept');

const selectedDataset = computed(() =>
  props.datasets.find((d) => d.id === props.selectedId)
);

const imageCount = computed(() => selectedDataset.value?.image_count || 0);
const skippedFiles = computed(() => new Set<string>((selectedDataset.value?.skipped_images || []).map(String)));
const skippedCount = computed(() => skippedFiles.value.size);
const trainingCount = computed(() => {
  const listed = selectedDataset.value?.training_image_count;
  if (typeof listed === 'number' && !(selectedDataset.value?.images || []).length) return listed;
  return (selectedDataset.value?.images || []).filter((img: { file: string }) => !skippedFiles.value.has(img.file)).length;
});
const captionHasTrigger = computed(() => {
  const trigger = triggerWordEdit.value.trim();
  if (!trigger) return false;
  return props.defaultPrompt.includes(trigger);
});

const vlmSampleMap = computed(() => {
  if (!props.datasetHealth?.vlm_audited) return new Map<string, VlmImageSample>();
  return buildVlmSampleMap(props.datasetHealth.vlm?.samples as VlmImageSample[] | undefined);
});

const hasVlmMarks = computed(() => props.datasetHealth?.vlm_audited && vlmSampleMap.value.size > 0);

function vlmMark(file: string): { level: 'good' | 'fair' | 'poor'; scoreText: string } | null {
  const sample = lookupVlmSample(vlmSampleMap.value, file);
  if (!sample || sample.score == null) return null;
  const level = vlmScoreLevel(sample.score);
  if (!level) return null;
  return { level, scoreText: `${Number(sample.score)}/5` };
}

function vlmMarkTitle(file: string): string {
  const sample = lookupVlmSample(vlmSampleMap.value, file);
  if (!sample) return '';
  const parts: string[] = [];
  if (sample.score != null) {
    const vlm = sample.vlm_score;
    const heuristic = sample.heuristic_score;
    if (
      vlm != null &&
      heuristic != null &&
      Math.abs(Number(vlm) - Number(heuristic)) > 0.05
    ) {
      parts.push(
        t('loraTrain.quality.vlmMarkTitleVlmHeuristic', {
          vlm: Number(vlm),
          heuristic: Number(heuristic),
        })
      );
      parts.push(t('loraTrain.quality.vlmMarkTitleScore', { score: Number(sample.score) }));
    } else {
      parts.push(t('loraTrain.quality.vlmMarkTitleScore', { score: Number(sample.score) }));
    }
  }
  if (sample.reason) parts.push(sample.reason);
  if (sample.issues?.length) {
    parts.push(t('loraTrain.quality.vlmMarkTitleIssues', { issues: sample.issues.join(', ') }));
  }
  return parts.join(' — ');
}

function vlmMediaClass(file: string): Record<string, boolean> {
  const mark = vlmMark(file);
  if (!mark) return {};
  return {
    'is-vlm-good': mark.level === 'good',
    'is-vlm-fair': mark.level === 'fair',
    'is-vlm-poor': mark.level === 'poor',
  };
}

const faceRowMap = computed(() => buildFaceRowMap(props.datasetHealth?.faces));

const hasFaceMarks = computed(() => {
  if (!faceRowMap.value.size) return false;
  for (const row of faceRowMap.value.values()) {
    if (row.action !== 'keep' || row.multi) return true;
  }
  return false;
});

function faceRow(file: string): FaceAuditRow | undefined {
  const key = (file || '').trim();
  if (!key) return undefined;
  return faceRowMap.value.get(key) ?? faceRowMap.value.get(key.split('/').pop() || '');
}

function faceCropBox(file: string): Record<string, string> | null {
  const row = faceRow(file);
  if (!row || row.action !== 'crop') return null;
  const box = faceCropOverlay(row);
  if (!box) return null;
  return {
    left: `${box.left}%`,
    top: `${box.top}%`,
    width: `${box.width}%`,
    height: `${box.height}%`,
  };
}

function isSkipped(file: string): boolean {
  return skippedFiles.value.has(file);
}

function isGateFailure(file: string): boolean {
  const row = faceRow(file);
  if (!row) return false;
  return row.action === 'tiny' || row.action === 'none' || row.action === 'error' || row.action === 'multi' || !!row.multi;
}

const blockedFiles = computed(() =>
  (selectedDataset.value?.images || [])
    .map((img: { file: string }) => img.file)
    .filter((file: string) => isGateFailure(file) && !isSkipped(file))
);

const showGateBar = computed(() =>
  datasetKindEdit.value === 'concept' && (blockedFiles.value.length > 0 || skippedCount.value > 0)
);

const displayImages = computed(() => {
  const images = [...(selectedDataset.value?.images || [])];
  const rank = (file: string) => {
    if (isGateFailure(file) && !isSkipped(file)) return 0;
    if (isSkipped(file)) return 1;
    return 2;
  };
  images.sort((a: { file: string }, b: { file: string }) => rank(a.file) - rank(b.file));
  if (!showBlockedOnly.value) return images;
  return images.filter((img: { file: string }) => isGateFailure(img.file) || isSkipped(img.file));
});

function faceMark(file: string): { kind: 'crop' | 'none' | 'tiny' | 'multi'; label: string; title: string } | null {
  const row = faceRow(file);
  if (!row) return null;
  if (row.multi) {
    return {
      kind: 'multi',
      label: t('loraTrain.quality.faceMarkMulti'),
      title: t('loraTrain.quality.faceMarkMultiTitle', { count: row.faces }),
    };
  }
  if (row.action === 'none' || row.action === 'error') {
    return {
      kind: 'none',
      label: t('loraTrain.quality.faceMarkNone'),
      title: t('loraTrain.quality.faceMarkNoneTitle'),
    };
  }
  if (row.action === 'tiny') {
    return {
      kind: 'tiny',
      label: t('loraTrain.quality.faceMarkTiny'),
      title: t('loraTrain.quality.faceMarkTinyTitle', { px: row.face_px ?? 0 }),
    };
  }
  if (row.action === 'crop') {
    const pct = Math.round((row.face_frac ?? 0) * 100);
    return {
      kind: 'crop',
      label: t('loraTrain.quality.faceMarkCrop'),
      title: t('loraTrain.quality.faceMarkCropTitle', { px: row.face_px ?? 0, pct }),
    };
  }
  return null;
}

const filteredDatasets = computed(() => {
  const q = datasetSearch.value.trim().toLowerCase();
  const sorted = [...props.datasets].sort((a, b) =>
    String(b.updated_at || b.created_at || '').localeCompare(String(a.updated_at || a.created_at || ''))
  );
  if (!q) return sorted;
  return sorted.filter(
    (d) =>
      String(d.name || '').toLowerCase().includes(q) ||
      String(d.id || '').toLowerCase().includes(q)
  );
});

watch(
  () => selectedDataset.value?.name,
  (name) => {
    datasetNameEdit.value = name || '';
  },
  { immediate: true }
);

watch(
  () => selectedDataset.value?.trigger_word,
  (trigger) => {
    triggerWordEdit.value = trigger || '';
  },
  { immediate: true }
);

watch(
  () => selectedDataset.value?.kind,
  (kind) => {
    datasetKindEdit.value = kind === 'style' ? 'style' : 'concept';
  },
  { immediate: true }
);

function apiErrorMessage(e: unknown): string {
  const err = e as { response?: { data?: { detail?: { message?: string } | string } }; message?: string };
  const detail = err?.response?.data?.detail;
  if (detail && typeof detail === 'object' && detail.message) return detail.message;
  if (typeof detail === 'string') return detail;
  return err?.message || String(e);
}

function patchDatasets(next: Array<Record<string, any>>) {
  emit('datasets-changed', next);
}

function selectDataset(id: string) {
  emit('update:selectedId', id);
}

function datasetImageUrl(file: string): string {
  if (!props.selectedId || !file) return '';
  return api.loras.datasetImageUrl(props.selectedId, file);
}

function datasetCoverUrl(d: Record<string, any>): string {
  const cover = d.cover_image || d.images?.[0]?.file;
  if (!cover || !d.id) return '';
  return api.loras.datasetImageUrl(d.id, cover);
}

function openUploadPicker() {
  if (!props.selectedId) return;
  uploadInputRef.value?.click();
}

async function refreshDatasetDetail(id: string) {
  const ds = (await api.loras.getDataset(id)) as Record<string, any>;
  const prompt = String(ds.default_prompt || '').trim();
  if (prompt) emit('update:defaultPrompt', prompt);
  patchDatasets(props.datasets.map((d) => (d.id === id ? { ...d, ...ds } : d)));
}

async function createDataset() {
  const name = newDatasetName.value.trim();
  if (!name) return;
  const kind = datasetKindEdit.value;
  const defaultPrompt = props.defaultPrompt.trim();
  const triggerWord = kind === 'concept' ? newTriggerWord.value.trim() : '';
  try {
    const ds = (await api.loras.createDataset({
      name,
      kind,
      trigger_word: triggerWord,
      default_prompt: defaultPrompt,
    })) as Record<string, any>;
    if (defaultPrompt) {
      emit('update:defaultPrompt', defaultPrompt);
    }
    patchDatasets([ds, ...props.datasets]);
    emit('update:selectedId', ds.id);
    showCreateDialog.value = false;
    newDatasetName.value = '';
    newTriggerWord.value = '';
    toast.success(t('loraTrain.datasetCreated'));
  } catch (e: unknown) {
    toast.error(apiErrorMessage(e));
  }
}

async function importDog6() {
  importingDog6.value = true;
  try {
    const ds = (await api.loras.importDog6()) as Record<string, any>;
    const exists = props.datasets.some((d) => d.id === ds.id);
    patchDatasets(exists ? props.datasets.map((d) => (d.id === ds.id ? { ...d, ...ds } : d)) : [ds, ...props.datasets]);
    emit('update:selectedId', ds.id);
    await refreshDatasetDetail(ds.id);
    toast.success(t('loraTrain.dog6Imported'));
  } catch (e: unknown) {
    toast.error(apiErrorMessage(e));
  } finally {
    importingDog6.value = false;
  }
}

function openCreateDialog() {
  newDatasetName.value = t('loraTrain.newDatasetDefault');
  newTriggerWord.value = '';
  showCreateDialog.value = true;
}

function removePendingAsset(path: string) {
  pendingGalleryAssets.value = pendingGalleryAssets.value.filter((p) => p !== path);
}

async function uploadFiles(files: File[]) {
  if (!files.length || !props.selectedId || uploading.value) return;
  uploading.value = true;
  try {
    await api.loras.uploadImages(props.selectedId, files, props.defaultPrompt);
    await refreshDatasetDetail(props.selectedId);
    toast.success(t('loraTrain.uploadDone', { count: files.length }));
  } catch (e: unknown) {
    toast.error(apiErrorMessage(e));
  } finally {
    uploading.value = false;
  }
}

async function onUploadImages(ev: Event) {
  const input = ev.target as HTMLInputElement;
  const files = input.files ? Array.from(input.files) : [];
  await uploadFiles(files);
  input.value = '';
}

async function onDropFiles(ev: DragEvent) {
  dragOver.value = false;
  if (!props.selectedId) return;
  const files = Array.from(ev.dataTransfer?.files || []).filter((f) => f.type.startsWith('image/'));
  if (!files.length) {
    toast.error(t('loraTrain.dropzoneNeedImage'));
    return;
  }
  await uploadFiles(files);
}

function onGalleryPick(payload: { path: string }) {
  const p = payload.path?.trim();
  if (!p || pendingGalleryAssets.value.includes(p)) return;
  pendingGalleryAssets.value.push(p);
}

async function confirmGalleryImport() {
  if (!props.selectedId || !pendingGalleryAssets.value.length) return;
  importingGallery.value = true;
  try {
    const ds = (await api.loras.importAssets(
      props.selectedId,
      pendingGalleryAssets.value,
      props.defaultPrompt
    )) as Record<string, any>;
    patchDatasets(props.datasets.map((d) => (d.id === props.selectedId ? { ...d, ...ds } : d)));
    await refreshDatasetDetail(props.selectedId);
    pendingGalleryAssets.value = [];
    showGalleryImport.value = false;
    toast.success(t('loraTrain.galleryImported'));
  } catch (e: unknown) {
    toast.error(apiErrorMessage(e));
  } finally {
    importingGallery.value = false;
  }
}

async function saveDraft() {
  await saveTriggerWord();
  await patchDefaultPrompt();
}

async function runPreflight() {
  if (!props.selectedId || preflightLoading.value) return;
  preflightLoading.value = true;
  try {
    await saveTriggerWord();
    await patchDefaultPrompt();
    await emit('preflight-requested');
  } finally {
    preflightLoading.value = false;
  }
}

async function writeTriggerIntoCaption() {
  const trigger = triggerWordEdit.value.trim();
  if (!props.selectedId || !trigger || props.defaultPrompt.includes(trigger)) return;
  await saveTriggerWord();
  const current = props.defaultPrompt.trim();
  const next = current ? `${trigger}, ${current}` : trigger;
  emit('update:defaultPrompt', next);
  try {
    const ds = (await api.loras.patchDataset(props.selectedId, {
      ...datasetMetaPatch(),
      default_prompt: next,
    })) as Record<string, any>;
    patchDatasets(props.datasets.map((d) => (d.id === props.selectedId ? { ...d, ...ds } : d)));
    toast.success(t('loraTrain.triggerBound'));
  } catch (e: unknown) {
    toast.error(apiErrorMessage(e));
  }
}

function datasetMetaPatch(name?: string): Record<string, unknown> {
  const kind = datasetKindEdit.value;
  return {
    name: name?.trim() || datasetNameEdit.value.trim() || selectedDataset.value?.name || '',
    trigger_word: kind === 'concept' ? triggerWordEdit.value.trim() : '',
    default_prompt: props.defaultPrompt,
    kind,
  };
}

async function patchDefaultPrompt() {
  if (!props.selectedId) return;
  try {
    const ds = (await api.loras.patchDataset(props.selectedId, datasetMetaPatch())) as Record<string, any>;
    patchDatasets(props.datasets.map((d) => (d.id === props.selectedId ? { ...d, ...ds } : d)));
  } catch (e: unknown) {
    toast.error(apiErrorMessage(e));
  }
}

async function saveDatasetKind() {
  if (!props.selectedId) return;
  const kind = datasetKindEdit.value;
  if (kind !== 'concept' && kind !== 'style') return;
  if (kind === (selectedDataset.value?.kind || 'concept')) return;
  try {
    const ds = (await api.loras.patchDataset(props.selectedId, datasetMetaPatch())) as Record<string, any>;
    patchDatasets(props.datasets.map((d) => (d.id === props.selectedId ? { ...d, ...ds } : d)));
  } catch (e: unknown) {
    toast.error(apiErrorMessage(e));
  }
}

async function saveDatasetName() {
  if (!props.selectedId) return;
  const name = datasetNameEdit.value.trim();
  if (!name || name === selectedDataset.value?.name) return;
  try {
    const ds = (await api.loras.patchDataset(props.selectedId, datasetMetaPatch(name))) as Record<string, any>;
    patchDatasets(props.datasets.map((d) => (d.id === props.selectedId ? { ...d, ...ds } : d)));
  } catch (e: unknown) {
    toast.error(apiErrorMessage(e));
  }
}

async function saveTriggerWord() {
  if (!props.selectedId) return;
  const triggerWord = triggerWordEdit.value.trim();
  if (triggerWord === String(selectedDataset.value?.trigger_word || '')) return;
  try {
    const ds = (await api.loras.patchDataset(props.selectedId, datasetMetaPatch())) as Record<string, any>;
    patchDatasets(props.datasets.map((d) => (d.id === props.selectedId ? { ...d, ...ds } : d)));
  } catch (e: unknown) {
    toast.error(apiErrorMessage(e));
  }
}

async function saveSkipped(files: string[], done: string) {
  if (!props.selectedId || skippingImages.value) return;
  skippingImages.value = true;
  try {
    const ds = (await api.loras.setSkippedImages(props.selectedId, files)) as Record<string, any>;
    patchDatasets(props.datasets.map((d) => (d.id === props.selectedId ? { ...d, ...ds } : d)));
    toast.success(done);
  } catch (e: unknown) {
    toast.error(apiErrorMessage(e));
  } finally {
    skippingImages.value = false;
  }
}

async function skipFiles(files: string[]) {
  const next = [...skippedFiles.value];
  for (const file of files) {
    if (!next.includes(file)) next.push(file);
  }
  await saveSkipped(next, t('loraTrain.gateSkipDone', { count: files.length }));
}

async function restoreFile(file: string) {
  await saveSkipped(
    [...skippedFiles.value].filter((item) => item !== file),
    t('loraTrain.gateRestoreDone')
  );
}

async function deleteBlocked() {
  const files = [...blockedFiles.value];
  if (!props.selectedId || !files.length || deletingBlocked.value) return;
  try {
    await confirm(
      t('loraTrain.gateDeleteAllConfirm', { count: files.length }),
      t('loraTrain.gateDeleteAll'),
      { type: 'warning' }
    );
  } catch {
    return;
  }
  deletingBlocked.value = true;
  try {
    let ds: Record<string, any> | null = null;
    for (const file of files) {
      ds = (await api.loras.deleteDatasetImage(props.selectedId, file)) as Record<string, any>;
    }
    if (ds) patchDatasets(props.datasets.map((d) => (d.id === props.selectedId ? { ...d, ...ds } : d)));
    toast.success(t('loraTrain.gateDeleteAllDone', { count: files.length }));
  } catch (e: unknown) {
    toast.error(apiErrorMessage(e));
  } finally {
    deletingBlocked.value = false;
  }
}

async function removeImage(file: string) {
  if (!props.selectedId) return;
  try {
    await confirm(
      t('loraTrain.removeImageConfirm'),
      t('loraTrain.removeImage'),
      { type: 'warning' }
    );
  } catch {
    return;
  }
  try {
    const ds = (await api.loras.deleteDatasetImage(props.selectedId, file)) as Record<string, any>;
    patchDatasets(props.datasets.map((d) => (d.id === props.selectedId ? { ...d, ...ds } : d)));
    toast.success(t('loraTrain.removeImageDone'));
  } catch (e: unknown) {
    toast.error(apiErrorMessage(e));
  }
}

async function deleteDataset(id: string, name?: string) {
  const datasetId = (id || '').trim();
  if (!datasetId || deletingDataset.value) return;
  const label = (name || datasetId).trim();
  try {
    await confirm(
      t('loraTrain.deleteDatasetConfirm', { name: label }),
      t('loraTrain.deleteDataset'),
      { type: 'warning' }
    );
  } catch {
    return;
  }
  deletingDataset.value = true;
  try {
    await api.loras.deleteDataset(datasetId);
    const next = props.datasets.filter((d) => d.id !== datasetId);
    patchDatasets(next);
    if (props.selectedId === datasetId) {
      emit('update:selectedId', next[0]?.id || '');
    }
    toast.success(t('loraTrain.deleteDatasetDone'));
  } catch (e: unknown) {
    toast.error(apiErrorMessage(e));
  } finally {
    deletingDataset.value = false;
  }
}

defineExpose({
  refreshDatasetDetail,
  saveDraft,
  async createDatasetWithName(name: string) {
    newDatasetName.value = name;
    await createDataset();
  },
});
</script>

<style scoped>
.lora-dataset-panel {
  display: flex;
  gap: 16px;
  min-height: 380px;
}

.lora-dataset-panel__guide {
  border-radius: 12px;
  border: 0.5px solid var(--dq-border-subtle);
  background: color-mix(in srgb, var(--dq-bg-base) 55%, var(--dq-fill-secondary));
  padding: 10px 12px;
}

.lora-dataset-panel__guide summary {
  cursor: pointer;
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  color: var(--dq-label-primary);
}

.lora-dataset-panel__guide p,
.lora-dataset-panel__guide .lora-dataset-panel__field {
  margin: 8px 0 0;
  font-size: var(--dq-font-size-caption);
  line-height: 1.5;
  color: var(--dq-label-secondary);
}

.lora-dataset-panel__trigger-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.lora-dataset-panel__trigger-row :deep(.dq-input),
.lora-dataset-panel__trigger-row > :first-child {
  flex: 1;
  min-width: 0;
}

.lora-dataset-panel__toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
}

.lora-dataset-panel__trigger-status {
  margin: 8px 0 0;
  font-size: var(--dq-font-size-caption);
  line-height: 1.5;
}

.lora-dataset-panel__trigger-status.is-ready { color: var(--dq-success); }
.lora-dataset-panel__trigger-status.is-warning { color: var(--dq-warning); }

.lora-dataset-panel__list {
  width: 248px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 14px;
  border-radius: 14px;
  background: color-mix(in srgb, var(--dq-bg-base) 45%, var(--dq-fill-secondary));
  border: 0.5px solid var(--dq-border-subtle);
}

.lora-dataset-panel__list-head {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.lora-dataset-panel__list-title {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  color: var(--dq-label-secondary);
  letter-spacing: var(--dq-tracking-wide);
}

.lora-dataset-panel__persist-note {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
  color: var(--dq-label-tertiary);
}

.lora-dataset-panel__search {
  width: 100%;
}

.lora-dataset-panel__ready-tag {
  margin-left: 4px;
}

.lora-dataset-panel__list-scroll {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.lora-dataset-panel__list-item {
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 8px 10px;
  border: 0.5px solid transparent;
  border-radius: var(--dq-radius-control-sm);
  background: transparent;
  cursor: pointer;
  text-align: left;
  transition: background 0.15s ease, border-color 0.15s ease;
}

.lora-dataset-panel__list-delete {
  flex-shrink: 0;
  opacity: 0;
  transition: opacity 0.15s ease;
}

.lora-dataset-panel__list-item:hover .lora-dataset-panel__list-delete,
.lora-dataset-panel__list-item.is-active .lora-dataset-panel__list-delete {
  opacity: 1;
}

.lora-dataset-panel__list-thumb {
  width: 36px;
  height: 36px;
  flex-shrink: 0;
  object-fit: cover;
  border-radius: var(--dq-radius-control-sm);
  border: 0.5px solid var(--dq-border-subtle);
  background: var(--dq-bg-base);
}

.lora-dataset-panel__list-thumb--empty {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: var(--dq-label-tertiary);
}

.lora-dataset-panel__list-body {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 2px;
  min-width: 0;
}

.lora-dataset-panel__list-item:hover {
  background: var(--dq-fill-tertiary);
}

.lora-dataset-panel__list-item.is-active {
  background: color-mix(in srgb, var(--dq-accent) 12%, var(--dq-fill-secondary));
  border-color: color-mix(in srgb, var(--dq-accent) 35%, transparent);
}

.lora-dataset-panel__list-item.is-ready:not(.is-active) {
  border-color: color-mix(in srgb, var(--dq-success) 25%, transparent);
}

.lora-dataset-panel__list-name {
  font-size: var(--dq-font-size-body);
  font-weight: 500;
  color: var(--dq-label-primary);
  word-break: break-word;
}

.lora-dataset-panel__list-meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
}

.lora-dataset-panel__list-empty {
  flex: 1;
  padding: 12px 0;
}

.lora-dataset-panel__list-foot {
  padding-top: 4px;
  border-top: 0.5px solid var(--dq-border-subtle);
}

.lora-dataset-panel__workspace {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.lora-dataset-panel__meta {
  display: grid;
  grid-template-columns: minmax(180px, 240px) minmax(0, 1fr);
  gap: 12px 20px;
  align-items: start;
  padding: 14px 16px;
  border-radius: 14px;
  border: 0.5px solid var(--dq-border-subtle);
  background: color-mix(in srgb, var(--dq-bg-base) 55%, var(--dq-fill-secondary));
}

.lora-dataset-panel__kind-row {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}

.lora-dataset-panel__kind-select {
  width: 100%;
}

.lora-dataset-panel__field-hint {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
  line-height: 1.4;
}

.lora-dataset-panel__health {
  margin-top: -4px;
}

.lora-dataset-panel__workspace-head {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-end;
  justify-content: space-between;
  gap: 10px;
}

.lora-dataset-panel__workspace-head-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.lora-dataset-panel__name-row {
  flex: 1;
  min-width: 180px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.lora-dataset-panel__name-input {
  max-width: 360px;
}

.lora-dataset-panel__dropzone {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 4px;
  padding: 20px 16px;
  border-radius: var(--dq-radius-group);
  border: 1.5px dashed var(--dq-border);
  background: var(--dq-fill-secondary);
  cursor: pointer;
  transition: border-color 0.15s ease, background 0.15s ease;
}

.lora-dataset-panel__dropzone:hover,
.lora-dataset-panel__dropzone.is-dragover {
  border-color: var(--dq-accent);
  background: color-mix(in srgb, var(--dq-accent) 8%, var(--dq-fill-secondary));
}

.lora-dataset-panel__dropzone.is-disabled {
  opacity: 0.5;
  pointer-events: none;
}

.lora-dataset-panel__dropzone.is-uploading {
  pointer-events: none;
  border-style: solid;
  border-color: var(--dq-accent);
}

.lora-dataset-panel__dropzone.is-compact {
  flex: 1 1 220px;
  flex-direction: row;
  justify-content: flex-start;
  gap: 8px;
  padding: 8px 12px;
  min-height: 0;
}

.lora-dataset-panel__toolbar .lora-dataset-panel__actions {
  flex: 1 1 280px;
  justify-content: flex-end;
}

.lora-dataset-panel__dropzone-icon {
  color: var(--dq-label-tertiary);
}

.lora-dataset-panel__dropzone-title {
  margin: 0;
  font-size: var(--dq-font-size-body);
  font-weight: 500;
  color: var(--dq-label-primary);
}

.lora-dataset-panel__dropzone-hint {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
}

.lora-dataset-panel__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.lora-dataset-panel__grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(148px, 1fr));
  gap: 12px;
}

.lora-dataset-panel__cell {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.lora-dataset-panel__cell.is-gate-blocked {
  padding: 6px;
  border-radius: var(--dq-radius-group);
  outline: 1px solid color-mix(in srgb, var(--dq-danger) 55%, transparent);
}

.lora-dataset-panel__cell.is-skipped {
  opacity: 0.62;
}

.lora-dataset-panel__gate-bar {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 10px 12px;
  border-radius: var(--dq-radius-group);
  background: color-mix(in srgb, var(--dq-danger) 12%, var(--dq-bg-base));
  border: 1px solid color-mix(in srgb, var(--dq-danger) 35%, transparent);
}

.lora-dataset-panel__gate-copy {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
  color: var(--dq-label-primary);
}

.lora-dataset-panel__gate-bar-actions,
.lora-dataset-panel__gate-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
}

.lora-dataset-panel__gate-skipped {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
}

.lora-dataset-panel__cell-media {
  position: relative;
}

.lora-dataset-panel__cell-media img {
  width: 100%;
  aspect-ratio: 1;
  object-fit: cover;
  border-radius: var(--dq-radius-group);
  border: 0.5px solid var(--dq-border);
  background: var(--dq-bg-base);
}

.lora-dataset-panel__cell-media.is-vlm-good img {
  border: 2px solid color-mix(in srgb, var(--dq-success) 70%, var(--dq-border));
}

.lora-dataset-panel__cell-media.is-vlm-fair img {
  border: 2px solid color-mix(in srgb, var(--dq-warning) 75%, var(--dq-border));
}

.lora-dataset-panel__cell-media.is-vlm-poor img {
  border: 2px solid color-mix(in srgb, var(--dq-danger) 75%, var(--dq-border));
}

.lora-dataset-panel__vlm-mark {
  position: absolute;
  left: 4px;
  bottom: 4px;
  padding: 2px 6px;
  border-radius: var(--dq-radius-control-sm);
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  line-height: 1.2;
  color: var(--dq-color-white);
  pointer-events: none;
  box-shadow: 0 1px 4px color-mix(in srgb, var(--dq-bg-base) 35%, transparent);
}

.lora-dataset-panel__vlm-mark.is-good {
  background: color-mix(in srgb, var(--dq-success) 88%, #000);
}

.lora-dataset-panel__vlm-mark.is-fair {
  background: color-mix(in srgb, var(--dq-warning) 92%, #000);
}

.lora-dataset-panel__vlm-mark.is-poor {
  background: color-mix(in srgb, var(--dq-danger) 90%, #000);
}

.lora-dataset-panel__cell-media {
  overflow: hidden;
  border-radius: var(--dq-radius-group);
}

.lora-dataset-panel__face-window {
  position: absolute;
  box-sizing: border-box;
  border: 2px dashed color-mix(in srgb, var(--dq-primary) 85%, #fff);
  box-shadow: 0 0 0 9999px color-mix(in srgb, var(--dq-bg-base) 45%, transparent);
  pointer-events: none;
}

.lora-dataset-panel__face-mark {
  position: absolute;
  left: 4px;
  top: 4px;
  padding: 2px 6px;
  border-radius: var(--dq-radius-control-sm);
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  line-height: 1.2;
  color: var(--dq-color-white);
  pointer-events: none;
  box-shadow: 0 1px 4px color-mix(in srgb, var(--dq-bg-base) 35%, transparent);
}

.lora-dataset-panel__face-mark.is-crop {
  background: color-mix(in srgb, var(--dq-primary) 88%, #000);
}

.lora-dataset-panel__face-mark.is-none,
.lora-dataset-panel__face-mark.is-tiny {
  background: color-mix(in srgb, var(--dq-danger) 90%, #000);
}

.lora-dataset-panel__face-mark.is-multi {
  background: color-mix(in srgb, var(--dq-warning) 92%, #000);
}

.lora-dataset-panel__vlm-legend-item.is-face-crop {
  border-color: color-mix(in srgb, var(--dq-primary) 45%, transparent);
  color: var(--dq-primary);
}

.lora-dataset-panel__vlm-legend-item.is-face-none {
  border-color: color-mix(in srgb, var(--dq-danger) 45%, transparent);
  color: var(--dq-danger);
}

.lora-dataset-panel__vlm-legend-item.is-face-multi {
  border-color: color-mix(in srgb, var(--dq-warning) 45%, transparent);
  color: var(--dq-warning, var(--dq-label-primary));
}

.lora-dataset-panel__vlm-legend {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
}

.lora-dataset-panel__vlm-legend-label {
  font-weight: 500;
  color: var(--dq-label-secondary);
}

.lora-dataset-panel__vlm-legend-item {
  padding: 2px 8px;
  border-radius: var(--dq-radius-control-sm);
  border: 1px solid var(--dq-border-subtle);
}

.lora-dataset-panel__vlm-legend-item.is-good {
  border-color: color-mix(in srgb, var(--dq-success) 45%, transparent);
  color: var(--dq-success);
}

.lora-dataset-panel__vlm-legend-item.is-fair {
  border-color: color-mix(in srgb, var(--dq-warning) 45%, transparent);
  color: var(--dq-warning, var(--dq-label-primary));
}

.lora-dataset-panel__vlm-legend-item.is-poor {
  border-color: color-mix(in srgb, var(--dq-danger) 45%, transparent);
  color: var(--dq-danger);
}

.lora-dataset-panel__cell-remove {
  position: absolute;
  top: 4px;
  right: 4px;
  background: color-mix(in srgb, var(--dq-bg-base) 85%, transparent) !important;
  border-radius: var(--dq-radius-control-sm);
}

.lora-dataset-panel__field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.lora-dataset-panel__label {
  font-size: var(--dq-font-size-caption);
  font-weight: 500;
  color: var(--dq-label-secondary);
}

.lora-dataset-panel__field-hint {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
  line-height: 1.4;
}

.lora-dataset-panel__import-hint {
  margin: 0 0 12px;
  font-size: var(--dq-font-size-body);
  color: var(--dq-label-tertiary);
}

.lora-dataset-panel__pending-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 12px;
}

.lora-dataset-panel__pending-tag {
  display: inline-flex;
  align-items: center;
  gap: 2px;
  max-width: 100%;
  padding: 2px 4px 2px 8px;
  border-radius: var(--dq-radius-control-sm);
  background: var(--dq-fill-secondary);
  border: 0.5px solid var(--dq-border-subtle);
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
}

.lora-dataset-panel__pending-tag-label {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 200px;
}

.lora-dataset-panel__dialog-field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

@media (max-width: 860px) {
  .lora-dataset-panel {
    flex-direction: column;
  }

  .lora-dataset-panel__list {
    width: 100%;
    max-height: 240px;
  }

  .lora-dataset-panel__meta {
    grid-template-columns: 1fr;
  }

  .lora-dataset-panel__toolbar {
    flex-direction: column;
    align-items: stretch;
  }

  .lora-dataset-panel__toolbar .lora-dataset-panel__actions {
    justify-content: flex-start;
  }
}
</style>
