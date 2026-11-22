<template>
  <StudioLayout
    class="studio-create-page"
    :freeform="viewMode === 'canvas'"
    hide-composer-bar
    @scroll="onCanvasScroll"
  >
    <template #filters>
      <div v-if="viewMode === 'grid'" class="create-filters create-filters--list">
        <DqInput
          v-model="stripSearchText"
          size="sm"
          class="create-filters__search"
          :placeholder="$t('canvas.stripSearchPlaceholder')"
          clearable
        >
          <template #prefix><DqIcon><Search /></DqIcon></template>
        </DqInput>
        <div class="create-filters__spacer" aria-hidden="true" />
        <div class="create-filters__actions">
          <DqButton size="sm" @click="onNewCanvas(() => onViewModeChange('canvas'))">
            {{ $t('canvas.newCanvas') }}
          </DqButton>
          <DqIconButton
            type="text"
            size="sm"
            :label="$t('common.refresh')"
            @click="refreshStripList"
          >
            <DqIcon><Refresh /></DqIcon>
          </DqIconButton>
        </div>
      </div>
      <div v-else class="create-filters create-filters--detail">
        <DqButton size="sm" type="text" @click="onBackToStripList(onViewModeChange)">
          ← {{ $t('canvas.backToCanvasList') }}
        </DqButton>
        <span class="create-filters__session-title" :title="canvasSessionTitle">
          {{ canvasSessionTitle }}
        </span>
        <div class="create-filters__spacer" />
        <DqIconButton
          type="text"
          size="sm"
          :label="$t('common.refresh')"
          @click="refreshGallery"
        >
          <DqIcon><Refresh /></DqIcon>
        </DqIconButton>
        <DqButton
          type="primary"
          size="sm"
          :class="{ 'is-busy': composerBusy }"
          @click="openGenerationComposer()"
        >
          {{ $t('studio.openComposer') }}
        </DqButton>
      </div>
    </template>
    <template #canvas>
      <StudioCanvasStripList
        v-if="viewMode === 'grid'"
        media="audio"
        :sessions="canvasSessions"
        :search-text="stripSearchText"
        :session-asset-ids="sessionAssetIdsBySession"
        :loading="!canvasReady"
        :active-tasks="activeAudioTasks"
        :gallery-by-asset-id="galleryByAssetId"
        @open="(id) => onStripOpen(id, onViewModeChange)"
        @preview="onStripPreview"
        @create="onStripCreate"
        @rename="onStripRename"
        @delete="onStripDelete"
      />
      <InfiniteCanvas
        v-else
        ref="infiniteCanvasRef"
        :items="enrichedGalleryItems"
        media="audio"
        hide-composer-entry
        @card-action="onCardAction"
        @download="downloadAudioItem"
        @toggle-grid-view="() => onBackToStripList(onViewModeChange)"
        @node-selected="onCanvasNodeSelected"
        @session-ready="onCanvasSessionReady"
        @open-preview="onCanvasOpenPreview"
        @composer-restore="onCanvasComposerRestore"
        @overlay-cleared="onCanvasOverlayCleared"
        @use-as-cover-source="onCanvasUseAsCoverSource"
        @open-composer="onCanvasOpenComposer"
        @request-close-composer="closeComposerDrawer()"
      />
    </template>
  </StudioLayout>

  <StudioComposerHost
    v-model:open="composerDrawerOpen"
    :drawer-title="composerDrawerTitle"
  >
    <AudioComposer
        v-model="params.prompt"
        v-model:title="params.title"
        v-model:work-mode="audioWorkTab"
        v-model:model="selectedModelVersion"
        v-model:duration="composerDuration"
        v-model:batch-count="params.n"
        :work-mode-options="composerWorkModeOptions"
        :composer-busy="composerBusy"
        :submitting="queueSubmitting"
        :generating="generating"
        :can-generate="audioWorkTab === 'create' ? canGenerateCreate : canGenerateEdit"
        :generate-label="generateLabel"
        :model-options="audioModelSelectOptions"
        :duration-options="durationOptions"
        :styles="filteredPresets"
        :params="params"
        @update:params="applyAudioComposerParams"
        :has-custom-params="hasCustomParams"
        :show-negative-prompt="supportsNegativePrompt"
        :show-lyrics="true"
        :show-codec-controls="false"
        :show-cover-fidelity="audioWorkTab === 'cover'"
        :reference-media="coverReferenceMedia"
        :source-duration="coverSourceDuration"
        :current-model-config="advancedParamModelConfig"
        :compatible-loras="compatibleLoras"
        @generate="onAudioComposerGenerate"
        @open-lm="openLmAssistant"
        @pick-reference="onPickCoverSource"
        @remove-reference="clearCoverSource"
        @model-change="onModelChange"
        @reset-defaults="restoreDefaults"
        :lyrics-loading="isGeneratingLyrics"
        :brief-enhancing="isEnhancing"
        @generate-lyrics="onGenerateLyrics"
        @enhance-brief="onEnhanceMusicBrief"
        :prompt-apply-preview="promptApplyPreview"
        :lyrics-apply-preview="lyricsApplyPreview"
        @prompt-apply-replace="onBriefApplyReplace"
        @prompt-apply-append="onBriefApplyAppend"
        @prompt-apply-dismiss="briefApply.clear()"
        @lyrics-apply-replace="onLyricsApplyReplace"
        @lyrics-apply-append="onLyricsApplyAppend"
        @lyrics-apply-dismiss="lyricsApply.clear()"
      />
  </StudioComposerHost>

  <AudioLmPanel
    v-model:open="lmPanelOpen"
    :initial="params"
    :result="lmResult"
    :submitting="lmSubmitting"
    :progress="lmProgress"
    :error="lmError"
    @run="runLmAssistant"
    @apply="applyLmResult"
  />

  <!-- Audio preview dialog -->
  <GalleryPreviewDialog
    v-model:open="audioPreviewVisible"
    v-model:index="selectedAudioIndex"
    :items="previewItems"
    media="audio"
    :transfer-targets="previewTransferTargets"
    @transfer="onPreviewTransfer"
  />

  <DqDialog
    v-model:open="stripRenameOpen"
    :title="$t('canvas.renameSession')"
    width="360px"
  >
    <DqInput
      v-model="stripRenameDraft"
      size="sm"
      :placeholder="$t('canvas.sessionSelect')"
      @keydown.enter="commitStripRename"
    />
    <template #footer>
      <DqButton size="sm" @click="stripRenameOpen = false">{{ $t('common.cancel') }}</DqButton>
      <DqButton type="primary" @click="commitStripRename">{{ $t('common.confirm') }}</DqButton>
    </template>
  </DqDialog>

  <StudioLineagePanel
    v-model:modelValue="showLineageDrawer"
    :asset-id="lineageTargetAssetId"
    :on-canvas-ids="canvasAssetIdsOnCanvas"
    @focus-asset="onLineageFocusAsset"
  />
</template>

<script setup lang="ts">
// @ts-nocheck — legacy create view; narrow types in a follow-up pass
import { ref, reactive, computed, watch, onMounted, onUnmounted, nextTick, inject, unref, type Ref } from 'vue';
import { useRouter } from 'vue-router';
import { toast } from '@/utils/feedback';
import { api, taskIdFromSubmitResponse } from '@/utils/api';
import { $tt, $mn, $md, $mvn, $pn } from '@/utils/i18n';
import { useTasksStore } from '@/stores/tasks';
import { useRegistryStore } from '@/stores/registry';
import { useComposerLlm } from '@/composables/useComposerLlm';
import { DQ_STORAGE, getItem, setItem, type StorageKey } from '@/utils/storage';
import { applyPromptDraft, consumePromptDraft } from '@/utils/promptApply';
import { usePromptApplyOffer } from '@/composables/usePromptApplyOffer';

import { useModelRegistryFilters } from '@/composables/useModelRegistryFilters';
import { applyModelVersionFilters } from '@/utils/modelPickerFilters';
import { compareModelPickerRows } from '@/utils/modelPickerSort';
import { modelSelectOption } from '@/utils/modelSelectOption';
import { warnIfRiskyMemory } from '@/composables/memoryHint';
import type { SystemInfo, Task, GalleryItem } from '@/types';
import { assetDisplayLabel, previewDisplayCaption } from '@/utils/assetDisplay';
import { seedForSubmit } from '@/utils/seed';
import {
  buildAudioEditPayload,
  buildAudioGenerationPayload,
  isAudioWorkMode,
  parseAudioTracks,
  supportsAudioAction,
  validateAudioEditInput,
  type AudioWorkMode,
} from '@/utils/audioTasks';
import { Refresh, Search } from '@danqing/dq-shell';
import StudioLayout from '@/components/studio/StudioLayout.vue';
import StudioCanvasStripList from '@/components/studio/StudioCanvasStripList.vue';
import AudioComposer from '@/components/studio/AudioComposer.vue';
import AudioLmPanel, { type AudioLmResult } from '@/components/audio/AudioLmPanel.vue';
import StudioComposerHost from '@/components/studio/StudioComposerHost.vue';
import InfiniteCanvas from '@/components/studio/InfiniteCanvas.vue';
import StudioLineagePanel from '@/components/studio/StudioLineagePanel.vue';
import GalleryPreviewDialog from '@/components/gallery/GalleryPreviewDialog.vue';
import { useStudioGallery } from '@/composables/useStudioGallery';
import { canvasAutoAddEnabled, useCanvasStore } from '@/composables/useCanvasStore';
import { useCanvasStripIa } from '@/composables/useCanvasStripIa';
import { useComposerDrawer } from '@/composables/useComposerDrawer';
import {
  activateCanvasViewForResults,
  maybeShowCanvasWorkspaceHint,
} from '@/utils/canvasWorkspaceHint';
import { previewUrlForGalleryItem } from '@/utils/canvasAssets';
import { isAudioLyricsRequired, audioLyricsRequiredHintKey, resolveVocalLanguageForSubmit } from '@/utils/audioLyrics';
import { ACTIVE_COMPOSER_TASK_STATUSES, splitComposerPromptLines } from '@/utils/composerQueue';
import {
  applyLoraComposeOverrides,
  withRegistryLoraMeta,
  type CompatibleLoraRow,
} from '@/utils/loraAdapterMeta';

const tasksStore = useTasksStore();
const registryStore = useRegistryStore();
const router = useRouter();
const systemInfo = inject<Ref<SystemInfo>>('systemInfo');

// ---- Helpers (migrated from window globals) ----
const LEGACY_AUDIO_PROMPT_DRAFT_KEY = 'dq-studio.audio-create-prompt-draft.v3';

function parseModelVersion(s: string) {
  const [m, v] = (s || '').split('|');
  return { model: m || '', version: v || '', modelKey: m || '' };
}

function isAudioModel(config: any) {
  return config && config.media === 'audio';
}

const TSU = {
  tagType: (s: string) => {
    if (s === 'running') return 'primary';
    if (s === 'completed') return 'success';
    if (s === 'failed') return 'danger';
    return 'info';
  },
  statusText: (s: string) => s || ''
};

// ---- Work mode (create vs cover are exclusive drawers — no in-drawer tab switch) ----
const audioWorkTab = ref<AudioWorkMode>('create');
const composerWorkModeOptions = computed(() => {
  if (audioWorkTab.value === 'create') return [] as Array<{ label: string; value: string }>;
  return ['cover', 'repaint', 'lego', 'extract', 'complete']
    .map((value) => ({ value, label: $tt(`action.audio.${value}`) }));
});

// ---- State ----
const generating = ref(false);
const queueSubmitting = ref(false);
const modelsLoading = ref(false);
const previewPlayerRef = ref(null);
const previewIsPlaying = ref(false);
const previewDurationSec = ref(0);

const modelRegistry = ref<Record<string, any>>({});
const modelsDetailedStatus = ref<Record<string, any>>({});

const params = reactive({
  model: '',
  title: '',
  prompt: '',
  negative_prompt: '',
  duration: 30 as number | null,
  instrumental: false,
  lyrics: '',
  vocal_language: '',
  vocal_type: '',
  bpm: null as number | null,
  key_scale: '',
  time_signature: '',
  steps: 50,
  guidance: 3.0,
  seed: null as number | null,
  n: 1,
  audio_format: 'wav',
  denoise: 'off',
  lm_expansion: 'auto',
  lora: '',
  lora_scale: 0.8,
  source_fidelity: 1.0,
  cover_variant: 'cover',
  cover_noise_strength: 0,
  timbre_from_source: false,
  region_start_sec: null as number | null,
  region_end_sec: null as number | null,
  repaint_injection_ratio: 1.0,
  repaint_crossfade_frames: 8,
  tracks: [] as string[],
  dcw_enabled: 'auto',
  dcw_mode: 'double',
  dcw_scaler: 0.05,
  dcw_high_scaler: 0.02,
  audio_code_indices: null as number[] | null,
});

const coverSourceSrc = ref('');
const coverSourceFile = ref<File | null>(null);
const coverSourceAssetPath = ref('');
const coverSourceName = ref('');
const coverSourceDuration = ref(0);
let coverSourceProbe = 0;

const currentTask = reactive({ id: '', progress: 0, step: null as number | null, total: null as number | null, status: '' });
const previewAudioSrc = ref('');
const previewPrompt = ref('');
const previewLyrics = ref('');
const previewLyricsDownload = ref('');
const previewAudioKey = ref(0);
const recentAudio = reactive<Array<any>>([]);
const previewArtHue = computed(() => artHueForPrompt(previewPrompt.value));

const previewPlayerSubtitle = computed(() => {
  const parts: string[] = [];
  if (currentModelDisplayName.value) parts.push(currentModelDisplayName.value);
  if (previewDurationSec.value > 0) parts.push(formatTime(previewDurationSec.value));
  else if (params.duration) parts.push(formatTime(params.duration));
  return parts.join(' · ');
});

function artHueForPrompt(text: string) {
  let h = 0;
  const s = String(text || 'audio');
  for (let i = 0; i < s.length; i += 1) {
    h = (h * 31 + s.charCodeAt(i)) % 360;
  }
  return h;
}

// ---- Model display name ----
const currentModelKey = computed(() => {
  if (!params.model) return '';
  const colon = params.model.lastIndexOf(':');
  return colon > 0 ? params.model.slice(0, colon) : params.model;
});

const currentModelConfig = computed(() => {
  const mk = currentModelKey.value;
  if (!mk) return null;
  return modelRegistry.value[mk] || null;
});

const currentModelDisplayName = computed(() => {
  const c = currentModelConfig.value;
  const mk = currentModelKey.value;
  return $mn(c, mk) || params.model || '';
});

const modelReady = computed(() => {
  const mk = currentModelKey.value;
  if (!mk) return false;
  const ds = modelsDetailedStatus.value[mk];
  if (ds && ds.status) return ds.status === 'ready';
  return true;
});

const selectedModelNotReady = computed(() => {
  return !!params.model && !modelReady.value;
});

const supportsNegativePrompt = computed(() => {
  const c = currentModelConfig.value;
  return c && c.parameters && c.parameters.negative_prompt_support === true;
});

const hasCustomParams = computed(() => {
  const p = currentModelConfig.value?.parameters || {};
  const stepsDef = p.steps?.default ?? 50;
  const guidanceDef = p.guidance?.default ?? 3.0;
  const durationDef = p.duration?.default ?? 30;
  const formatDef = (p.audio_formats && p.audio_formats[0]) || audioFormats.value[0] || 'wav';
  const denoiseDef = p.denoise?.default ?? 'off';
  const lmDef = p.lm_expansion?.default ?? 'auto';
  return params.steps !== stepsDef || params.guidance !== guidanceDef
    || params.seed !== null || params.n !== 1 || params.duration !== durationDef
    || params.audio_format !== formatDef || params.denoise !== denoiseDef
    || params.lm_expansion !== lmDef
    || params.cover_variant !== 'cover'
    || params.source_fidelity !== 1
    || params.cover_noise_strength !== 0
    || params.timbre_from_source
    || params.region_start_sec != null
    || params.region_end_sec != null
    || params.repaint_injection_ratio !== 1
    || params.repaint_crossfade_frames !== 8
    || params.tracks.length > 0
    || params.dcw_enabled !== (p.dcw_enabled?.default ?? 'auto')
    || params.dcw_mode !== (p.dcw_mode?.default ?? 'double')
    || params.dcw_scaler !== (p.dcw_scaler?.default ?? 0.05)
    || params.dcw_high_scaler !== (p.dcw_high_scaler?.default ?? 0.02)
    || !!params.audio_code_indices?.length
    || !!params.lora;
});

const compatibleLoras = ref<CompatibleLoraRow[]>([]);

function syncAudioLoraSelection() {
  if (!params.lora) return;
  const row = compatibleLoras.value.find((item) => String(item.id) === String(params.lora));
  if (!row || row.ready === false) return;
  const bare = String(row.id || params.lora).split(':')[0];
  applyLoraComposeOverrides(
    params as unknown as Record<string, unknown>,
    row,
    modelRegistry.value[bare],
  );
}

async function loadCompatibleAdapters(modelKey: string) {
  if (!modelKey) {
    compatibleLoras.value = [];
    return;
  }
  try {
    const loras = await api.settings.getCompatibleLoras(modelKey);
    compatibleLoras.value = ((loras as CompatibleLoraRow[]) || []).map((row) => {
      const bare = String(row.id || '').split(':')[0];
      return withRegistryLoraMeta(row, modelRegistry.value[bare]);
    });
    if (
      params.lora &&
      !compatibleLoras.value.some(
        (l) => String(l.id) === String(params.lora) && l.ready !== false,
      )
    ) {
      params.lora = '';
    }
    syncAudioLoraSelection();
  } catch (e) {
    console.error('Failed to load compatible adapters:', e);
    compatibleLoras.value = [];
  }
}

watch(() => params.lora, () => {
  syncAudioLoraSelection();
});

// ---- Registry-driven capability flags ----
const supportsBpm = computed(() => currentModelConfig.value?.parameters?.supports_bpm === true);
const supportsKeyScale = computed(() => currentModelConfig.value?.parameters?.supports_key_scale === true);
const supportsTimeSignature = computed(
  () => currentModelConfig.value?.parameters?.supports_time_signature === true,
);

const audioFormats = computed(() => {
  const allowed = currentModelConfig.value?.parameters?.audio_formats;
  if (Array.isArray(allowed) && allowed.length > 0) {
    return allowed.map((f) => String(f).toLowerCase());
  }
  return ['mp3', 'flac', 'wav', 'opus', 'aac'];
});

const promptPlaceholder = computed(() => $tt('audio.promptPlaceholder'));

const advancedParamModelConfig = computed(() => currentModelConfig.value);

const supportsVocalType = computed(() => {
  return currentModelConfig.value?.parameters?.supports_vocal_type === true;
});

const vocalTypes = computed(() => [
  { label: $tt('audio.vocalTypeMale'), value: 'male' },
  { label: $tt('audio.vocalTypeFemale'), value: 'female' },
  { label: $tt('audio.vocalTypeChorus'), value: 'chorus' },
  { label: $tt('audio.vocalTypeDuet'), value: 'duet' },
]);
const musicalKeys = [
  'C Major', 'C# Major', 'D Major', 'D# Major', 'E Major', 'F Major', 'F# Major',
  'G Major', 'G# Major', 'A Major', 'A# Major', 'B Major',
  'C Minor', 'C# Minor', 'D Minor', 'D# Minor', 'E Minor', 'F Minor', 'F# Minor',
  'G Minor', 'G# Minor', 'A Minor', 'A# Minor', 'B Minor',
];
const timeSignatures = [
  { label: '2/4', value: '2' }, { label: '3/4', value: '3' },
  { label: '4/4', value: '4' }, { label: '6/8', value: '6' },
];

// ---- Model selector ----
const selectedModelVersion = ref('');

const { commercialOnly: modelFilterCommercialOnly } = useModelRegistryFilters();

const allAudioVersions = computed(() => {
  const rows: Array<Record<string, unknown>> = [];
  for (const [modelKey, config] of Object.entries(modelRegistry.value)) {
    if (!isAudioModel(config)) continue;
    if (!supportsAudioAction(config, audioWorkTab.value)) continue;
    const versions = config.versions || {};
    const ds = modelsDetailedStatus.value[modelKey] || {};
    const versionStatuses = ds.versions || {};
    for (const [versionKey, versionConfig] of Object.entries(versions)) {
      const vst = versionStatuses[versionKey] || { status: 'not_downloaded', ready: false };
      const ready =
        vst.ready === true || (ds.status === 'ready' && vst.ready !== false);
      rows.push({
        modelKey,
        versionKey,
        name: $mvn(modelKey, config, versionConfig),
        description: $md(config as { description?: string | { zh?: string; en?: string }; description_en?: string }),
        size: (versionConfig as Record<string, unknown>).size || '',
        status: ready ? 'ready' : (vst.status || 'not_downloaded'),
        ready,
        recommended: config.recommended === true,
        commercialUseAllowed: config.commercial_use_allowed === true,
      });
    }
  }
  return rows;
});

const selectedModelPickerItem = computed(() => {
  const key = selectedModelVersion.value;
  if (!key) return null;
  return allAudioVersions.value.find((item) => `${item.modelKey}|${item.versionKey}` === key) ?? null;
});

const filteredModelPickerVersions = computed(() => {
  const rows = applyModelVersionFilters(allAudioVersions.value, {
    installedOnly: true,
    commercialOnly: modelFilterCommercialOnly.value,
  });
  rows.sort(compareModelPickerRows);
  return rows;
});

const lyricsRequiredForModel = computed(() =>
  isAudioLyricsRequired(currentModelConfig.value, !!params.instrumental),
);

const lyricsMissing = computed(() =>
  audioWorkTab.value === 'create' && lyricsRequiredForModel.value && !String(params.lyrics || '').trim(),
);

const canGenerateCreate = computed(() => {
  if (!params.model) return false;
  if (!modelReady.value) return false;
  if (!params.prompt.trim()) return false;
  if (lyricsMissing.value) return false;
  return true;
});

const canGenerateEdit = computed(() => {
  if (!params.model || !modelReady.value || !supportsEditAction.value) return false;
  if (lyricsMissing.value || !hasCoverSource()) return false;
  const mode = audioWorkTab.value;
  if (mode === 'create') return true;
  return !validateAudioEditInput(mode, {
    ...params,
    sourceAssetId: coverSourceAssetPath.value || (coverSourceFile.value ? 'pending-upload' : ''),
    sourceDuration: coverSourceDuration.value,
  });
});

const supportsEditAction = computed(() => {
  return audioWorkTab.value !== 'create' && supportsAudioAction(currentModelConfig.value, audioWorkTab.value);
});

// ---- Composer adapters ----

function applyAudioComposerParams(val: Record<string, unknown>) {
  Object.assign(params, val);
}

const audioModelSelectOptions = computed(() =>
  filteredModelPickerVersions.value.map((v) => modelSelectOption(v)),
);

const coverReferenceMedia = computed(() => {
  if (audioWorkTab.value === 'create') return null;
  if (!coverSourceSrc.value) return null;
  return {
    type: 'audio',
    previewUrl: coverSourceSrc.value,
    label: coverSourceName.value || $tt('audio.coverSource'),
  };
});

function onAudioComposerGenerate() {
  if (composerBusy.value) {
    if (audioWorkTab.value === 'create') {
      void queueAudioCreateToServer();
    } else {
      void queueAudioEditToServer();
    }
    return;
  }
  if (audioWorkTab.value === 'create') {
    startGeneration();
  } else {
    startEditGeneration();
  }
}

// ---- Methods ----
async function loadModelRegistry() {
  modelsLoading.value = true;
  try {
    const [registryData, detailedStatusData] = await Promise.all([
      registryStore.load(),
      api.settings.getModelsDetailedStatus(),
    ]);
    modelRegistry.value = registryData?.models || {};
    if (detailedStatusData && typeof detailedStatusData === 'object') {
      modelsDetailedStatus.value = detailedStatusData as Record<string, unknown>;
    }
    // 恢复当前模式的模型
    const saved = getItem(getAudioModeStorageKey(audioWorkTab.value));
    if (saved) {
      const parsed = parseModelVersion(saved);
      const mk = parsed.modelKey || parsed.model || '';
      const vk = parsed.version || '';
      const key = vk ? `${mk}|${vk}` : mk;
      const stillValid = filteredModelPickerVersions.value.some(
        (r) => `${r.modelKey}|${r.versionKey}` === key && r.ready
      );
      if (stillValid) {
        selectedModelVersion.value = key;
        onModelChange(key);
      }
    }
  } catch (e) {
    console.error('Failed to load audio model registry:', e);
  } finally {
    modelsLoading.value = false;
  }
}

function applyDefaults(modelConfig: any) {
  if (!modelConfig || !modelConfig.parameters) return;
  const p = modelConfig.parameters;
  if (p.steps && p.steps.default != null) params.steps = p.steps.default;
  if (p.guidance && p.guidance.default != null) params.guidance = p.guidance.default;
  if (p.duration) {
    if (p.duration.default != null) params.duration = p.duration.default;
    const dmin = p.duration.min ?? 10;
    const dmax = p.duration.max ?? 600;
    if (params.duration < dmin) params.duration = dmin;
    if (params.duration > dmax) params.duration = dmax;
  }
  if (p.audio_formats && Array.isArray(p.audio_formats) && p.audio_formats.length > 0) {
    const fmt = String(p.audio_formats[0]).toLowerCase();
    params.audio_format = fmt;
  }
  if (!audioFormats.value.includes(String(params.audio_format || '').toLowerCase())) {
    params.audio_format = audioFormats.value[0] || 'wav';
  }
  params.denoise = p.denoise?.default ?? 'off';
  params.lm_expansion = p.lm_expansion?.default ?? 'auto';
  params.negative_prompt = '';
  params.lyrics = '';
  params.bpm = null;
  params.key_scale = '';
  params.time_signature = '';
  params.vocal_language = '';
  params.vocal_type = '';
  params.lora = '';
  params.lora_scale = 0.8;
  if (p.supports_instrumental === false) params.instrumental = false;
  params.audio_format = audioFormats.value.includes(params.audio_format) ? params.audio_format : audioFormats.value[0] || 'wav';
  params.source_fidelity = 1.0;
  params.cover_variant = 'cover';
  params.cover_noise_strength = 0;
  params.timbre_from_source = false;
  params.region_start_sec = null;
  params.region_end_sec = null;
  params.repaint_injection_ratio = p.repaint_injection_ratio?.default ?? 1.0;
  params.repaint_crossfade_frames = p.repaint_crossfade_frames?.default ?? 8;
  params.tracks = [];
  params.dcw_enabled = p.dcw_enabled?.default ?? 'auto';
  params.dcw_mode = p.dcw_mode?.default ?? 'double';
  params.dcw_scaler = p.dcw_scaler?.default ?? 0.05;
  params.dcw_high_scaler = p.dcw_high_scaler?.default ?? 0.02;
  params.audio_code_indices = null;
  if (audioWorkTab.value !== 'create') params.duration = null;
}

function getAudioModeStorageKey(mode: string): StorageKey {
  const map: Record<string, StorageKey> = {
    create: DQ_STORAGE.AUDIO_MODEL_CREATE,
    cover: DQ_STORAGE.AUDIO_MODEL_COVER,
    repaint: DQ_STORAGE.AUDIO_MODEL_REPAINT,
    lego: DQ_STORAGE.AUDIO_MODEL_LEGO,
    extract: DQ_STORAGE.AUDIO_MODEL_EXTRACT,
    complete: DQ_STORAGE.AUDIO_MODEL_COMPLETE,
  };
  return map[mode] || DQ_STORAGE.AUDIO_MODEL_CREATE;
}

function onModelChange(val: string) {
  const parsed = parseModelVersion(val);
  const mk = parsed.modelKey || parsed.model || '';
  const vk = parsed.version || '';
  params.model = vk ? mk + ':' + vk : mk;
  const mc = modelRegistry.value[mk];
  applyDefaults(mc);
  loadCompatibleAdapters(mk);
  loadPromptDraft();
  // 保存到当前模式
  setItem(getAudioModeStorageKey(audioWorkTab.value), val);
}

function randomizeSeed() {
  params.seed = Math.floor(Math.random() * 2147483647);
}

function restoreDefaults() {
  const mc = currentModelConfig.value;
  applyDefaults(mc);
  params.n = 1;
  params.seed = null;
}

const {
  isEnhancing,
  isGeneratingLyrics,
  enhance: doEnhance,
  generateLyrics,
} = useComposerLlm();
const briefApply = usePromptApplyOffer();
const lyricsApply = usePromptApplyOffer();
const promptApplyPreview = computed(() => briefApply.pending.value?.result ?? null);
const lyricsApplyPreview = computed(() => lyricsApply.pending.value?.result ?? null);

const lmPanelOpen = ref(false);
const lmSubmitting = ref(false);
const lmProgress = ref(0);
const lmError = ref('');
const lmResult = ref<AudioLmResult | null>(null);

function openLmAssistant() {
  lmError.value = '';
  lmPanelOpen.value = true;
}

async function runLmAssistant(payload: Record<string, unknown>) {
  if (!params.model) {
    toast.warning($tt('studio.pleaseSelectModel'));
    return;
  }
  if (lmSubmitting.value) return;
  lmSubmitting.value = true;
  lmProgress.value = 0;
  lmError.value = '';
  lmResult.value = null;
  try {
    const response = await api.audios.createLm({
      ...payload,
      model: params.model,
      seed: seedForSubmit(payload.seed),
    });
    const tid = taskIdFromSubmitResponse(response);
    if (!tid) throw new Error($tt('studio.missingTaskId'));
    tasksStore.registerPageOwnedStream(tid);
    api.gen.streamMediaTask(tid, {
      onLog: (log: unknown) => tasksStore.ingestTaskLog(tid, log),
      onTrace: (trace: unknown) => tasksStore.ingestTaskPipelineTrace(tid, trace),
      onProgress: (raw: unknown) => {
        const row = raw as Record<string, unknown>;
        const step = Number(row.step || 0);
        const total = Number(row.total || 0);
        lmProgress.value = total > 0 ? Math.round(step / total * 100) : Number(row.progress || 0);
      },
      onDone: (raw: unknown) => {
        const row = raw as Record<string, unknown>;
        tasksStore.unregisterPageOwnedStream(tid);
        lmSubmitting.value = false;
        if (row.status !== 'completed') {
          lmError.value = String(row.error || row.error_message || $tt('audio.editFailed'));
          return;
        }
        const result = (row.result || {}) as Record<string, unknown>;
        lmResult.value = {
          caption: String(result.caption || ''),
          lyrics: String(result.lyrics || ''),
          bpm: result.bpm == null ? null : Number(result.bpm),
          duration: result.duration == null ? null : Number(result.duration),
          keyscale: String(result.keyscale || ''),
          timesignature: String(result.timesignature || ''),
          language: String(result.language || ''),
          audio_code_indices: Array.isArray(result.audio_code_indices)
            ? result.audio_code_indices.map(Number).filter(Number.isFinite)
            : [],
        };
      },
      onError: () => {
        tasksStore.unregisterPageOwnedStream(tid);
        lmSubmitting.value = false;
        lmError.value = $tt('studio.connectionLost');
      },
    });
  } catch (error: unknown) {
    lmSubmitting.value = false;
    lmError.value = error instanceof Error ? error.message : String(error);
  }
}

function applyLmResult(result: AudioLmResult, generate: boolean) {
  audioWorkTab.value = 'create';
  params.prompt = result.caption;
  params.lyrics = result.lyrics;
  params.instrumental = !result.lyrics.trim();
  params.bpm = result.bpm;
  params.duration = result.duration ?? Number(currentModelConfig.value?.parameters?.duration?.default ?? 30);
  params.key_scale = result.keyscale;
  params.time_signature = result.timesignature;
  params.vocal_language = result.language;
  params.audio_code_indices = result.audio_code_indices.length ? [...result.audio_code_indices] : null;
  params.lm_expansion = params.audio_code_indices ? 'off' : 'auto';
  if (generate) nextTick(() => onAudioComposerGenerate());
}

async function onGenerateLyrics() {
  const prompt = String(params.prompt || '').trim();
  if (!prompt) {
    toast.warning($tt('studio.enterPrompt'));
    return;
  }
  const lyrics = await generateLyrics(prompt, { quietSuccess: true });
  if (lyrics) {
    lyricsApply.offer(params.lyrics, lyrics, (text) => { params.lyrics = text; });
  }
}

async function onEnhanceMusicBrief(ctx?: { stylePositive?: string }) {
  const prompt = String(params.prompt || '').trim();
  if (!prompt) {
    toast.warning($tt('studio.enterPrompt'));
    return;
  }
  const enhanced = await doEnhance(
    prompt,
    ctx?.stylePositive,
    'audio_create',
    params.model,
    { quietSuccess: true },
  );
  if (enhanced) {
    briefApply.offer(params.prompt, enhanced, (text) => { params.prompt = text; });
  }
}

function onBriefApplyReplace() {
  briefApply.applyReplace((text) => { params.prompt = text; });
}

function onBriefApplyAppend() {
  briefApply.applyAppend(() => params.prompt, (text) => { params.prompt = text; });
}

function onLyricsApplyReplace() {
  lyricsApply.applyReplace((text) => { params.lyrics = text; });
}

function onLyricsApplyAppend() {
  lyricsApply.applyAppend(() => params.lyrics, (text) => { params.lyrics = text; });
}

function goToDownload() {
  router.push({ name: 'models' });
}

function loadPromptDraft() {
  try {
    const assistantDraft = consumePromptDraft(DQ_STORAGE.AUDIO_CREATE_PROMPT_DRAFT);
    if (assistantDraft) {
      params.prompt = applyPromptDraft(params.prompt, assistantDraft);
      return;
    }
    const saved = getItem(DQ_STORAGE.AUDIO_CREATE_PROMPT_DRAFT);
    if (saved) {
      params.prompt = saved;
      return;
    }
    const legacy = localStorage.getItem(LEGACY_AUDIO_PROMPT_DRAFT_KEY);
    if (legacy) {
      params.prompt = legacy;
      setItem(DQ_STORAGE.AUDIO_CREATE_PROMPT_DRAFT, legacy);
      localStorage.removeItem(LEGACY_AUDIO_PROMPT_DRAFT_KEY);
    }
  } catch (e) { /* ignore */ }
}

function loadAssistantLyricsDraft() {
  const lyricsDraft = consumePromptDraft(DQ_STORAGE.AUDIO_CREATE_LYRICS_DRAFT);
  if (lyricsDraft) {
    params.lyrics = applyPromptDraft(params.lyrics, lyricsDraft);
  }
}

function savePromptDraft() {
  try {
    setItem(DQ_STORAGE.AUDIO_CREATE_PROMPT_DRAFT, params.prompt || '');
  } catch (e) { /* ignore */ }
}

function formatTime(seconds: number) {
  const s = Math.floor(seconds || 0);
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return m + ':' + String(sec).padStart(2, '0');
}

function onPickCoverSource() {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'audio/*';
  input.onchange = (e: Event) => {
    const target = e.target as HTMLInputElement;
    if (target.files && target.files[0]) {
      onCoverFilePicked(target.files[0]);
    }
  };
  input.click();
}

function hasCoverSource(): boolean {
  return (
    coverSourceFile.value != null ||
    (typeof coverSourceAssetPath.value === 'string' &&
      coverSourceAssetPath.value.startsWith('asset:'))
  );
}

function probeCoverSourceDuration() {
  const src = coverSourceSrc.value;
  const ticket = ++coverSourceProbe;
  coverSourceDuration.value = 0;
  if (!src) return;
  const audio = new Audio();
  audio.preload = 'metadata';
  audio.onloadedmetadata = () => {
    if (ticket === coverSourceProbe && Number.isFinite(audio.duration)) {
      coverSourceDuration.value = audio.duration;
    }
    audio.src = '';
  };
  audio.onerror = () => { audio.src = ''; };
  audio.src = src;
}

const composerDuration = computed({
  get: () => audioWorkTab.value === 'create' ? Number(params.duration || 30) : Number(params.duration || 0),
  set: (value: number | null) => {
    params.duration = audioWorkTab.value === 'create' ? Number(value || 30) : Number(value || 0) || null;
  },
});

function onCoverFilePicked(file: File) {
  if (coverSourceSrc.value && coverSourceSrc.value.startsWith('blob:')) {
    URL.revokeObjectURL(coverSourceSrc.value);
  }
  coverSourceAssetPath.value = '';
  coverSourceFile.value = file;
  coverSourceName.value = file.name;
  coverSourceSrc.value = URL.createObjectURL(file);
  probeCoverSourceDuration();
}

function clearCoverSource() {
  if (coverSourceSrc.value && coverSourceSrc.value.startsWith('blob:')) {
    URL.revokeObjectURL(coverSourceSrc.value);
  }
  coverSourceFile.value = null;
  coverSourceAssetPath.value = '';
  coverSourceName.value = '';
  coverSourceSrc.value = '';
  coverSourceDuration.value = 0;
  coverSourceProbe += 1;
}

function onCanvasUseAsCoverSource(payload: { path: string; previewUrl: string; quiet?: boolean }) {
  if (!payload.path.startsWith('asset:')) {
    toast.warning($tt('audio.coverNeedSource'));
    return;
  }
  audioWorkTab.value = 'cover';
  clearCoverSource();
  coverSourceAssetPath.value = payload.path;
  coverSourceSrc.value = payload.previewUrl || '';
  const item = galleryItems.value.find((g) => g.path === payload.path);
  coverSourceName.value = item?.title || item?.name || $tt('audio.coverSource');
  coverSourceDuration.value = Number(item?.duration_seconds || 0);
  if (!coverSourceDuration.value) probeCoverSourceDuration();
  audioCanvas.clearOverlay('cover_source');
  if (!payload.quiet) toast.success($tt('canvas.coverSourceBound'));
}

function attachTaskStream(tid: string) {
  if (!api.gen?.streamMediaTask) {
    generating.value = false;
    return;
  }

  tasksStore.registerPageOwnedStream(tid);
  tasksStore.clearTaskLogs(tid);
  tasksStore.appendTaskLog(tid, $tt('studio.startingGen'), 'info');

  const assetFileUrl = (aid: string) =>
    api.gallery?.getImageUrl
      ? api.gallery.getImageUrl('asset:' + aid)
      : '/api/assets/' + aid + '/file';

  const applyResultMeta = (meta: Record<string, unknown> | undefined) => {
    const m = meta || {};
    const eff = String(m.lyrics_effective || '').trim();
    previewLyrics.value = eff;
    previewLyricsDownload.value = eff;
  };

  const addRecentFromAsset = (aid: string, meta?: Record<string, unknown>, promptLabel?: string) => {
    const url = assetFileUrl(aid);
    const eff = meta ? String(meta.lyrics_effective || '').trim() : '';
    recentAudio.unshift({
      id: aid,
      url,
      title: String(params.title || '').trim(),
      prompt: promptLabel || params.prompt,
      lyrics: eff,
      duration: 0,
      playing: false,
    });
  };

  const setPreviewFromAsset = (aid: string, meta?: Record<string, unknown>, promptLabel?: string) => {
    const url = assetFileUrl(aid);
    const changed = previewAudioSrc.value !== url;
    previewAudioSrc.value = url;
    previewPrompt.value = previewDisplayCaption(String(params.title || ''), promptLabel || params.prompt);
    applyResultMeta(meta);
    previewIsPlaying.value = false;
    if (changed) previewAudioKey.value += 1;
  };

  api.gen.streamMediaTask(tid, {
    onLog: (logData: unknown) => {
      tasksStore.ingestTaskLog(tid, logData);
    },
    onTrace: (traceData: unknown) => {
      tasksStore.ingestTaskPipelineTrace(tid, traceData);
    },
    onStatus: (statusData: unknown) => {
      const row = statusData as Record<string, unknown>;
      if (row.status) currentTask.status = row.status as string;
      if (row.progress != null) currentTask.progress = row.progress as number;
    },
    onProgress: (progressData: unknown) => {
      const row = progressData as Record<string, unknown>;
      tasksStore.ingestTaskProgressLog(tid, row);
      tasksStore.patchLiveTaskProgress(tid, {
        progress: row.progress as number | undefined,
        step: row.step as number | undefined,
        total: row.total as number | undefined,
        eta_seconds: row.eta_seconds as number | undefined,
        progressMessage: (row.message ?? row.phase) as string | undefined,
      });
      if (row.progress != null) currentTask.progress = row.progress as number;
      if (row.step != null) currentTask.step = row.step as number;
      if (row.total != null) currentTask.total = row.total as number;
    },
    onResult: (resultData: unknown) => {
      const row = resultData as Record<string, unknown>;
      const meta = row.metadata as Record<string, unknown> | undefined;
      applyResultMeta(meta);
      const ids = row.asset_ids as string[] | undefined;
      if (ids?.length) {
        pendingCanvasAssetIds.value = ids;
        ids.forEach((aid) => addRecentFromAsset(aid, meta));
        setPreviewFromAsset(ids[0], meta);
      }
    },
    onDone: async (doneData: unknown) => {
      const row = doneData as Record<string, unknown>;
      tasksStore.unregisterPageOwnedStream(tid);
      if (row.status === 'completed') {
        tasksStore.appendTaskLog(tid, $tt('studio.genComplete'), 'success');
        const res = row.result as Record<string, unknown> | undefined;
        const meta = res?.metadata as Record<string, unknown> | undefined;
        applyResultMeta(meta);
        const pid = res?.primary_asset_id;
        if (pid) setPreviewFromAsset(String(pid), meta);
        const ids = [...pendingCanvasAssetIds.value];
        const willAutoAdd = shouldAutoAddToCanvas() && ids.length > 0;
        pendingCanvasAssetIds.value = [];
        if (!willAutoAdd) {
          toast.success($tt('studio.genComplete'));
        }
        await loadGallery(true);
        if (willAutoAdd) {
          await activateCanvasViewForResults(viewMode, syncCompositorOverlaysOnCanvasEnter);
          addAssetPathsToCanvas(ids.map((id) => `asset:${id}`), { placement: 'staging' });
        }
      } else if (row.status === 'failed') {
        let errMsg = String(row.error || row.error_message || '');
        if (!errMsg) {
          try {
            const updated = await api.gen.getMediaTask(tid) as Record<string, unknown>;
            errMsg = String(updated.error || updated.error_message || '');
          } catch {
            /* keep empty */
          }
        }
        tasksStore.appendTaskLog(tid, $tt('studio.genFailed', { msg: errMsg }), 'error');
        toast.error($tt('studio.genFailed', { msg: errMsg }));
      }
      generating.value = false;
    },
    onError: () => {
      tasksStore.unregisterPageOwnedStream(tid);
      tasksStore.appendTaskLog(tid, $tt('studio.connectionLost'), 'warning');
      toast.error($tt('studio.connectionLost'));
      generating.value = false;
    },
  });
}

async function queueAudioCreateToServer() {
  const lines = splitComposerPromptLines(params.prompt);
  if (lines.length === 0) {
    toast.warning($tt('studio.enterPrompt'));
    return;
  }
  if (queueSubmitting.value) return;

  queueSubmitting.value = true;
  try {
    for (let i = 0; i < lines.length; i += 1) {
      params.prompt = lines[i];
      await startGeneration({ queueOnly: true });
    }
    params.prompt = '';
    toast.success(
      lines.length > 1
        ? $tt('create.batchSubmitted', { count: lines.length })
        : $tt('assistant.taskQueued'),
    );
    tasksStore.pollQueueOnce();
  } catch (e: unknown) {
    toast.error($tt('studio.error', { msg: e instanceof Error ? e.message : String(e) }));
  } finally {
    queueSubmitting.value = false;
  }
}

async function queueAudioEditToServer() {
  if (queueSubmitting.value) return;

  queueSubmitting.value = true;
  try {
    await startEditGeneration({ queueOnly: true });
    toast.success($tt('assistant.taskQueued'));
    tasksStore.pollQueueOnce();
  } catch (e: unknown) {
    toast.error($tt('studio.error', { msg: e instanceof Error ? e.message : String(e) }));
  } finally {
    queueSubmitting.value = false;
  }
}

function audioSubmitErrorMessage(e: unknown, kind: 'create' | 'edit'): string {
  const err = e as { response?: { data?: { detail?: string } }; message?: string };
  if (err?.response?.data?.detail) return String(err.response.data.detail);
  if (err?.message) return String(err.message);
  return kind === 'edit'
    ? $tt('studio.error', { msg: $tt('audio.editFailed') })
    : $tt('studio.genFailed', { msg: '' });
}

type AudioGenerationOptions = {
  queueOnly?: boolean;
};

async function startEditGeneration(options: AudioGenerationOptions = {}) {
  const queueOnly = options.queueOnly === true;
  if (!queueOnly && generating.value) return;
  if (!params.model) { toast.warning($tt('studio.pleaseSelectModel')); return; }
  if (!modelReady.value) { toast.warning($tt('studio.modelNotReadyShort')); return; }
  if (!supportsEditAction.value) {
    toast.warning($tt('audio.modeNotSupported'));
    return;
  }
  if (!hasCoverSource()) {
    toast.warning($tt('audio.coverNeedSource'));
    return;
  }
  if (lyricsMissing.value) {
    toast.warning($tt(audioLyricsRequiredHintKey(currentModelConfig.value)));
    return;
  }

  if (!queueOnly) {
    generating.value = true;
    currentTask.id = '';
    currentTask.progress = 0;
    currentTask.step = null;
    currentTask.total = null;
    currentTask.status = '';
    previewLyrics.value = '';
    previewLyricsDownload.value = '';
  }

  try {
    persistComposerSnapshot();
    let source_asset_id: string;
    const assetPath = coverSourceAssetPath.value;
    if (typeof assetPath === 'string' && assetPath.startsWith('asset:')) {
      source_asset_id = assetPath.slice('asset:'.length);
    } else if (coverSourceFile.value) {
      const up = await api.gen.uploadAsset(coverSourceFile.value);
      source_asset_id = (up as any)?.id;
      if (!source_asset_id) {
        const msg = $tt('studio.uploadFailedGeneric');
        if (queueOnly) throw new Error(msg);
        toast.error($tt('studio.error', { msg }));
        generating.value = false;
        return;
      }
    } else {
      toast.warning($tt('audio.coverNeedSource'));
      if (!queueOnly) generating.value = false;
      return;
    }

    const adapters: Array<{ id: string; weight: number }> = [];
    if (params.lora) {
      adapters.push({ id: String(params.lora), weight: Number(params.lora_scale) || 0.8 });
    }
    const mode = audioWorkTab.value;
    if (mode === 'create') return;
    const validation = validateAudioEditInput(mode, {
      ...params,
      sourceAssetId: source_asset_id,
      sourceDuration: coverSourceDuration.value,
    });
    if (validation) {
      const msg = $tt(`audio.invalid.${validation}`);
      if (queueOnly) throw new Error(msg);
      toast.warning(msg);
      generating.value = false;
      return;
    }
    const body = buildAudioEditPayload(mode, source_asset_id, {
      ...params,
      seed: seedForSubmit(params.seed),
      vocal_language: resolveVocalLanguageForSubmit(params.lyrics, params.vocal_language),
      adapters,
      metadata: buildCanvasMeta(),
    });

    const resp = await api.audios.createEdit(body);
    const tid = taskIdFromSubmitResponse(resp);
    if (!tid) {
      const msg = $tt('studio.missingTaskId');
      if (queueOnly) throw new Error(msg);
      toast.error($tt('studio.error', { msg }));
      generating.value = false;
      return;
    }
    if (queueOnly) return;
    currentTask.id = tid;
    attachTaskStream(tid);
  } catch (e: unknown) {
    if (queueOnly) throw e;
    toast.error(audioSubmitErrorMessage(e, 'edit'));
    generating.value = false;
  }
}

async function startGeneration(options: AudioGenerationOptions = {}) {
  const queueOnly = options.queueOnly === true;
  if (!queueOnly && generating.value) return;
  if (!params.model) { toast.warning($tt('studio.pleaseSelectModel')); return; }
  if (!modelReady.value) { toast.warning($tt('studio.modelNotReadyShort')); return; }
  if (!params.prompt.trim()) { toast.warning($tt('studio.enterPrompt')); return; }
  if (lyricsMissing.value) {
    toast.warning($tt(audioLyricsRequiredHintKey(currentModelConfig.value)));
    return;
  }

  const parsed = parseModelVersion(selectedModelVersion.value || '');
  const mk = parsed.modelKey || parsed.model || currentModelKey.value;
  const vk = parsed.version || '';
  const verCfg =
    mk && currentModelConfig.value?.versions
      ? currentModelConfig.value.versions[vk]
      : null;
  const sizeHuman = verCfg?.size ? String(verCfg.size) : '';
  const minMemRaw = currentModelConfig.value?.parameters?.min_unified_memory_gb;
  const minUnifiedMemoryGb =
    minMemRaw != null && Number(minMemRaw) > 0 ? Number(minMemRaw) : null;
  warnIfRiskyMemory({
    systemInfo: unref(systemInfo),
    versionSizeHuman: sizeHuman,
    minUnifiedMemoryGb,
    $tt,
  });

  if (!queueOnly) {
    generating.value = true;
    currentTask.id = '';
    currentTask.progress = 0;
    currentTask.step = null;
    currentTask.total = null;
    currentTask.status = '';
    previewLyrics.value = '';
    previewLyricsDownload.value = '';
  }

  try {
    persistComposerSnapshot();
    const adapters: Array<{ id: string; weight: number }> = [];
    if (params.lora) {
      adapters.push({ id: String(params.lora), weight: Number(params.lora_scale) || 0.8 });
    }
    const body = buildAudioGenerationPayload({
      ...params,
      seed: seedForSubmit(params.seed),
      vocal_language: resolveVocalLanguageForSubmit(params.lyrics, params.vocal_language),
      lm_expansion: params.audio_code_indices?.length ? 'off' : params.lm_expansion || 'auto',
      adapters,
      metadata: buildCanvasMeta(),
    });

    const resp = await api.audios.createGeneration(body);
    const tid = taskIdFromSubmitResponse(resp);
    if (!tid) {
      const msg = $tt('studio.missingTaskId');
      if (queueOnly) throw new Error(msg);
      toast.error($tt('studio.error', { msg }));
      generating.value = false;
      return;
    }
    if (queueOnly) return;
    currentTask.id = tid;
    attachTaskStream(tid);
  } catch (e: unknown) {
    if (queueOnly) throw e;
    toast.error(audioSubmitErrorMessage(e, 'create'));
    generating.value = false;
  }
}

function recentAudioLabel(item: { title?: string; prompt?: string; name?: string }) {
  return assetDisplayLabel(item, 'Audio');
}

function isRecentActive(ra: any) {
  return !!ra?.url && ra.url === previewAudioSrc.value;
}

function selectRecent(item: any) {
  if (!item?.url) return;
  const same = isRecentActive(item);
  previewAudio(item);
}

function previewAudio(item: any) {
  const nextUrl = String(item?.url || '');
  const changed = previewAudioSrc.value !== nextUrl;
  previewAudioSrc.value = nextUrl;
  previewPrompt.value = recentAudioLabel(item);
  previewLyrics.value = item.lyrics || '';
  previewLyricsDownload.value = item.lyrics || '';
  previewDurationSec.value = item.duration || 0;
  previewIsPlaying.value = false;
  if (changed) previewAudioKey.value += 1;
}

function toggleRecentPlay(ra: any) {
  if (!ra?.url) return;
  if (!isRecentActive(ra)) {
    previewAudio(ra);
    return;
  }
  previewPlayerRef.value?.togglePlay?.();
}

function downloadPreviewAudio() {
  downloadAudioUrl(previewAudioSrc.value);
}

function downloadAudioUrl(url: string) {
  if (!url) return;
  const a = document.createElement('a');
  a.href = url;
  a.download = 'audio.' + (params.audio_format || 'wav');
  a.click();
}

function downloadLyricsText() {
  const text = previewLyricsDownload.value || previewLyrics.value;
  if (!text) return;
  const blob = new Blob([text + '\n'], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'lyrics.txt';
  a.click();
  URL.revokeObjectURL(url);
}

async function loadRecentAudio() {
  try {
    const data = await api.gen.listAssets('audio', 20, 0);
    const items = (data && data.items) || [];
    recentAudio.length = 0;
    items.forEach((item) => {
      const aid = String(item.id || '');
      if (!aid) return;
      const url = api.gallery.getImageUrl(`asset:${aid}`);
      const meta = (item.metadata as Record<string, unknown>) || {};
      recentAudio.push({
        id: aid,
        url,
        title: String(meta.title || ''),
        prompt: String(meta.prompt || ''),
        lyrics: String(meta.lyrics_effective || '').trim(),
        name: String(item.name || ''),
        duration:
          item.duration_seconds != null
            ? Number(item.duration_seconds)
            : meta.duration_seconds != null
              ? Number(meta.duration_seconds)
              : 0,
      });
    });
  } catch (e) {
    console.error('loadRecentAudio', e);
    toast.error($tt('studio.error', { msg: $tt('gallery.loadFailed') }));
  }
}

// ---- Lifecycle ----
onMounted(async () => {
  await loadModelRegistry();
  if (!params.model) {
    const versions = filteredModelPickerVersions.value;
    if (versions.length > 0) {
      const first = versions[0];
      selectedModelVersion.value = `${first.modelKey}|${first.versionKey}`;
      onModelChange(selectedModelVersion.value);
    }
  }
  loadPromptDraft();
  loadAssistantLyricsDraft();
  loadRecentAudio();
  loadGallery();
  loadPresets();
  try {
    await audioCanvas.ensureDefaultAndAdoptOrphans();
  } catch (e) {
    console.error('audio canvas adopt failed', e);
  }
  try {
    await audioCanvas.refreshSessions();
    await rebuildSessionAssetIndex();
  } catch (e) {
    console.error('audio canvas strip index failed', e);
  }
  window.addEventListener('keydown', onCreatePageKeydown);
});

onUnmounted(() => {
  window.removeEventListener('keydown', onCreatePageKeydown);
});

function onCreatePageKeydown(e: KeyboardEvent) {
  const target = e.target as HTMLElement | null;
  if (
    target &&
    (target.tagName === 'INPUT' ||
      target.tagName === 'TEXTAREA' ||
      target.isContentEditable)
  ) {
    return;
  }
  if (e.key === 'Escape' && composerDrawerOpen.value) {
    e.preventDefault();
    closeComposerDrawer();
    return;
  }
  if (e.key.toLowerCase() === 'c' && !e.metaKey && !e.ctrlKey && !e.altKey) {
    e.preventDefault();
    openGenerationComposer();
  }
}

let _draftTimer: ReturnType<typeof setTimeout> | null = null;
watch(() => params.prompt, () => {
  if (_draftTimer) clearTimeout(_draftTimer);
  _draftTimer = setTimeout(savePromptDraft, 500);
});

// Refresh model picker when tab changes
watch(audioWorkTab, (newMode, oldMode) => {
  if (newMode !== oldMode) {
    params.duration = newMode === 'create'
      ? Number(currentModelConfig.value?.parameters?.duration?.default ?? 30)
      : null;
    params.tracks = [];
    params.audio_code_indices = null;
    if (newMode === 'repaint' || newMode === 'lego') {
      params.region_start_sec = 0;
      params.region_end_sec = coverSourceDuration.value || null;
    }
  }
  // 保存旧模式模型
  if (oldMode && selectedModelVersion.value) {
    setItem(getAudioModeStorageKey(oldMode), selectedModelVersion.value);
  }
  // 恢复新模式模型
  const saved = getItem(getAudioModeStorageKey(newMode));
  if (saved) {
    const parsed = parseModelVersion(saved);
    const mk = parsed.modelKey || parsed.model || '';
    const vk = parsed.version || '';
    const key = vk ? `${mk}|${vk}` : mk;
    const stillValid = filteredModelPickerVersions.value.some(
      (r) => `${r.modelKey}|${r.versionKey}` === key && r.ready
    );
    if (stillValid) {
      selectedModelVersion.value = key;
      onModelChange(key);
      return;
    }
  }
  // fallback: 检查当前模型是否仍可用
  if (selectedModelVersion.value) {
    const currentKey = selectedModelVersion.value;
    const stillValid = filteredModelPickerVersions.value.some(
      (r) => `${r.modelKey}|${r.versionKey}` === currentKey && r.ready
    );
    if (stillValid) {
      onModelChange(currentKey);
      return;
    }
  }
  // fallback: 选择第一个可用模型
  const pick = filteredModelPickerVersions.value.find((r) => r.ready) || filteredModelPickerVersions.value[0];
  if (pick) {
    selectedModelVersion.value = `${pick.modelKey}|${pick.versionKey}`;
    onModelChange(selectedModelVersion.value);
  }
});

watch(modelFilterCommercialOnly, () => {
  const rows = filteredModelPickerVersions.value;
  const key = selectedModelVersion.value;
  if (key && rows.some((r) => `${r.modelKey}|${r.versionKey}` === key)) {
    return;
  }
  const pick = rows.find((r) => r.ready) || rows[0];
  if (!pick) {
    return;
  }
  selectedModelVersion.value = `${pick.modelKey}|${pick.versionKey}`;
  onModelChange(selectedModelVersion.value);
});

const infiniteCanvasRef = ref<InstanceType<typeof InfiniteCanvas> | null>(null);
const pendingCanvasAssetIds = ref<string[]>([]);
const audioCanvas = useCanvasStore('audio');
const canvasSelectedItem = ref<GalleryItem | null>(null);
const showLineageDrawer = ref(false);
const {
  composerDrawerOpen,
  openComposerDrawer,
  closeComposerDrawer,
} = useComposerDrawer('audio');

const savedViewMode = getItem(DQ_STORAGE.AUDIO_VIEW_MODE);
const viewMode = ref<'grid' | 'canvas'>(
  savedViewMode === 'canvas' || savedViewMode === 'grid' ? savedViewMode : 'grid'
);

const audioPreviewVisible = ref(false);
const selectedAudioIndex = ref(0);

watch(viewMode, (mode) => {
  setItem(DQ_STORAGE.AUDIO_VIEW_MODE, mode);
  if (mode === 'grid') closeComposerDrawer();
});

const lineageTargetAssetId = ref('');

const canvasAssetIdsOnCanvas = computed(() =>
  Object.keys(audioCanvas.items)
    .filter((p) => p.startsWith('asset:'))
    .map((p) => p.slice('asset:'.length))
);

async function onLineageFocusAsset(assetId: string) {
  if (viewMode.value !== 'canvas') {
    await activateCanvasViewForResults(viewMode, syncCompositorOverlaysOnCanvasEnter);
  }
  infiniteCanvasRef.value?.focusLineageAsset(assetId);
}

function shouldAutoAddToCanvas(): boolean {
  return viewMode.value === 'canvas' || canvasAutoAddEnabled('audio');
}

function addAssetPathsToCanvas(
  paths: string[],
  opts?: {
    fit?: boolean;
    placement?: 'staging' | 'center';
    focusLast?: boolean;
  }
) {
  if (!paths.length) return;
  if (viewMode.value === 'canvas' && infiniteCanvasRef.value) {
    infiniteCanvasRef.value.addPathsToCanvas(paths, {
      fit: opts?.fit,
      placement: opts?.placement ?? 'staging',
      focusLast: opts?.focusLast ?? true,
    });
    return;
  }
  audioCanvas.addPathsFromGallery(paths, galleryItems.value, {
    placement: opts?.placement ?? 'center',
  });
}

function onCanvasSessionReady(_payload: { sessionId: string }) {
  if (viewMode.value === 'canvas') {
    nextTick(() => {
      syncCompositorOverlaysOnCanvasEnter();
      if (Object.keys(audioCanvas.items).length > 0) {
        infiniteCanvasRef.value?.fitAll();
      }
    });
  }
}

function syncCompositorOverlaysOnCanvasEnter() {
  // Cover source lives in the composer only (no canvas COVER overlay — unlike image refs).
  audioCanvas.clearOverlay('cover_source');
}

function onCanvasOverlayCleared(kind: import('@/types').CanvasOverlayKind) {
  if (kind === 'cover_source') {
    clearCoverSource();
  }
}

function fillComposerFromGalleryItem(item: GalleryItem) {
  if (item.model) {
    selectedModelVersion.value = item.model.replace(':', '|');
    onModelChange(selectedModelVersion.value);
  }
  const meta = item.metadata || {};
  params.prompt = item.prompt || String(meta.prompt || '');
  params.title = item.title || String(meta.title || '');
  params.lyrics = String(meta.lyrics_input || meta.lyrics_effective || '');
  params.instrumental = String(meta.lyrics_effective || '').trim().toLowerCase() === '[inst]'
    || String(meta.lyrics_effective || '').trim().toLowerCase() === '[instrumental]';
  params.seed = meta.seed == null ? null : Number(meta.seed);
  params.duration = item.duration_seconds ?? (meta.duration as number | null) ?? null;
  params.steps = meta.steps == null ? params.steps : Number(meta.steps);
  params.guidance = meta.guidance == null ? params.guidance : Number(meta.guidance);
  params.bpm = meta.bpm == null ? null : Number(meta.bpm);
  params.key_scale = String(meta.keyscale || meta.key_scale || '');
  params.time_signature = String(meta.timesignature || meta.time_signature || '');
  params.vocal_language = String(meta.language || meta.vocal_language || '');
  params.vocal_type = String(meta.vocal_type || '');
  params.audio_format = String(meta.audio_format || meta.format || 'wav');
  params.denoise = String(meta.denoise || 'off');
  params.dcw_enabled = meta.dcw_enabled == null ? 'auto' : meta.dcw_enabled ? 'on' : 'off';
  params.dcw_mode = String(meta.dcw_mode || 'double');
  params.dcw_scaler = Number(meta.dcw_scaler ?? 0.05);
  params.dcw_high_scaler = Number(meta.dcw_high_scaler ?? 0.02);
  params.source_fidelity = Number(meta.audio_cover_strength ?? meta.source_fidelity ?? 1);
  params.cover_noise_strength = Number(meta.cover_noise_strength ?? 0);
  params.timbre_from_source = meta.timbre_from_source === true;
  params.cover_variant = String(meta.operation || '') === 'cover-nofsq' ? 'cover-nofsq' : 'cover';
  params.region_start_sec = meta.region_start_sec == null ? null : Number(meta.region_start_sec);
  params.region_end_sec = meta.region_end_sec == null ? null : Number(meta.region_end_sec);
  params.repaint_injection_ratio = Number(meta.repaint_injection_ratio ?? 1);
  params.repaint_crossfade_frames = Number(meta.repaint_crossfade_frames ?? 8);
  params.tracks = parseAudioTracks(meta.track);
  const adapters = Array.isArray(meta.adapters) ? meta.adapters : [];
  const first = adapters[0] as Record<string, unknown> | undefined;
  params.lora = String(first?.id || '');
  params.lora_scale = Number(first?.weight ?? 0.8);
  persistComposerSnapshot();
}

const composerDrawerTitle = computed(() => $tt(`action.audio.${audioWorkTab.value}`));

function openGenerationComposer() {
  if (audioWorkTab.value === 'cover') {
    audioWorkTab.value = 'create';
  }
  openComposerDrawer();
}

function onCanvasOpenComposer(payload?: { intent?: 'generate' | 'edit' | AudioWorkMode }) {
  if (canvasSelectedItem.value) fillComposerFromGalleryItem(canvasSelectedItem.value);
  if (payload?.intent && isAudioWorkMode(payload.intent) && payload.intent !== 'create') {
    audioWorkTab.value = payload.intent;
    openComposerDrawer();
    return;
  }
  if (payload?.intent === 'edit') {
    audioWorkTab.value = 'cover';
    openComposerDrawer();
    return;
  }
  openGenerationComposer();
}

function onCanvasNodeSelected(item: GalleryItem | null) {
  canvasSelectedItem.value = item;
  if (!item) persistComposerSnapshot();
}

watch(composerDrawerOpen, (open) => {
  if (open) {
    showLineageDrawer.value = false;
    infiniteCanvasRef.value?.closeMutexPanels?.();
  }
});

watch(showLineageDrawer, (open) => {
  if (open) closeComposerDrawer();
});

function persistComposerSnapshot() {
  audioCanvas.setComposerSnapshot({
    prompt: String(params.prompt || ''),
    title: String(params.title || ''),
    model: String(params.model || ''),
    negative_prompt: String(params.negative_prompt || ''),
    seed: params.seed != null ? String(params.seed) : undefined,
    denoise: params.denoise || 'off',
    lm_expansion: params.lm_expansion || 'auto',
    mode: audioWorkTab.value,
    cover_source_path: coverSourceAssetPath.value || undefined,
    audio_params: JSON.stringify(params),
  });
}

function onCanvasComposerRestore(snapshot: {
  prompt?: string;
  title?: string;
  model?: string;
  negative_prompt?: string;
  seed?: string;
  denoise?: string;
  lm_expansion?: string;
  mode?: string;
  cover_source_path?: string;
  audio_params?: string;
}) {
  if (!snapshot || typeof snapshot !== 'object') return;
  if (snapshot.model) {
    selectedModelVersion.value = snapshot.model.replace(':', '|');
    onModelChange(selectedModelVersion.value);
  }
  if (snapshot.prompt != null) params.prompt = snapshot.prompt;
  if (snapshot.title != null) params.title = snapshot.title;
  if (snapshot.negative_prompt != null) params.negative_prompt = snapshot.negative_prompt;
  if (['off', 'mild', 'strong'].includes(snapshot.denoise || '')) {
    params.denoise = snapshot.denoise as string;
  }
  if (['auto', 'format', 'off'].includes(snapshot.lm_expansion || '')) {
    params.lm_expansion = snapshot.lm_expansion as string;
  }
  if (snapshot.seed != null && snapshot.seed !== '') {
    const n = parseInt(snapshot.seed, 10);
    params.seed = Number.isNaN(n) ? null : n;
  }
  if (snapshot.mode === 'create' || snapshot.mode === 'cover') {
    audioWorkTab.value = snapshot.mode;
  }
  if (snapshot.audio_params) {
    try {
      const restored = JSON.parse(snapshot.audio_params);
      if (restored && typeof restored === 'object') Object.assign(params, restored);
    } catch {
      // Invalid legacy snapshots keep the scalar fields restored above.
    }
  }
  if (snapshot.cover_source_path) {
    const item = galleryItems.value.find((g) => g.path === snapshot.cover_source_path);
    coverSourceAssetPath.value = snapshot.cover_source_path;
    coverSourceSrc.value = item
      ? previewUrlForGalleryItem(item)
      : `/api/assets/${snapshot.cover_source_path.slice('asset:'.length)}/file`;
    coverSourceName.value = item?.title || item?.name || $tt('audio.coverSource');
    coverSourceDuration.value = Number(item?.duration_seconds || 0);
    if (!coverSourceDuration.value) probeCoverSourceDuration();
    coverSourceFile.value = null;
  }
  if (viewMode.value === 'canvas') {
    nextTick(() => syncCompositorOverlaysOnCanvasEnter());
  }
}

watch(coverSourceAssetPath, () => {
  persistComposerSnapshot();
  if (viewMode.value === 'canvas') {
    audioCanvas.clearOverlay('cover_source');
  }
});

function buildCanvasMeta(extra: Record<string, unknown> = {}): Record<string, unknown> {
  const meta: Record<string, unknown> = { ...extra };
  const sid = audioCanvas.sessionId.value;
  if (sid) meta.canvas_session_id = sid;
  // Pure create has no source track — never inherit canvas selection as parent.
  const mode = audioWorkTab.value;
  if (mode === 'create') return meta;
  const parentPath = coverSourceAssetPath.value || '';
  if (parentPath.startsWith('asset:')) {
    meta.parent_asset_id = parentPath.slice('asset:'.length);
    meta.relation_type = mode;
  }
  return meta;
}

function onCanvasOpenPreview(item: GalleryItem) {
  markPreviewSource(audioCanvas.sessionId.value);
  const list = previewItems.value;
  const idx = list.findIndex((g) => g.path === item.path);
  if (idx >= 0) {
    selectedAudioIndex.value = idx;
    audioPreviewVisible.value = true;
    return;
  }
  const gIdx = galleryItems.value.findIndex((g) => g.path === item.path);
  if (gIdx >= 0) {
    selectedAudioIndex.value = gIdx;
    audioPreviewVisible.value = true;
  }
}

function downloadAudioItem(item: GalleryItem) {
  const a = document.createElement('a');
  a.href = api.gallery.getImageUrl(item.path);
  a.download = item.name || 'audio.wav';
  a.click();
  toast.success($tt('gallery.startDownload'));
}

// ---- Gallery / Studio Canvas ----
const {
  galleryItems,
  galleryLoading,
  galleryHasMore,
  filterTime,
  filterModels,
  selectionMode,
  selectedPaths,
  allLoadedSelected,
  timeOptions,
  allModelOptions,
  hasActiveFilters,
  loadGallery,
  refreshGallery,
  onCanvasScroll,
  resetGalleryFilters,
  toggleSelect,
  toggleSelectionMode,
  selectAllLoaded,
  deleteItem,
  batchDeleteSelected,
  clearSelection,
} = useStudioGallery('audio');

const {
  stripSearchText,
  stripRenameOpen,
  stripRenameDraft,
  sessionAssetIdsBySession,
  canvasSessions,
  canvasSessionTitle,
  canvasReady,
  galleryByAssetId,
  enrichedGalleryItems,
  previewItems,
  previewTransferTargets,
  rebuildSessionAssetIndex,
  onStripOpen,
  onStripPreview,
  onStripCreate,
  onBackToStripList,
  onNewCanvas,
  refreshStripList,
  onStripRename,
  commitStripRename,
  onStripDelete,
  markPreviewSource,
  clearPreviewOverride,
  onPreviewTransfer,
} = useCanvasStripIa({
  media: 'audio',
  store: audioCanvas,
  galleryItems,
  viewMode,
  previewVisible: audioPreviewVisible,
  selectedIndex: selectedAudioIndex,
  onComposerRestore: (snap) => onCanvasComposerRestore(snap as any),
  openCreateComposer: () => openGenerationComposer(),
  reloadGallery: () => loadGallery(true),
});

watch(audioPreviewVisible, (open) => {
  if (!open) clearPreviewOverride();
});

function onViewModeChange(mode: 'grid' | 'canvas') {
  if (viewMode.value === mode) return;
  viewMode.value = mode;
  if (mode === 'canvas') {
    clearSelection();
    maybeShowCanvasWorkspaceHint();
    nextTick(() => {
      syncCompositorOverlaysOnCanvasEnter();
      persistComposerSnapshot();
      if (Object.keys(audioCanvas.items).length > 0) {
        infiniteCanvasRef.value?.fitAll();
      }
    });
  } else {
    clearSelection();
    closeComposerDrawer();
    void audioCanvas.refreshSessions().then(() => rebuildSessionAssetIndex());
  }
}

async function addGalleryItemToCanvas(
  item: GalleryItem,
  opts?: { switchView?: boolean; fit?: boolean }
) {
  if (opts?.switchView !== false && viewMode.value !== 'canvas') {
    onViewModeChange('canvas');
    await nextTick();
    await nextTick();
  }
  addAssetPathsToCanvas([item.path], {
    placement: 'center',
    focusLast: true,
    fit: opts?.fit,
  });
  toast.success($tt('canvas.addedToCanvas'));
}

const activeAudioTasks = computed(() => {
  const running = tasksStore.queueState.running.filter((t: Task) =>
    String(t.kind || '').startsWith('audio.')
  );
  const queued = tasksStore.queueState.queued.filter((t: Task) =>
    String(t.kind || '').startsWith('audio.')
  );
  return [...running, ...queued].map((t: Task) => {
    const live = tasksStore.liveTaskProgress[t.id];
    return live ? { ...t, ...live } : t;
  });
});

const composerBusy = computed(() => {
  if (generating.value) return true;
  return activeAudioTasks.value.some((t) =>
    ACTIVE_COMPOSER_TASK_STATUSES.has(String(t.status || '')),
  );
});

function onCardAction({ action, item }: { action: string; item: any }) {
  if (action === 'compose-from-item') {
    fillComposerFromGalleryItem(item);
    openGenerationComposer();
  } else if (action === 'add-to-canvas') {
    void addGalleryItemToCanvas(item, { fit: true });
  } else if (action === 'delete') {
    deleteItem(item);
  } else if (action === 'download') {
    downloadAudioItem(item);
  } else if (action === 'lineage') {
    lineageTargetAssetId.value = item.path?.startsWith('asset:')
      ? item.path.slice('asset:'.length)
      : '';
    showLineageDrawer.value = true;
  }
}

const presets = ref<Record<string, any>>({});

const filteredPresets = computed(() => {
  const want = audioWorkTab.value === 'cover' ? new Set(['cover']) : new Set(['create']);
  function planPresetShapeOk(preset: any) {
    return (
      Array.isArray(preset.applies_to) &&
      preset.applies_to.length > 0 &&
      preset.media_scope === 'audio'
    );
  }
  function matches(preset: any) {
    if (!planPresetShapeOk(preset)) return false;
    return preset.applies_to.some((k: string) => want.has(k));
  }
  const entries = Object.entries(presets.value)
    .filter(([, preset]) => matches(preset))
    .sort((a, b) => a[0].localeCompare(b[0], 'zh'));
  const result: Record<string, any> = {};
  for (const [name, preset] of entries) {
    result[name] = preset;
  }
  return result;
});

const durationOptions = computed(() => {
  const cfg = currentModelConfig.value;
  const dmin = cfg?.parameters?.duration?.min ?? 10;
  const dmax = cfg?.parameters?.duration?.max ?? 600;
  const ddef = cfg?.parameters?.duration?.default ?? 30;
  const opts: { label: string; value: number }[] = [];
  [10, 15, 30, 45, 60, 90, 120, 180, 300].forEach((sec) => {
    if (sec >= dmin && sec <= dmax) {
      opts.push({ label: `${sec}s`, value: sec });
    }
  });
  if (!opts.find((o) => o.value === ddef)) {
    opts.push({ label: `${ddef}s`, value: ddef });
  }
  opts.sort((a, b) => a.value - b.value);
  if (audioWorkTab.value !== 'create') {
    opts.unshift({ label: $tt('audio.durationFollowSource'), value: 0 });
  }
  return opts;
});

const generateLabel = computed(() => $tt(`audio.generateMode.${audioWorkTab.value}`));

async function loadPresets() {
  try {
    const data = await api.settings.getPresets();
    presets.value = (data as any) || {};
  } catch (e) {
    console.error('Failed to load presets:', e);
    presets.value = {};
  }
}
</script>

<style scoped>
.create-filters {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  min-width: 0;
}

.create-filters--list {
  width: 100%;
  max-width: none;
}

.create-filters__search {
  flex: 0 1 320px;
  min-width: 160px;
  max-width: 420px;
}

.create-filters__spacer {
  flex: 1 1 auto;
  min-width: 8px;
}

.create-filters__actions {
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  gap: 8px;
  margin-left: auto;
}

.create-filters__session-title {
  font-size: var(--dq-font-size-body);
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 40vw;
}
</style>
