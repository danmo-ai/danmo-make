<template>
  <StudioLayout
    class="studio-create-page"
    :freeform="viewMode === 'canvas'"
    hide-composer-bar
    @scroll="onCanvasScroll"
  >
    <template #filters>
      <StudioGalleryFilters
        :filter-time="filterTime"
        :filter-models="filterModels"
        :search-text="searchText"
        :time-options="timeOptions"
        :model-options="allModelOptions"
        :selection-mode="selectionMode"
        :selected-count="selectedPaths.size"
        :all-selected="allLoadedSelected"
        :view-mode="viewMode"
        :supports-canvas="true"
        canvas-media="audio"
        :composer-busy="composerBusy"
        @update:filter-time="filterTime = $event"
        @update:filter-models="filterModels = $event"
        @update:search-text="searchText = $event"
        @refresh="refreshGallery"
        @toggle-selection-mode="toggleSelectionMode"
        @select-all="selectAllLoaded"
        @batch-delete="batchDeleteSelected"
        @clear-selection="clearSelection"
        @update:view-mode="onViewModeChange"
        @composer-restore="onCanvasComposerRestore"
        @open-composer="openComposerDrawer()"
      />
    </template>
    <template #canvas>
      <StudioCanvas
        v-if="viewMode === 'grid'"
        :items="galleryItems"
        :active-tasks="activeAudioTasks"
        :loading="galleryLoading"
        :has-more="galleryHasMore"
        media="audio"
        :has-active-filters="hasActiveFilters"
        :selection-mode="selectionMode"
        :selected-paths="selectedPaths"
        :all-selected="allLoadedSelected"
        @select="onGallerySelect"
        @card-action="onCardAction"
        @reset-filters="resetGalleryFilters"
        @load-more="loadGallery(false)"
        @toggle-select="toggleSelect"
        @select-all="selectAllLoaded"
        @batch-delete="batchDeleteSelected"
        @clear-selection="clearSelection"
        @open-composer="openComposerDrawer()"
      />
      <InfiniteCanvas
        v-else
        ref="infiniteCanvasRef"
        :items="galleryItems"
        media="audio"
        @card-action="onCardAction"
        @download="downloadAudioItem"
        @toggle-grid-view="() => onViewModeChange('grid')"
        @node-selected="onCanvasNodeSelected"
        @session-ready="onCanvasSessionReady"
        @open-preview="onCanvasOpenPreview"
        @composer-restore="onCanvasComposerRestore"
        @overlay-cleared="onCanvasOverlayCleared"
        @use-as-cover-source="onCanvasUseAsCoverSource"
        @open-composer="onCanvasOpenComposer()"
        @request-close-composer="closeComposerDrawer()"
      />
    </template>
  </StudioLayout>

  <StudioComposeFab
    v-if="!composerDrawerOpen"
    media="audio"
    :busy="composerBusy"
    @open="openComposerDrawer()"
  />

  <StudioComposerHost
    v-model:open="composerDrawerOpen"
    :drawer-title="$t('studio.composerDrawerTitle')"
  >
    <AudioComposer
        v-model="params.prompt"
        v-model:title="params.title"
        v-model:work-mode="audioWorkTab"
        v-model:model="selectedModelVersion"
        v-model:duration="params.duration"
        v-model:batch-count="params.n"
        :composer-busy="composerBusy"
        :submitting="queueSubmitting"
        :generating="generating"
        :can-generate="audioWorkTab === 'create' ? canGenerateCreate : canGenerateCover"
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
        :current-model-config="advancedParamModelConfig"
        :compatible-loras="compatibleLoras"
        @generate="onAudioComposerGenerate"
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

  <!-- Audio preview dialog -->
  <GalleryPreviewDialog
    v-model:open="audioPreviewVisible"
    v-model:index="selectedAudioIndex"
    :items="galleryItems"
    media="audio"
  />

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
import { warnIfRiskyMemory } from '@/composables/memoryHint';
import type { SystemInfo, Task, GalleryItem } from '@/types';
import { assetDisplayLabel, previewDisplayCaption } from '@/utils/assetDisplay';
import StudioLayout from '@/components/studio/StudioLayout.vue';
import StudioCanvas from '@/components/studio/StudioCanvas.vue';
import StudioGalleryFilters from '@/components/studio/StudioGalleryFilters.vue';
import AudioComposer from '@/components/studio/AudioComposer.vue';
import StudioComposerHost from '@/components/studio/StudioComposerHost.vue';
import StudioComposeFab from '@/components/studio/StudioComposeFab.vue';
import InfiniteCanvas from '@/components/studio/InfiniteCanvas.vue';
import StudioLineagePanel from '@/components/studio/StudioLineagePanel.vue';
import GalleryPreviewDialog from '@/components/gallery/GalleryPreviewDialog.vue';
import { useStudioGallery } from '@/composables/useStudioGallery';
import { canvasAutoAddEnabled, useCanvasStore } from '@/composables/useCanvasStore';
import { useComposerDrawer } from '@/composables/useComposerDrawer';
import {
  activateCanvasViewForResults,
  maybeShowCanvasWorkspaceHint,
} from '@/utils/canvasWorkspaceHint';
import { previewUrlForGalleryItem } from '@/utils/canvasAssets';
import { isAudioLyricsRequired, audioLyricsRequiredHintKey, resolveVocalLanguageForSubmit } from '@/utils/audioLyrics';
import { ACTIVE_COMPOSER_TASK_STATUSES, splitComposerPromptLines } from '@/utils/composerQueue';

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

/** Registry v2: `actions` is `{ create: {}, ... }`, not a string array. */
function supportsAction(actions: unknown, action: string): boolean {
  if (actions == null) return false;
  if (Array.isArray(actions)) {
    return actions.includes(action);
  }
  if (typeof actions === 'object') {
    const rec = actions as Record<string, unknown>;
    return Object.prototype.hasOwnProperty.call(rec, action) && rec[action] != null;
  }
  return false;
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

// ---- Tab navigation ----
const audioWorkTab = ref('create');
const audioWorkSegmentOptions = computed(() => [
  { label: $tt('action.audio.create'), value: 'create' },
  { label: $tt('action.audio.cover'), value: 'cover' },
]);

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
  duration: 30,
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
  lora: '',
  lora_scale: 0.8,
});

const coverParams = reactive({
  source_fidelity: 1.0,
  duration: null as number | null,
});

const coverSourceSrc = ref('');
const coverSourceFile = ref<File | null>(null);
const coverSourceAssetPath = ref('');
const coverSourceName = ref('');

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
  return params.steps !== stepsDef || params.guidance !== guidanceDef
    || params.seed !== null || params.n !== 1 || params.duration !== durationDef
    || params.audio_format !== formatDef
    || !!params.lora;
});

const compatibleLoras = ref<Record<string, unknown>[]>([]);

async function loadCompatibleAdapters(modelKey: string) {
  if (!modelKey) {
    compatibleLoras.value = [];
    return;
  }
  try {
    const loras = await api.settings.getCompatibleLoras(modelKey);
    compatibleLoras.value = (loras as Record<string, unknown>[]) || [];
    if (params.lora && !compatibleLoras.value.some((l) => String(l.id) === String(params.lora))) {
      params.lora = '';
    }
  } catch (e) {
    console.error('Failed to load compatible adapters:', e);
    compatibleLoras.value = [];
  }
}

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
    const act = audioWorkTab.value === 'cover' ? 'cover' : 'create';
    if (act === 'cover' && !supportsAction(config.actions, 'cover')) continue;
    if (act === 'create' && !supportsAction(config.actions, 'create')) continue;
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
        recommended: config.recommended === true && (versionConfig as Record<string, unknown>).default === true,
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
  rows.sort((a, b) => {
    const ar = a.recommended ? 1 : 0;
    const br = b.recommended ? 1 : 0;
    if (ar !== br) return br - ar;
    const an = String(a.name || '');
    const bn = String(b.name || '');
    try {
      return an.localeCompare(bn, 'zh');
    } catch {
      return an < bn ? -1 : an > bn ? 1 : 0;
    }
  });
  return rows;
});

const lyricsRequiredForModel = computed(() =>
  isAudioLyricsRequired(currentModelConfig.value, !!params.instrumental),
);

const lyricsMissing = computed(() =>
  lyricsRequiredForModel.value && !String(params.lyrics || '').trim(),
);

const canGenerateCreate = computed(() => {
  if (!params.model) return false;
  if (!modelReady.value) return false;
  if (!params.prompt.trim()) return false;
  if (lyricsMissing.value) return false;
  return true;
});

const canGenerateCover = computed(() => {
  if (!params.model) return false;
  if (!modelReady.value) return false;
  if (!supportsCoverAction.value) return false;
  if (!hasCoverSource()) return false;
  if (lyricsMissing.value) return false;
  return true;
});

const supportsCoverAction = computed(() => {
  return supportsAction(currentModelConfig.value?.actions, 'cover');
});

// ---- Composer adapters ----

function applyAudioComposerParams(val: Record<string, unknown>) {
  params.steps = val.steps as number;
  params.guidance = val.guidance as number;
  params.seed = val.seed as number | null;
  params.negative_prompt = val.negative_prompt as string;
  params.lyrics = val.lyrics as string;
  params.instrumental = val.instrumental as boolean;
  params.bpm = val.bpm as number | null;
  params.key_scale = val.key_scale as string;
  params.time_signature = val.time_signature as string;
  params.vocal_language = val.vocal_language as string;
  params.vocal_type = val.vocal_type as string;
  params.lora = String(val.lora || '');
  params.lora_scale = Number(val.lora_scale) || 0.8;
  coverParams.source_fidelity = (val.source_fidelity as number) ?? 1.0;
}

const audioModelSelectOptions = computed(() =>
  filteredModelPickerVersions.value.map((v) => ({
    label: String(v.name || ''),
    value: `${v.modelKey}|${v.versionKey}`,
    disabled: !v.ready,
    commercialUseAllowed: v.commercialUseAllowed === true,
  }))
);

const coverReferenceMedia = computed(() => {
  if (audioWorkTab.value !== 'cover') return null;
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
      void queueAudioCoverToServer();
    }
    return;
  }
  if (audioWorkTab.value === 'create') {
    startGeneration();
  } else {
    startCoverGeneration();
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
  params.negative_prompt = '';
  params.lyrics = '';
  params.bpm = null;
  params.key_scale = '';
  params.time_signature = '';
  params.vocal_language = '';
  params.vocal_type = '';
  params.lora = '';
  params.lora_scale = 0.8;
  if (p.supports_instrumental === false) {
    params.instrumental = false;
  }
}

function getAudioModeStorageKey(mode: string): StorageKey {
  const map: Record<string, StorageKey> = {
    create: DQ_STORAGE.AUDIO_MODEL_CREATE,
    cover: DQ_STORAGE.AUDIO_MODEL_COVER,
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

function onCoverFilePicked(file: File) {
  if (coverSourceSrc.value && coverSourceSrc.value.startsWith('blob:')) {
    URL.revokeObjectURL(coverSourceSrc.value);
  }
  coverSourceAssetPath.value = '';
  coverSourceFile.value = file;
  coverSourceName.value = file.name;
  coverSourceSrc.value = URL.createObjectURL(file);
}

function clearCoverSource() {
  if (coverSourceSrc.value && coverSourceSrc.value.startsWith('blob:')) {
    URL.revokeObjectURL(coverSourceSrc.value);
  }
  coverSourceFile.value = null;
  coverSourceAssetPath.value = '';
  coverSourceName.value = '';
  coverSourceSrc.value = '';
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
  syncCanvasCoverOverlay(payload.path);
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

async function queueAudioCoverToServer() {
  if (queueSubmitting.value) return;

  queueSubmitting.value = true;
  try {
    await startCoverGeneration({ queueOnly: true });
    toast.success($tt('assistant.taskQueued'));
    tasksStore.pollQueueOnce();
  } catch (e: unknown) {
    toast.error($tt('studio.error', { msg: e instanceof Error ? e.message : String(e) }));
  } finally {
    queueSubmitting.value = false;
  }
}

function audioSubmitErrorMessage(e: unknown, kind: 'create' | 'cover'): string {
  const err = e as { response?: { data?: { detail?: string } }; message?: string };
  if (err?.response?.data?.detail) return String(err.response.data.detail);
  if (err?.message) return String(err.message);
  return kind === 'cover'
    ? $tt('studio.error', { msg: $tt('audio.coverFailed') })
    : $tt('studio.genFailed', { msg: '' });
}

type AudioGenerationOptions = {
  queueOnly?: boolean;
};

async function startCoverGeneration(options: AudioGenerationOptions = {}) {
  const queueOnly = options.queueOnly === true;
  if (!queueOnly && generating.value) return;
  if (!params.model) { toast.warning($tt('studio.pleaseSelectModel')); return; }
  if (!modelReady.value) { toast.warning($tt('studio.modelNotReadyShort')); return; }
  if (!supportsCoverAction.value) {
    toast.warning($tt('audio.coverNotSupported'));
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

    const body = {
      model: params.model,
      operation: 'cover',
      source_asset_id,
      prompt: String(params.prompt || '').trim(),
      source_fidelity: coverParams.source_fidelity ?? 1.0,
      seed: params.seed ?? null,
      n: params.n ?? 1,
      audio_format: params.audio_format || 'wav',
      duration: coverParams.duration ?? null,
      lyrics: String(params.lyrics || '').trim(),
      vocal_language: String(params.vocal_language || '').trim(),
      vocal_type: String(params.vocal_type || '').trim(),
      bpm: params.bpm ?? null,
      key_scale: String(params.key_scale || '').trim(),
      time_signature: String(params.time_signature || '').trim(),
      metadata: buildCanvasMeta(),
    };

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
    toast.error(audioSubmitErrorMessage(e, 'cover'));
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
    const body = {
      model: params.model,
      title: String(params.title || '').trim(),
      prompt: params.prompt.trim(),
      negative_prompt: params.negative_prompt || '',
      duration: params.duration ?? null,
      instrumental: !!params.instrumental,
      lyrics: params.lyrics?.trim() || '',
      vocal_language: resolveVocalLanguageForSubmit(
        String(params.lyrics || '').trim(),
        String(params.vocal_language || '').trim(),
      ),
      vocal_type: params.vocal_type?.trim() || '',
      bpm: params.bpm ?? null,
      key_scale: params.key_scale || '',
      time_signature: params.time_signature || '',
      steps: params.steps ?? null,
      guidance: params.guidance ?? null,
      seed: params.seed ?? null,
      n: params.n ?? 1,
      audio_format: params.audio_format || 'wav',
      adapters,
      metadata: buildCanvasMeta(),
    };

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
    openComposerDrawer();
  }
}

let _draftTimer: ReturnType<typeof setTimeout> | null = null;
watch(() => params.prompt, () => {
  if (_draftTimer) clearTimeout(_draftTimer);
  _draftTimer = setTimeout(savePromptDraft, 500);
});

// Refresh model picker when tab changes
watch(audioWorkTab, (newMode, oldMode) => {
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

watch(viewMode, (mode) => {
  setItem(DQ_STORAGE.AUDIO_VIEW_MODE, mode);
});

function onViewModeChange(mode: 'grid' | 'canvas') {
  if (viewMode.value === mode) return;
  const batchPaths =
    mode === 'canvas' && selectedPaths.value.size > 0
      ? Array.from(selectedPaths.value)
      : [];
  viewMode.value = mode;
  if (mode === 'canvas') {
    clearSelection();
    maybeShowCanvasWorkspaceHint();
    const batchCount = batchPaths.length;
    nextTick(() => {
      syncCompositorOverlaysOnCanvasEnter();
      persistComposerSnapshot();
      if (batchCount > 0) {
        addAssetPathsToCanvas(batchPaths, {
          fit: true,
          placement: 'center',
          focusLast: false,
        });
        toast.success($tt('canvas.batchAdded', { n: batchCount }));
      } else if (Object.keys(audioCanvas.items).length > 0) {
        infiniteCanvasRef.value?.fitAll();
      }
    });
  } else {
    clearSelection();
  }
}

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

function syncCanvasCoverOverlay(path: string | null) {
  if (!path) {
    audioCanvas.clearOverlay('cover_source');
    return;
  }
  const item = galleryItems.value.find((g) => g.path === path) ?? undefined;
  audioCanvas.setOverlay('cover_source', path, item);
}

function restoreComposerFromCanvasOverlays() {
  const cov = audioCanvas.overlays.cover_source;
  if (!coverSourceAssetPath.value && cov?.path) {
    const item = galleryItems.value.find((g) => g.path === cov.path);
    coverSourceAssetPath.value = cov.path;
    coverSourceSrc.value = item ? previewUrlForGalleryItem(item) : '';
    coverSourceName.value = item?.title || item?.name || $tt('audio.coverSource');
    coverSourceFile.value = null;
    audioWorkTab.value = 'cover';
  }
}

function syncCompositorOverlaysOnCanvasEnter() {
  restoreComposerFromCanvasOverlays();
  syncCanvasCoverOverlay(coverSourceAssetPath.value || null);
}

function onCanvasOverlayCleared(kind: import('@/types').CanvasOverlayKind) {
  if (kind === 'cover_source') {
    clearCoverSource();
  }
}

function fillComposerFromGalleryItem(item: GalleryItem) {
  if (item.prompt) params.prompt = item.prompt;
  if (item.title) params.title = item.title;
  persistComposerSnapshot();
}

function onCanvasOpenComposer() {
  if (canvasSelectedItem.value) {
    fillComposerFromGalleryItem(canvasSelectedItem.value);
  }
  openComposerDrawer();
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
    mode: audioWorkTab.value,
    cover_source_path: coverSourceAssetPath.value || undefined,
  });
}

function onCanvasComposerRestore(snapshot: {
  prompt?: string;
  title?: string;
  model?: string;
  negative_prompt?: string;
  seed?: string;
  mode?: string;
  cover_source_path?: string;
}) {
  if (!snapshot || typeof snapshot !== 'object') return;
  if (snapshot.prompt != null) params.prompt = snapshot.prompt;
  if (snapshot.title != null) params.title = snapshot.title;
  if (snapshot.negative_prompt != null) params.negative_prompt = snapshot.negative_prompt;
  if (snapshot.seed != null && snapshot.seed !== '') {
    const n = parseInt(snapshot.seed, 10);
    params.seed = Number.isNaN(n) ? null : n;
  }
  if (snapshot.mode === 'create' || snapshot.mode === 'cover') {
    audioWorkTab.value = snapshot.mode;
  }
  if (snapshot.cover_source_path) {
    const item = galleryItems.value.find((g) => g.path === snapshot.cover_source_path);
    coverSourceAssetPath.value = snapshot.cover_source_path;
    coverSourceSrc.value = item
      ? previewUrlForGalleryItem(item)
      : `/api/assets/${snapshot.cover_source_path.slice('asset:'.length)}/file`;
    coverSourceName.value = item?.title || item?.name || $tt('audio.coverSource');
    coverSourceFile.value = null;
  }
  if (viewMode.value === 'canvas') {
    nextTick(() => syncCompositorOverlaysOnCanvasEnter());
  }
}

watch(coverSourceAssetPath, () => {
  persistComposerSnapshot();
  if (viewMode.value === 'canvas') {
    syncCanvasCoverOverlay(coverSourceAssetPath.value || null);
  }
});

function buildCanvasMeta(extra: Record<string, unknown> = {}): Record<string, unknown> {
  const meta: Record<string, unknown> = { ...extra };
  const sid = audioCanvas.sessionId.value;
  if (sid) meta.canvas_session_id = sid;
  const parentPath =
    canvasSelectedItem.value?.path || audioCanvas.activeAssetPath.value || '';
  if (parentPath.startsWith('asset:')) {
    meta.parent_asset_id = parentPath.slice('asset:'.length);
    meta.relation_type = audioWorkTab.value === 'cover' ? 'cover' : 'create';
  }
  return meta;
}

function onCanvasOpenPreview(item: GalleryItem) {
  const idx = galleryItems.value.findIndex((g) => g.path === item.path);
  if (idx >= 0) {
    selectedAudioIndex.value = idx;
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

// ---- Gallery preview dialog ----
const audioPreviewVisible = ref(false);
const selectedAudioIndex = ref(0);

function onGallerySelect(item: any) {
  const idx = galleryItems.value.findIndex((it) => it.path === item.path);
  selectedAudioIndex.value = idx >= 0 ? idx : 0;
  audioPreviewVisible.value = true;
}

function onCardAction({ action, item }: { action: string; item: any }) {
  if (action === 'compose-from-item') {
    fillComposerFromGalleryItem(item);
    openComposerDrawer();
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
  return opts;
});

const generateLabel = computed(() => {
  if (audioWorkTab.value === 'cover') return $tt('audio.coverGenerate');
  return $tt('audio.generate');
});

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
/* AudioCreateView styles moved to AudioComposer.vue */
</style>
