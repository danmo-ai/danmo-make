<template>
  <StudioLayout class="studio-create-page" hide-composer-bar>
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
        view-mode="grid"
        :supports-canvas="false"
        canvas-media="video"
        :composer-busy="composerBusy"
        :show-model-filter="false"
        :show-search="true"
        :search-placeholder="$tt('avatar.searchPlaceholder')"
        @update:filter-time="filterTime = $event"
        @update:filter-models="filterModels = $event"
        @update:search-text="searchText = $event"
        @refresh="refreshGallery"
        @toggle-selection-mode="toggleSelectionMode"
        @select-all="selectAllLoaded"
        @batch-delete="batchDeleteSelected"
        @clear-selection="clearSelection"
        @open-composer="openComposerDrawer()"
      />
    </template>
    <template #canvas>
      <StudioCanvas
        :items="galleryItems"
        :active-tasks="activeVideoTasks"
        :loading="galleryLoading"
        :has-more="galleryHasMore"
        media="video"
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
    </template>
  </StudioLayout>

  <StudioComposeFab
    v-if="!composerDrawerOpen"
    media="video"
    :busy="composerBusy"
    @open="openComposerDrawer()"
  />

  <StudioComposerHost v-model:open="composerDrawerOpen" :drawer-title="$t('avatar.composerTitle')">
    <AvatarComposer
      v-model="params.prompt"
      v-model:title="params.title"
      v-model:model="selectedModelVersion"
      v-model:resolution="selectedResolution"
      :params="params"
      :generating="generating"
      :can-generate="!submitDisabled"
      :generate-label="$tt('avatar.generate')"
      :model-options="avatarModelSelectOptions"
      :resolution-options="resolutionSelectOptions"
      :current-model-config="currentModelConfig"
      :show-negative-prompt="supportsNegativePrompt"
      :mode="mode"
      :script-text="scriptText"
      :portrait-preview="portraitPreviewUrl"
      :portrait-label="portraitLabel"
      :audio-label="audioLabel"
      @update:params="applyComposerParams"
      @update:mode="mode = $event"
      @update:script-text="scriptText = $event"
      @generate="onGenerate"
      @pick-portrait="openPortraitPicker"
      @remove-portrait="clearPortrait"
      @pick-audio="openAudioPicker"
      @remove-audio="clearAudio"
      @model-change="onModelChange"
    />
  </StudioComposerHost>

  <DqDialog v-model:open="portraitPickerOpen" :title="$tt('avatar.pickPortrait')" width="min(560px, 92vw)">
    <AssetPicker accept-kind="image" @pick="onPortraitPicked" />
  </DqDialog>

  <DqDialog v-model:open="audioPickerOpen" :title="$tt('avatar.pickAudio')" width="min(560px, 92vw)">
    <AssetPicker accept-kind="audio" @pick="onAudioPicked" />
  </DqDialog>

  <GalleryPreviewDialog
    v-model:open="videoPreviewVisible"
    v-model:index="selectedVideoIndex"
    :items="galleryItems"
    media="video"
  />
</template>

<script setup lang="ts">
// @ts-nocheck — avatar create view
import { ref, reactive, computed, watch, onMounted, onBeforeUnmount, inject, type Ref } from 'vue';
import { toast } from '@/utils/feedback';
import { api, taskIdFromSubmitResponse } from '@/utils/api';
import { $tt, $md, $mvn } from '@/utils/i18n';
import { useTasksStore } from '@/stores/tasks';
import { useRegistryStore } from '@/stores/registry';
import { useModelRegistryFilters } from '@/composables/useModelRegistryFilters';
import { applyModelVersionFilters } from '@/utils/modelPickerFilters';
import { modelSelectOption } from '@/utils/modelSelectOption';
import { warnIfRiskyMemory } from '@/composables/memoryHint';
import { useStudioGallery } from '@/composables/useStudioGallery';
import { useComposerDrawer } from '@/composables/useComposerDrawer';
import { ACTIVE_COMPOSER_TASK_STATUSES } from '@/utils/composerQueue';
import type { GalleryItem, SystemInfo, Task } from '@/types';
import StudioLayout from '@/components/studio/StudioLayout.vue';
import StudioCanvas from '@/components/studio/StudioCanvas.vue';
import StudioGalleryFilters from '@/components/studio/StudioGalleryFilters.vue';
import AvatarComposer from '@/components/studio/AvatarComposer.vue';
import StudioComposerHost from '@/components/studio/StudioComposerHost.vue';
import StudioComposeFab from '@/components/studio/StudioComposeFab.vue';
import AssetPicker from '@/components/asset/AssetPicker.vue';
import GalleryPreviewDialog from '@/components/gallery/GalleryPreviewDialog.vue';

const tasksStore = useTasksStore();
const registryStore = useRegistryStore();
const systemInfo = inject<Ref<SystemInfo>>('systemInfo');

function parseModelVersion(s: string) {
  const [m, v] = (s || '').split('|');
  return { model: m || '', version: v || '' };
}

function hasAvatarAction(actions: unknown) {
  return actions && typeof actions === 'object' && Object.prototype.hasOwnProperty.call(actions, 'avatar');
}

function isVideoModel(config: Record<string, unknown> | null | undefined) {
  return config && config.media === 'video';
}

const generating = ref(false);
const portraitPickerOpen = ref(false);
const audioPickerOpen = ref(false);
const portraitAssetId = ref('');
const portraitPreviewUrl = ref('');
const portraitLabel = ref('');
const portraitFile = ref<File | null>(null);
const audioAssetId = ref('');
const audioLabel = ref('');
const audioFile = ref<File | null>(null);
const selectedModelVersion = ref('');
const selectedResolution = ref('512x512');
const mode = ref<'lip_sync' | 'script'>('lip_sync');
const scriptText = ref('');

const params = reactive({
  title: '',
  prompt: '',
  model: '',
  version: '',
  width: 512,
  height: 512,
  num_frames: 93,
  fps: 25,
  steps: 8,
  seed: null as number | null,
  negative_prompt: '',
});

const {
  composerDrawerOpen,
  openComposerDrawer,
  closeComposerDrawer,
} = useComposerDrawer('avatar');

const {
  galleryItems,
  galleryLoading,
  galleryHasMore,
  filterTime,
  filterModels,
  searchText,
  timeOptions,
  allModelOptions,
  selectionMode,
  selectedPaths,
  allLoadedSelected,
  hasActiveFilters,
  refreshGallery,
  loadGallery,
  resetGalleryFilters,
  toggleSelectionMode,
  toggleSelect,
  selectAllLoaded,
  batchDeleteSelected,
  clearSelection,
  deleteItem,
} = useStudioGallery('video', {
  sourceAction: ['avatar', 'avatar_script'],
  searchFields: ['title', 'prompt'],
});

const videoPreviewVisible = ref(false);
const selectedVideoIndex = ref(0);

const modelRegistry = ref<Record<string, unknown>>({});
const modelsDetailedStatus = ref<Record<string, unknown>>({});

const allAvatarVersions = computed(() => {
  const result: Record<string, unknown>[] = [];
  for (const [modelKey, config] of Object.entries(modelRegistry.value)) {
    if (!isVideoModel(config as Record<string, unknown>) || !hasAvatarAction((config as any).actions)) {
      continue;
    }
    const versions = (config as any).versions || {};
    const detailed = (modelsDetailedStatus.value as any)[modelKey] || {};
    const versionStatuses = detailed.versions || {};
    for (const [versionKey, versionConfig] of Object.entries(versions)) {
      const status = versionStatuses[versionKey] || { status: 'not_downloaded', ready: false };
      result.push({
        modelKey,
        versionKey,
        name: $mvn(modelKey, config as any, versionConfig as any),
        description: $md(config as any),
        ready: status.ready,
        recommended: (config as any).recommended && (versionConfig as any).default,
        commercialUseAllowed: (config as any).commercial_use_allowed === true,
      });
    }
  }
  return result;
});

const { commercialOnly: modelFilterCommercialOnly } = useModelRegistryFilters();

const avatarModelPickerVersions = computed(() => {
  const rows = applyModelVersionFilters(allAvatarVersions.value as any[], {
    installedOnly: true,
    commercialOnly: modelFilterCommercialOnly.value,
  });
  rows.sort((a, b) => {
    const ar = a.recommended ? 1 : 0;
    const br = b.recommended ? 1 : 0;
    if (ar !== br) return br - ar;
    return String(a.name || '').localeCompare(String(b.name || ''), 'zh');
  });
  return rows;
});

const avatarModelSelectOptions = computed(() =>
  avatarModelPickerVersions.value.map((v) => modelSelectOption(v)),
);

const currentModelConfig = computed(() => (modelRegistry.value[params.model] as Record<string, unknown>) || null);

const resolutionSelectOptions = computed(() => {
  const presets = (currentModelConfig.value?.parameters as any)?.resolution_presets;
  const options = presets?.options || ['512x512', '832x480'];
  return options.map((s: string) => ({ label: s, value: s }));
});

const supportsNegativePrompt = computed(() => {
  const p = (currentModelConfig.value?.parameters as any) || {};
  return p.negative_prompt_support !== false;
});

const modelReady = computed(() => {
  const row = avatarModelPickerVersions.value.find(
    (v) => v.modelKey === params.model && v.versionKey === params.version,
  );
  return Boolean(row?.ready);
});

const submitDisabled = computed(() => {
  if (generating.value) return true;
  if (!params.model || !modelReady.value) return true;
  if (!portraitAssetId.value && !portraitFile.value) return true;
  if (mode.value === 'lip_sync' && !audioAssetId.value && !audioFile.value) return true;
  if (mode.value === 'script' && !scriptText.value.trim()) return true;
  return false;
});

const activeVideoTasks = computed(() => {
  const running = tasksStore.queueState.running.filter((t: Task) =>
    String(t.kind || '').startsWith('video.'),
  );
  const queued = tasksStore.queueState.queued.filter((t: Task) =>
    String(t.kind || '').startsWith('video.'),
  );
  return [...running, ...queued].map((t: Task) => {
    const live = tasksStore.liveTaskProgress[t.id];
    return live ? { ...t, ...live } : t;
  });
});

const composerBusy = computed(() => {
  if (generating.value) return true;
  return activeVideoTasks.value.some((t) =>
    ACTIVE_COMPOSER_TASK_STATUSES.has(String(t.status || '')),
  );
});

function onGallerySelect(item: GalleryItem) {
  const idx = galleryItems.value.findIndex((it) => it.path === item.path);
  selectedVideoIndex.value = idx >= 0 ? idx : 0;
  videoPreviewVisible.value = true;
}

function onCardAction({ action, item }: { action: string; item: GalleryItem }) {
  if (action === 'delete') {
    deleteItem(item);
  } else if (action === 'download') {
    const url = api.gallery.getImageUrl(item.path);
    const a = document.createElement('a');
    a.href = url;
    a.download = item.name || 'video.mp4';
    a.click();
  } else if (action === 'compose-from-item') {
    onGallerySelect(item);
    openComposerDrawer();
  }
}

function applyComposerParams(next: Record<string, unknown>) {
  Object.assign(params, next);
}

function syncParamsFromModelConfig() {
  const cfg = currentModelConfig.value;
  const p = (cfg?.parameters as Record<string, unknown>) || {};
  if (p.steps && typeof (p.steps as any).default === 'number') {
    params.steps = (p.steps as any).default;
  }
  if (p.num_frames && typeof (p.num_frames as any).default === 'number') {
    params.num_frames = (p.num_frames as any).default;
  }
  if (p.fps && typeof (p.fps as any).default === 'number') {
    params.fps = (p.fps as any).default;
  }
  const preset = (p.resolution_presets as any)?.default || '512x512';
  selectedResolution.value = preset;
  const [w, h] = preset.split('x').map((x: string) => parseInt(x, 10));
  if (w > 0) params.width = w;
  if (h > 0) params.height = h;
}

function onModelChange() {
  const parsed = parseModelVersion(selectedModelVersion.value);
  params.model = parsed.model;
  params.version = parsed.version;
  syncParamsFromModelConfig();
}

watch(selectedResolution, (val) => {
  const [w, h] = String(val || '512x512').split('x').map((x) => parseInt(x, 10));
  if (w > 0) params.width = w;
  if (h > 0) params.height = h;
});

watch(
  avatarModelSelectOptions,
  (opts) => {
    if (!opts.length) return;
    const still = opts.some((o) => o.value === selectedModelVersion.value && !o.disabled);
    if (!still) {
      const pick = opts.find((o) => !o.disabled) || opts[0];
      selectedModelVersion.value = pick.value;
      onModelChange();
    }
  },
  { immediate: true },
);

function openPortraitPicker() {
  portraitPickerOpen.value = true;
}

function onPortraitPicked(payload: { path?: string; previewUrl?: string }) {
  portraitPickerOpen.value = false;
  const path = String(payload?.path || '');
  if (!path.startsWith('asset:')) return;
  portraitAssetId.value = path.slice('asset:'.length);
  portraitPreviewUrl.value = payload.previewUrl || '';
  portraitLabel.value = path;
  portraitFile.value = null;
}

function clearPortrait() {
  portraitAssetId.value = '';
  portraitPreviewUrl.value = '';
  portraitLabel.value = '';
  portraitFile.value = null;
}

function openAudioPicker() {
  audioPickerOpen.value = true;
}

function onAudioPicked(payload: { path?: string; previewUrl?: string }) {
  audioPickerOpen.value = false;
  const path = String(payload?.path || '');
  if (!path.startsWith('asset:')) return;
  audioAssetId.value = path.slice('asset:'.length);
  audioLabel.value = path;
  audioFile.value = null;
}

function clearAudio() {
  audioAssetId.value = '';
  audioLabel.value = '';
  audioFile.value = null;
}

async function resolveAssetId(existingId: string, file: File | null): Promise<string | null> {
  if (existingId) return existingId;
  if (!file) return null;
  const up = (await api.gen.uploadAsset(file)) as { id?: string };
  return up?.id || null;
}

async function runGenerationTask(submitRes: unknown) {
  const tid = taskIdFromSubmitResponse(submitRes);
  if (!tid) {
    throw new Error($tt('studio.missingTaskId'));
  }
  tasksStore.clearTaskLogs(tid);
  tasksStore.appendTaskLog(tid, $tt('studio.startingGen'), 'info');
  tasksStore.registerPageOwnedStream(tid);
  toast.success($tt('studio.taskQueued'));
  composerDrawerOpen.value = false;

  api.gen.streamMediaTask(tid, {
    onLog: (logData: unknown) => {
      tasksStore.ingestTaskLog(tid, logData);
    },
    onTrace: (traceData: unknown) => {
      tasksStore.ingestTaskPipelineTrace(tid, traceData);
    },
    onDone: async (doneData: { status?: string; error?: string; error_message?: string }) => {
      generating.value = false;
      tasksStore.unregisterPageOwnedStream(tid);
      if (doneData.status === 'completed') {
        tasksStore.appendTaskLog(tid, $tt('studio.genComplete'), 'success');
        toast.success($tt('studio.genComplete'));
        await loadGallery(true);
      } else if (doneData.status === 'failed') {
        let errMsg = String(doneData.error || doneData.error_message || '');
        if (!errMsg) {
          try {
            const updated = (await api.gen.getMediaTask(tid)) as Record<string, unknown>;
            errMsg = String(updated.error || updated.error_message || '');
          } catch {
            /* keep empty */
          }
        }
        tasksStore.appendTaskLog(tid, $tt('studio.genFailed', { msg: errMsg }), 'error');
        toast.error($tt('studio.genFailed', { msg: errMsg }));
      }
    },
    onError: () => {
      generating.value = false;
      tasksStore.unregisterPageOwnedStream(tid);
      tasksStore.appendTaskLog(tid, $tt('studio.connectionLost'), 'warning');
      toast.warning($tt('studio.connectionLost'));
    },
  });
}

async function onGenerate() {
  if (submitDisabled.value) return;
  warnIfRiskyMemory(systemInfo?.value, currentModelConfig.value as Record<string, unknown>);

  generating.value = true;
  try {
    const refId = await resolveAssetId(portraitAssetId.value, portraitFile.value);
    if (!refId) {
      toast.warning($tt('avatar.needPortrait'));
      generating.value = false;
      return;
    }

    const modelField = params.version ? `${params.model}:${params.version}` : params.model;
    const commonBody: Record<string, unknown> = {
      model: modelField,
      title: params.title || '',
      prompt: params.prompt || '',
      negative_prompt: params.negative_prompt || '',
      reference_asset_id: refId,
      size: `${params.width}x${params.height}`,
      num_frames: params.num_frames,
      fps: params.fps,
      steps: params.steps,
      seed: params.seed,
    };

    let submitRes: unknown;
    if (mode.value === 'script') {
      const text = scriptText.value.trim();
      if (!text) {
        toast.warning($tt('avatar.needScript'));
        generating.value = false;
        return;
      }
      submitRes = await api.gen.createVideoAvatarScript({
        ...commonBody,
        script_text: text,
        tts_model: '',
        voice_id: '',
      });
    } else {
      const audId = await resolveAssetId(audioAssetId.value, audioFile.value);
      if (!audId) {
        toast.warning($tt('avatar.needAudio'));
        generating.value = false;
        return;
      }
      submitRes = await api.gen.createVideoAvatar({
        ...commonBody,
        audio_asset_id: audId,
      });
    }
    await runGenerationTask(submitRes);
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    toast.error($tt('studio.error', { msg }));
    generating.value = false;
  }
}

async function loadModelRegistry() {
  const [registryData, detailedStatusData] = await Promise.all([
    registryStore.load(true),
    api.settings.getModelsDetailedStatus(),
  ]);
  modelRegistry.value = (registryData && (registryData as { models?: Record<string, unknown> }).models) || {};
  modelsDetailedStatus.value = (detailedStatusData as Record<string, unknown>) || {};
}

onMounted(async () => {
  tasksStore.ensureQueuePoller();
  window.addEventListener('keydown', onCreatePageKeydown);
  try {
    await loadModelRegistry();
  } catch (e) {
    console.error('Failed to load model registry:', e);
  }
  loadGallery(true);
});

onBeforeUnmount(() => {
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
</script>

<style scoped>
</style>
