<template>
  <StudioLayout
    class="studio-create-page"
    :freeform="viewMode === 'canvas'"
    hide-composer-bar
    @scroll="onCanvasScroll"
  >
    <template #filters>
      <!-- List: search canvases + new / refresh -->
      <div v-if="viewMode === 'grid'" class="image-create-filters image-create-filters--list">
        <DqInput
          v-model="stripSearchText"
          size="sm"
          class="image-create-filters__search"
          :placeholder="$t('canvas.stripSearchPlaceholder')"
          clearable
        >
          <template #prefix><DqIcon><Search /></DqIcon></template>
        </DqInput>
        <div class="image-create-filters__spacer" aria-hidden="true" />
        <div class="image-create-filters__actions">
          <DqButton size="sm" @click="onNewCanvas">{{ $t('canvas.newCanvas') }}</DqButton>
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
      <!-- Detail: back + title + refresh + text-to-image composer -->
      <div v-else class="image-create-filters image-create-filters--detail">
        <DqButton size="sm" type="text" @click="onBackToStripList">
          ← {{ $t('canvas.backToCanvasList') }}
        </DqButton>
        <span class="image-create-filters__session-title" :title="canvasSessionTitle">
          {{ canvasSessionTitle }}
        </span>
        <div class="image-create-filters__spacer" />
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
          @click="openTextGenComposer"
        >
          {{ $t('studio.openComposer') }}
        </DqButton>
      </div>
    </template>

    <template #canvas>
      <StudioCanvasStripList
        v-if="viewMode === 'grid'"
        :sessions="canvasSessions"
        :search-text="stripSearchText"
        :session-asset-ids="sessionAssetIdsBySession"
        :loading="!canvasReady"
        :active-tasks="activeImageTasks"
        :gallery-by-asset-id="galleryByAssetId"
        @open="onStripOpen"
        @preview="onStripPreview"
        @create="onStripCreate"
        @rename="onStripRename"
        @delete="onStripDelete"
      />
      <InfiniteCanvas
        v-else
        ref="infiniteCanvasRef"
        :items="galleryItems"
        media="image"
        hide-composer-entry
        :editing-path="showEditorDrawer && editDrawerItem ? editDrawerItem.path : ''"
        :mask-preview="canvasMaskPreview"
        :extend-preview="canvasExtendPreview"
        @use-as-reference="onCanvasUseAsRef"
        @use-as-control="onCanvasUseAsControl"
        @card-action="onCardAction"
        @download="downloadItem"
        @toggle-grid-view="onBackToStripList"
        @node-selected="onCanvasNodeSelected"
        @session-ready="onCanvasSessionReady"
        @open-preview="onCanvasOpenPreview"
        @composer-restore="onCanvasComposerRestore"
        @overlay-cleared="onCanvasOverlayCleared"
        @open-composer="onCanvasOpenComposer"
        @request-close-composer="closeComposerDrawer()"
      />
    </template>
  </StudioLayout>

  <StudioComposerHost
    v-if="viewMode === 'canvas' || composerDrawerOpen"
    v-model:open="composerDrawerOpen"
    :drawer-title="$t('studio.composerDrawerTitle')"
  >
    <ImageComposer
        v-model="params.prompt"
        v-model:title="params.title"
        v-model:model="selectedModelVersion"
        v-model:size="selectedSize"
        v-model:batch-count="batchCount"
        :composer-busy="composerBusy"
        :submitting="queueSubmitting"
        :generating="generating"
        :can-generate="canGenerate"
        :model-options="modelSelectOptions"
        :size-options="sizeOptions"
        :styles="filteredPresets"
        :params="params"
        :has-custom-params="hasCustomParams"
        :show-negative-prompt="!!currentModelConfig?.parameters?.negative_prompt_support"
        :reference-image="referenceImage"
        :extra-reference-images="extraReferenceImages"
        :references-blocked="referencesBlocked"
        :generate-block-reason="generateBlockReason"
        :control-image="controlImage"
        :mode="imageMode"
        :mode-options="[]"
        :current-model-config="currentModelConfig"
        :compatible-loras="compatibleLoras"
        :compatible-control-nets="compatibleControlNets"
        :control-net-runtime-available="controlNetHostRuntimeAvailable"
        @update:mode="onModeChange"
        @generate="onComposerSubmit"
        @pick-reference="openAssetPicker('reference')"
        @remove-reference="removeReferenceImage"
        @pick-extra-reference="openAssetPicker('extra_reference')"
        @remove-extra-reference="removeExtraReferenceImage"
        @pick-control="openAssetPicker('control')"
        @remove-control="removeControlImage"
        @model-change="onModelVersionChange"
        @reset-defaults="resetToDefaults"
        @enhance="onEnhancePrompt"
        @reverse-prompt="onReversePromptFromReference"
        :prompt-apply-preview="promptApplyPreview"
        :successor-hint="successorHint"
        @prompt-apply-replace="onPromptApplyReplace"
        @prompt-apply-append="onPromptApplyAppend"
        @prompt-apply-dismiss="promptApply.clear()"
        @successor-switch="onSuccessorSwitch"
        @successor-dismiss="onSuccessorDismiss"
        :enhancing="isEnhancing"
        :reversing="isReversing"
      />
  </StudioComposerHost>

  <!-- Asset picker for reference image -->
  <DqDialog v-model:open="showAssetPicker" :title="$t('assetPicker.dialogTitle')" width="70%">
    <AssetPicker
      accept-kind="image"
      :recent-gallery="recentImages"
      @pick="onAssetPickerPick"
    />
  </DqDialog>

  <!-- Preview dialog: canvas detail uses session node order -->
  <GalleryPreviewDialog
    v-model:open="previewVisible"
    v-model:index="selectedImageIndex"
    :items="previewItems"
    media="image"
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

  <!-- Unified Editor Drawer: retouch / extend / upscale -->
  <DqDrawer
    v-model:open="showEditorDrawer"
    :title="editorDrawerTitle"
    direction="rtl"
    size="520px"
    class="studio-image-editor-drawer"
  >
    <div v-if="editDrawerItem" class="studio-editor-drawer">
      <div v-if="editorMode === 'retouch'" class="studio-retouch-panel">
        <DqPrefPane class="studio-create-pref-pane studio-editor-drawer-pref-pane">
          <DqPrefRow :label="$t('studio.model')">
            <ModelSelect
              v-model="retouchModelVersion"
              :options="retouchModelOptions"
              :placeholder="$t('studio.selectModel')"
            />
          </DqPrefRow>
        </DqPrefPane>

        <p class="studio-retouch-fill-hint">{{ retouchHintText }}</p>
        <DqPrefPane class="studio-create-pref-pane studio-editor-drawer-pref-pane">
          <DqPrefRow :label="$t('studio.retouchPromptLabel')">
            <DqInput
              v-model="params.prompt"
              type="textarea"
              :rows="3"
              :placeholder="$t('studio.retouchPromptPlaceholder')"
            />
          </DqPrefRow>
        </DqPrefPane>
        <CreateFillEditParams :params="fillEditParams" />
        <div class="studio-retouch-editor-wrap">
          <ImageEditor
            ref="imageEditorRef"
            :src="getImageUrl(editDrawerItem)"
            mode="inpainting"
            :show-submit-button="false"
            @mask-preview="onEditorMaskPreview"
          />
        </div>
        <DqButton type="primary" block class="studio-drawer-submit" @click="onEditorSubmit">
          {{ $t('action.image.retouch') }}
        </DqButton>
      </div>
      <div v-else-if="editorMode === 'extend'" class="studio-extend-panel">
        <p class="studio-retouch-fill-hint">{{ $t('studio.extendFillHint') }}</p>
        <CreateFillEditParams :params="fillEditParams" />
        <CreateExtendParams
          v-model="extendModelVersion"
          :model-options="extendModelOptions"
          :params="extendParams"
        />
        <DqButton type="primary" block class="studio-drawer-submit" @click="onExtendSubmit">
          {{ $t('action.image.extend') }}
        </DqButton>
      </div>
      <div v-else-if="editorMode === 'upscale'" class="studio-upscale-panel">
        <CreateUpscaleParams
          v-model="upscaleModelVersion"
          :model-options="upscaleModelOptions"
          :params="upscaleParams"
          media="image"
        />
        <DqButton type="primary" block class="studio-drawer-submit" @click="onUpscaleSubmit">
          {{ $t('action.image.upscale') }}
        </DqButton>
      </div>
    </div>
  </DqDrawer>

  <!-- Lineage Panel -->
  <StudioLineagePanel
    v-model:modelValue="showLineageDrawer"
    :asset-id="lineageTargetAssetId"
    :on-canvas-ids="canvasAssetIdsOnCanvas"
    @focus-asset="onLineageFocusAsset"
  />
</template>

<script setup lang="ts">
// @ts-nocheck — legacy create view; narrow types in a follow-up pass
import { ref, reactive, computed, watch, onMounted, onUnmounted, inject, nextTick, unref } from 'vue';
import type { Ref } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import { toast } from '@/utils/feedback';
import { api, taskIdFromSubmitResponse } from '@/utils/api';
import { $tt, $mn, $md, $mvn, $pn } from '@/utils/i18n';
import { DQ_STORAGE, getItem, setItem } from '@/utils/storage';
import { applyPromptDraft, consumePromptDraft } from '@/utils/promptApply';
import { usePromptApplyOffer } from '@/composables/usePromptApplyOffer';
import { useTasksStore } from '@/stores/tasks';
import { useRegistryStore } from '@/stores/registry';
import type { SystemInfo, GalleryItem, Task } from '@/types';
import { applyDefaults, buildResolutionSizeOptions, extraReferenceSlotCap, hasDeviation, parseSizeValue, pickResolutionForModel, referenceImageSlotCap, schedulerForModel, strengthDefaultFromRegistry, strengthToSourceFidelity } from '@/utils/registryParamSchema';
import { getImageSizeForModel, migrateLegacyImageLastSize, setImageSizeForModel } from '@/utils/imageSizeStorage';

import { warnIfRiskyMemory } from '@/composables/memoryHint';
import { useComposerLlm } from '@/composables/useComposerLlm';
import { assetIdFromGalleryPath } from '@/utils/copilotHandoff';
import {
  assetIdsFromGalleryItems,
  navigateToLoraTrainWithAssets,
} from '@/utils/loraTrainHandoff';
import {
  applyLoraComposeOverrides,
  findCompatibleLora,
  withRegistryLoraMeta,
  type CompatibleLoraRow,
} from '@/utils/loraAdapterMeta';
import { reconcileVersionPickerSelection } from '@/composables/useModelRegistryFilters';
import { applyModelVersionFilters } from '@/utils/modelPickerFilters';
import { pickDefaultVersionKey } from '@/utils/defaultModelSettings';
import { dismissSuccessorHint, isSuccessorHintDismissed } from '@/utils/modelSuccessor';
import { ACTIVE_COMPOSER_TASK_STATUSES, splitComposerPromptLines } from '@/utils/composerQueue';
import { seedForSubmit } from '@/utils/seed';
// Studio components
import { Refresh, Search } from '@danqing/dq-shell';
import StudioLayout from '@/components/studio/StudioLayout.vue';
import StudioCanvasStripList from '@/components/studio/StudioCanvasStripList.vue';
import ImageComposer from '@/components/studio/ImageComposer.vue';
import ModelSelect from '@/components/model/ModelSelect.vue';
import { modelSelectOption } from '@/utils/modelSelectOption';
import StudioComposerHost from '@/components/studio/StudioComposerHost.vue';
import AssetPicker from '@/components/asset/AssetPicker.vue';
import ImageEditor from '@/components/image/ImageEditor.vue';
import CreateExtendParams from '@/components/create/CreateExtendParams.vue';
import CreateFillEditParams from '@/components/create/CreateFillEditParams.vue';
import CreateUpscaleParams from '@/components/create/CreateUpscaleParams.vue';
import GalleryPreviewDialog from '@/components/gallery/GalleryPreviewDialog.vue';
import StudioLineagePanel from '@/components/studio/StudioLineagePanel.vue';
import InfiniteCanvas from '@/components/studio/InfiniteCanvas.vue';
import { canvasAutoAddEnabled, useCanvasStore } from '@/composables/useCanvasStore';
import { useComposerDrawer } from '@/composables/useComposerDrawer';
import {
  activateCanvasViewForResults,
  maybeShowCanvasWorkspaceHint,
} from '@/utils/canvasWorkspaceHint';
import { galleryStubForAssetPath, previewUrlForGalleryItem } from '@/utils/canvasAssets';
import { CANVAS_OVERLAY_KINDS } from '@/utils/canvasOverlays';
import { useStudioGallery } from '@/composables/useStudioGallery';
import { isStructuralComposerControlNet } from '@/utils/composerModelFilter';
import {
  appendZImageEnhancementFields,
  applyControlNetRegistryDefaults,
  buildStructuralGuidePayload,
  fillModelRegistryDefaultsPatch,
  isStandaloneEditControlNet,
  isStructuralRetouchControlNet,
  isControlNetHostRuntimeAvailable,
  pickDefaultStructuralControlNet,
  resolveControlAssetId,
  useCompatibleControlNets,
  validateFillEditPrompt,
  validateStructuralGuideForCreate,
} from '@/composables/useStructuralGuide';

/* ------------------------------------------------------------------ */
/*  Injected / External                                                */
/* ------------------------------------------------------------------ */

const systemInfo = inject<Ref<SystemInfo>>('systemInfo');
const tasksStore = useTasksStore();
const registryStore = useRegistryStore();
const router = useRouter();
const route = useRoute();

/* ------------------------------------------------------------------ */
/*  RegistryActions helpers                                            */
/* ------------------------------------------------------------------ */

function hasAction(actions: Record<string, unknown>, key: string): boolean {
  if (!actions || typeof actions !== 'object') return false;
  return Object.prototype.hasOwnProperty.call(actions, key) && actions[key] != null;
}
function imageSupportsCreate(actions: Record<string, unknown>): boolean {
  return hasAction(actions, 'create');
}
function imageSupportsUpscale(actions: Record<string, unknown>): boolean {
  return hasAction(actions, 'upscale');
}
function imageSupportsExtend(actions: Record<string, unknown>): boolean {
  return hasAction(actions, 'extend');
}
function imageSupportsRetouch(actions: Record<string, unknown>): boolean {
  return hasAction(actions, 'retouch');
}
function imageModelRow(config: Record<string, unknown>): boolean {
  return config && config.media === 'image' && config.category !== 'loras';
}

/* ------------------------------------------------------------------ */
/*  Params                                                             */
/* ------------------------------------------------------------------ */

/** When true, skip model-default / size-memory writes so an exported recipe stays intact. */
const verificationRecipeActive = ref(false);
const params = reactive<Record<string, unknown>>({
  title: '',
  prompt: '',
  negative_prompt: '',
  model: '',
  version: '',
  steps: 4,
  guidance: 3.5,
  width: 1024,
  height: 1024,
  lora: '',
  lora_scale: 0.8,
  seed: '',
  strength: 0.4,
  img2img: false,
  controlnet: '',
  controlnet_strength: 0.8,
  scheduler: 'flow_match_euler_discrete',
  upscale_scale: 2,
  upscale_denoise: 0.3,
  upscale_tile: 1024,
  extend_directions: ['right'],
  extend_pixels: 256,
});

const selectedModelVersion = ref('');
const selectedSize = ref('1024x1024');
const batchCount = ref(1);
const generating = ref(false);
const queueSubmitting = ref(false);
const infiniteCanvasRef = ref<InstanceType<typeof InfiniteCanvas> | null>(null);
const pendingCanvasAssetIds = ref<string[]>([]);
const imageCanvas = useCanvasStore('image');
const canvasSessions = computed(() => unref(imageCanvas.sessions));
const canvasSessionId = computed(() => unref(imageCanvas.sessionId));
const canvasSessionTitle = computed(() => unref(imageCanvas.sessionTitle));
const canvasReady = computed(() => unref(imageCanvas.ready));
const canvasSelectedItem = ref<GalleryItem | null>(null);
const stripSearchText = ref('');
const stripRenameOpen = ref(false);
const stripRenameDraft = ref('');
const stripRenameId = ref('');

const showEditorDrawer = ref(false);
const {
  composerDrawerOpen,
  openComposerDrawer,
  closeComposerDrawer,
} = useComposerDrawer('image', { editorDrawerOpen: showEditorDrawer });

const savedViewMode = getItem(DQ_STORAGE.IMAGE_VIEW_MODE);
/** grid = canvas strip list; canvas = infinite canvas detail */
const viewMode = ref<'grid' | 'canvas'>(
  savedViewMode === 'canvas' || savedViewMode === 'grid' ? savedViewMode : 'grid'
);

watch(viewMode, (mode) => {
  setItem(DQ_STORAGE.IMAGE_VIEW_MODE, mode);
  if (mode === 'grid') closeComposerDrawer();
});

function onViewModeChange(mode: 'grid' | 'canvas') {
  if (viewMode.value === mode) return;
  if (mode === 'canvas' && selectedGroupId.value) {
    exitGroup();
  }
  viewMode.value = mode;
  if (mode === 'canvas') {
    clearSelection();
    maybeShowCanvasWorkspaceHint();
    nextTick(() => {
      syncCompositorOverlaysOnCanvasEnter();
      persistComposerSnapshot();
      if (Object.keys(imageCanvas.items).length > 0) {
        infiniteCanvasRef.value?.fitAll();
      }
    });
  } else {
    clearSelection();
    closeComposerDrawer();
  }
}

function openTextGenComposer() {
  // List/detail 「创作」and top-bar open are text-to-image only; asset ops enter via toolbar.
  imageMode.value = 'text2img';
  referenceImage.value = null;
  extraReferenceImages.value = [];
  controlImage.value = null;
  imageCanvas.setControlOverlay(null);
  imageCanvas.clearOverlay('reference');
  openComposerDrawer();
}

async function enterCanvasDetail(sessionId: string, opts?: { openComposer?: boolean }) {
  if (sessionId && sessionId !== canvasSessionId.value) {
    await imageCanvas.switchSession(sessionId);
    onCanvasComposerRestore({ ...imageCanvas.composerSnapshot });
  }
  onViewModeChange('canvas');
  if (opts?.openComposer) {
    nextTick(() => openTextGenComposer());
  }
}

async function onStripOpen(sessionId: string) {
  await enterCanvasDetail(sessionId, { openComposer: false });
}

async function onStripPreview(payload: { sessionId: string; assetId: string }) {
  const sessionId = String(payload?.sessionId || '').trim();
  const assetId = String(payload?.assetId || '').trim();
  if (!sessionId || !assetId) return;

  const byPath = new Map(galleryItems.value.map((g) => [g.path, g]));
  const resolveItem = (path: string): GalleryItem =>
    byPath.get(path) || galleryStubForAssetPath(path);

  let paths: string[] = [];
  try {
    const detail = await api.canvas.getSession(sessionId);
    const items = detail?.state?.items || {};
    const board = Object.entries(items)
      .sort((a, b) => (Number(a[1]?.zIndex) || 0) - (Number(b[1]?.zIndex) || 0))
      .map(([path]) => path)
      .filter((p) => p.startsWith('asset:'));
    const boardSet = new Set(board);
    const shelf = (detail?.state?.shelf || []).filter(
      (p) => typeof p === 'string' && p.startsWith('asset:') && !boardSet.has(p),
    );
    paths = [...board, ...shelf];
  } catch (e) {
    toast.error($tt('studio.error', { msg: String(e) }));
    return;
  }

  if (!paths.length) {
    paths = [`asset:${assetId}`];
  }

  let list = sortGalleryItemsByTime(paths.map(resolveItem));
  const q = stripSearchText.value.trim().toLowerCase();
  if (q) {
    const matched = list.filter((g) => {
      const hay = [g.title, g.prompt, g.model, g.name]
        .filter(Boolean)
        .join(' ')
        .toLowerCase();
      return hay.includes(q);
    });
    if (matched.length) list = matched;
  }
  let idx = list.findIndex((g) => assetIdFromGalleryPath(g.path) === assetId);
  if (idx < 0) {
    list = [resolveItem(`asset:${assetId}`), ...list];
    idx = 0;
  }

  stripPreviewOverride.value = list;
  previewSourceSessionId.value = sessionId;
  selectedImageIndex.value = idx;
  previewVisible.value = true;
}

async function onStripCreate(sessionId: string) {
  // Stay on strip list — only bind the target session and open the composer.
  if (sessionId && sessionId !== canvasSessionId.value) {
    await imageCanvas.switchSession(sessionId);
    onCanvasComposerRestore({ ...imageCanvas.composerSnapshot });
  }
  openTextGenComposer();
}

function onBackToStripList() {
  onViewModeChange('grid');
  void imageCanvas.refreshSessions().then(() => rebuildSessionAssetIndex());
}

async function onNewCanvas() {
  try {
    const created = await imageCanvas.createSession();
    toast.success($tt('canvas.sessionCreated'));
    await rebuildSessionAssetIndex();
    await enterCanvasDetail(created.id, { openComposer: true });
  } catch (e) {
    toast.error($tt('studio.error', { msg: String(e) }));
  }
}

async function refreshStripList() {
  try {
    if (!unref(imageCanvas.ready) || !unref(imageCanvas.sessionId)) {
      await imageCanvas.ensureDefaultAndAdoptOrphans();
    } else {
      await imageCanvas.refreshSessions();
    }
    await loadGallery(true);
    await rebuildSessionAssetIndex();
  } catch (e) {
    toast.error($tt('studio.error', { msg: String(e) }));
  }
}

/** Full membership index so strip search can match any asset on a canvas. */
const sessionAssetIdsBySession = ref<Record<string, string[]>>({});

async function rebuildSessionAssetIndex() {
  const sessions = canvasSessions.value || [];
  if (!sessions.length) {
    sessionAssetIdsBySession.value = {};
    return;
  }
  const entries = await Promise.all(
    sessions.map(async (s) => {
      try {
        const detail = await api.canvas.getSession(s.id);
        const ids: string[] = [];
        const seen = new Set<string>();
        for (const p of Object.keys(detail?.state?.items || {})) {
          if (!p.startsWith('asset:')) continue;
          const id = p.slice('asset:'.length);
          if (!id || seen.has(id)) continue;
          seen.add(id);
          ids.push(id);
        }
        for (const p of detail?.state?.shelf || []) {
          if (typeof p !== 'string' || !p.startsWith('asset:')) continue;
          const id = p.slice('asset:'.length);
          if (!id || seen.has(id)) continue;
          seen.add(id);
          ids.push(id);
        }
        return [s.id, ids] as const;
      } catch {
        return [s.id, s.preview_asset_ids || []] as const;
      }
    }),
  );
  sessionAssetIdsBySession.value = Object.fromEntries(entries);
}

function onStripRename(sessionId: string) {
  const s = canvasSessions.value.find((x) => x.id === sessionId);
  stripRenameId.value = sessionId;
  stripRenameDraft.value = s?.title || '';
  stripRenameOpen.value = true;
}

async function commitStripRename() {
  stripRenameOpen.value = false;
  const id = stripRenameId.value;
  const t = stripRenameDraft.value.trim();
  if (!id || !t) return;
  const prev = canvasSessionId.value;
  try {
    if (id !== prev) await imageCanvas.switchSession(id);
    await imageCanvas.renameSession(t);
    if (prev && prev !== id) await imageCanvas.switchSession(prev);
    await imageCanvas.refreshSessions();
    toast.success($tt('canvas.sessionRenamed'));
  } catch (e) {
    toast.error($tt('studio.error', { msg: String(e) }));
  }
}

async function onStripDelete(sessionId: string) {
  try {
    await imageCanvas.deleteSession(sessionId);
    toast.success($tt('canvas.sessionDeleted'));
    await rebuildSessionAssetIndex();
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    if (msg.includes('cannot_delete_last_canvas') || msg.includes('last canvas')) {
      toast.error($tt('canvas.sessionCannotDeleteLast'));
      return;
    }
    toast.error($tt('studio.error', { msg }));
  }
}

const galleryByAssetId = computed(() => {
  const map: Record<
    string,
    { prompt?: string; model?: string; title?: string; name?: string; created_at?: string }
  > = {};
  for (const item of galleryItems.value) {
    const id = assetIdFromGalleryPath(item.path);
    if (!id) continue;
    map[id] = {
      prompt: item.prompt,
      model: item.model,
      title: item.title,
      name: item.name,
      created_at: item.created_at,
    };
  }
  return map;
});

function sortGalleryItemsByTime(items: GalleryItem[]): GalleryItem[] {
  return [...items].sort((a, b) => {
    const c = String(b.created_at || '').localeCompare(String(a.created_at || ''));
    if (c !== 0) return c;
    return String(b.path || '').localeCompare(String(a.path || ''));
  });
}

/** One-shot strip-list preview when asset is not yet in galleryItems. */
const stripPreviewOverride = ref<GalleryItem[] | null>(null);
/** Canvas that owns the asset currently shown in the lightbox (for transfer). */
const previewSourceSessionId = ref('');

/** Preview sequence: board + shelf members when in detail, else gallery. */
const previewItems = computed(() => {
  if (stripPreviewOverride.value?.length) return stripPreviewOverride.value;
  if (viewMode.value !== 'canvas') return galleryItems.value;
  const byPath = new Map(galleryItems.value.map((g) => [g.path, g]));
  const paths = new Set<string>([
    ...Object.keys(imageCanvas.items),
    ...unref(imageCanvas.shelf),
  ]);
  const items = [...paths]
    .filter((p) => p.startsWith('asset:'))
    .map((path) => byPath.get(path) || galleryStubForAssetPath(path));
  return sortGalleryItemsByTime(items);
});

function canvasTitleLabel(title: string): string {
  const t = (title || '').trim().toLowerCase();
  if (
    t === 'default canvas' ||
    t === '默认画布' ||
    t === 'image canvas' ||
    t === 'video canvas' ||
    t === 'audio canvas'
  ) {
    return $tt('canvas.defaultCanvasTitle');
  }
  return title || $tt('canvas.untitledSession');
}

const previewTransferTargets = computed(() => {
  const source = resolvePreviewSourceSessionId();
  return (canvasSessions.value || [])
    .filter((s) => s.id && s.id !== source)
    .map((s) => ({ id: s.id, title: canvasTitleLabel(s.title || '') }));
});

function resolvePreviewSourceSessionId(path?: string): string {
  if (previewSourceSessionId.value) return previewSourceSessionId.value;
  if (viewMode.value === 'canvas' && canvasSessionId.value) return canvasSessionId.value;
  const p = path || previewItems.value[selectedImageIndex.value]?.path || '';
  if (p) {
    const assetId = assetIdFromGalleryPath(p);
    for (const [sid, ids] of Object.entries(sessionAssetIdsBySession.value || {})) {
      if (assetId && ids.includes(assetId)) return sid;
    }
  }
  return canvasSessionId.value || '';
}

function sessionHasAssetPath(
  state: { items?: Record<string, unknown>; shelf?: string[] } | null | undefined,
  path: string,
): boolean {
  if (!state || !path) return false;
  if (state.items && path in state.items) return true;
  return (state.shelf || []).includes(path);
}

function stripAssetFromSessionState(
  state: import('@/types').CanvasSessionState,
  path: string,
): import('@/types').CanvasSessionState {
  const items = { ...(state.items || {}) };
  delete items[path];
  const shelf = (state.shelf || []).filter((p) => p !== path);
  const overlays = { ...(state.overlays || {}) };
  for (const kind of CANVAS_OVERLAY_KINDS) {
    if (overlays[kind]?.path === path) overlays[kind] = null;
  }
  const active =
    state.active_asset_path === path ? '' : state.active_asset_path || '';
  const edges = (state.edges || []).filter(
    (e) => e?.from !== path && e?.to !== path,
  );
  return {
    ...state,
    items,
    shelf,
    overlays,
    active_asset_path: active,
    edges,
  };
}

async function onPreviewTransfer(payload: {
  item: GalleryItem;
  targetSessionId: string;
}) {
  const path = String(payload?.item?.path || '').trim();
  const toId = String(payload?.targetSessionId || '').trim();
  if (!path.startsWith('asset:') || !toId) return;

  const fromId = resolvePreviewSourceSessionId(path);
  if (!fromId) {
    toast.error($tt('studio.error', { msg: $tt('canvas.transferNeedSource') }));
    return;
  }
  if (fromId === toId) return;

  try {
    // Stop local debounce / wait out in-flight writes so they cannot resurrect the asset.
    await imageCanvas.flushPersistBarrier();

    const [fromDetail, toDetail] = await Promise.all([
      api.canvas.getSession(fromId),
      api.canvas.getSession(toId),
    ]);

    const fromState = stripAssetFromSessionState(fromDetail.state, path);
    const toItems = { ...(toDetail.state?.items || {}) };
    const toShelf = [...(toDetail.state?.shelf || [])].filter((p) => p !== path);
    if (!(path in toItems) && !toShelf.includes(path)) toShelf.push(path);
    const toState = {
      ...toDetail.state,
      items: toItems,
      shelf: toShelf,
    };

    await api.canvas.updateSession(fromId, { state: fromState });
    await api.canvas.updateSession(toId, { state: toState });

    // If a stale persist raced us, source may still list the asset — fix once more.
    const verifyFrom = await api.canvas.getSession(fromId);
    if (sessionHasAssetPath(verifyFrom.state, path)) {
      await api.canvas.updateSession(fromId, {
        state: stripAssetFromSessionState(verifyFrom.state, path),
      });
    }

    const liveId = unref(imageCanvas.sessionId);
    if (liveId === fromId || liveId === toId) {
      imageCanvas.cancelPendingPersist();
      const live = await api.canvas.getSession(liveId);
      imageCanvas.applyState(live.state);
    }

    if (stripPreviewOverride.value?.length) {
      const next = stripPreviewOverride.value.filter((g) => g.path !== path);
      stripPreviewOverride.value = next.length ? next : null;
      if (!next.length) {
        previewVisible.value = false;
      } else {
        selectedImageIndex.value = Math.min(
          selectedImageIndex.value,
          next.length - 1,
        );
      }
    }

    await imageCanvas.refreshSessions();
    await rebuildSessionAssetIndex();

    const targetTitle =
      canvasTitleLabel(
        (canvasSessions.value || []).find((s) => s.id === toId)?.title || '',
      ) || toId;
    toast.success($tt('canvas.transferToCanvasDone', { title: targetTitle }));
  } catch (e) {
    toast.error($tt('studio.error', { msg: String(e) }));
  }
}

function addAssetPathsToCanvas(
  paths: string[],
  opts?: {
    fit?: boolean;
    placement?: 'staging' | 'center';
    focusLast?: boolean;
  }
): string[] {
  if (!paths.length) return [];
  const catalog = galleryItems.value;
  if (viewMode.value === 'canvas' && infiniteCanvasRef.value) {
    return (
      infiniteCanvasRef.value.addPathsToCanvas(paths, {
        fit: opts?.fit,
        placement: opts?.placement ?? 'staging',
        focusLast: opts?.focusLast ?? true,
        galleryItems: catalog,
      }) || []
    );
  }
  return imageCanvas.addPathsFromGallery(paths, catalog, {
    placement: opts?.placement ?? 'center',
  });
}

function onCanvasSessionReady(_payload: { sessionId: string }) {
  if (viewMode.value === 'canvas') {
    nextTick(() => syncCompositorOverlaysOnCanvasEnter());
  }
}

function fillComposerFromGalleryItem(item: GalleryItem) {
  if (item.prompt) params.prompt = item.prompt;
  if (item.title) params.title = item.title;
  if (item.model) {
    const raw = String(item.model);
    if (raw.includes(':')) {
      const [modelKey, versionKey] = raw.split(':', 2);
      params.model = modelKey;
      params.version = versionKey;
      selectedModelVersion.value = `${modelKey}|${versionKey}`;
    } else {
      params.model = raw;
      selectedModelVersion.value = raw;
    }
    loadCompatibleAdapters(String(params.model || ''));
  }
  persistComposerSnapshot();
}

function onCanvasOpenComposer() {
  // Asset-driven open (toolbar) may already set img2img; bare open stays text-gen.
  if (canvasSelectedItem.value) {
    fillComposerFromGalleryItem(canvasSelectedItem.value);
  }
  openComposerDrawer();
}

function onCanvasNodeSelected(item: GalleryItem | null) {
  canvasSelectedItem.value = item;
  if (!item) persistComposerSnapshot();
}

function persistComposerSnapshot() {
  imageCanvas.setComposerSnapshot({
    prompt: String(params.prompt || ''),
    title: String(params.title || ''),
    model: String(params.model || ''),
    version: String(params.version || ''),
    negative_prompt: String(params.negative_prompt || ''),
    seed: params.seed != null ? String(params.seed) : undefined,
    mode: imageMode.value,
    reference_path: referenceImage.value?.path || undefined,
    reference_image_paths: extraReferenceImages.value.map((img) => img.path),
    control_path: controlImage.value?.path || undefined,
    controlnet: params.controlnet ? String(params.controlnet) : undefined,
    controlnet_strength:
      params.controlnet != null ? String(params.controlnet_strength ?? 0.8) : undefined,
    extend_directions: JSON.stringify(extendParams.extend_directions || []),
    extend_pixels: String(extendParams.extend_pixels ?? 256),
    editor_mode: editorMode.value,
    edit_asset_path: editDrawerItem.value?.path || undefined,
    retouch_model_version: retouchModelVersion.value || undefined,
    extend_model_version: extendModelVersion.value || undefined,
    upscale_model_version: upscaleModelVersion.value || undefined,
    upscale_scale: String(upscaleParams.upscale_scale),
    upscale_denoise: String(upscaleParams.upscale_denoise),
    fill_edit_steps: String(fillEditParams.steps),
    fill_edit_guidance: String(fillEditParams.guidance),
  });
}

function restoreFillEditParamsFromSnapshot(snapshot: {
  fill_edit_steps?: string;
  fill_edit_guidance?: string;
}) {
  if (snapshot.fill_edit_steps != null) {
    const steps = parseInt(snapshot.fill_edit_steps, 10);
    if (!Number.isNaN(steps)) fillEditParams.steps = Math.min(50, Math.max(1, steps));
  }
  if (snapshot.fill_edit_guidance != null) {
    const guidance = parseFloat(snapshot.fill_edit_guidance);
    if (!Number.isNaN(guidance)) fillEditParams.guidance = Math.min(50, Math.max(1, guidance));
  }
}

function restoreEditorParamsFromSnapshot(snapshot: {
  editor_mode?: string;
  edit_asset_path?: string;
  retouch_model_version?: string;
  extend_model_version?: string;
  upscale_model_version?: string;
  upscale_scale?: string;
  upscale_denoise?: string;
  fill_edit_steps?: string;
  fill_edit_guidance?: string;
}) {
  if (
    snapshot.editor_mode === 'retouch' ||
    snapshot.editor_mode === 'extend' ||
    snapshot.editor_mode === 'upscale'
  ) {
    editorMode.value = snapshot.editor_mode;
  }
  if (snapshot.edit_asset_path) {
    const item = galleryItems.value.find((g) => g.path === snapshot.edit_asset_path);
    if (item) editDrawerItem.value = item;
  }
  if (snapshot.retouch_model_version) {
    retouchModelVersion.value = snapshot.retouch_model_version;
  }
  if (snapshot.extend_model_version) {
    extendModelVersion.value = snapshot.extend_model_version;
  }
  if (snapshot.upscale_model_version) {
    upscaleModelVersion.value = snapshot.upscale_model_version;
  }
  if (snapshot.upscale_scale != null) {
    const scale = parseInt(snapshot.upscale_scale, 10);
    if (!Number.isNaN(scale)) upscaleParams.upscale_scale = scale;
  }
  if (snapshot.upscale_denoise != null) {
    const denoise = parseFloat(snapshot.upscale_denoise);
    if (!Number.isNaN(denoise)) upscaleParams.upscale_denoise = denoise;
  }
  restoreFillEditParamsFromSnapshot(snapshot);
}

function restoreExtendParamsFromSnapshot(snapshot: {
  extend_directions?: string;
  extend_pixels?: string;
}) {
  if (snapshot.extend_directions) {
    try {
      const parsed = JSON.parse(snapshot.extend_directions);
      if (Array.isArray(parsed)) {
        extendParams.extend_directions = parsed.filter((d: string) =>
          ['top', 'bottom', 'left', 'right'].includes(d)
        );
      }
    } catch {
      /* ignore malformed snapshot */
    }
  }
  if (snapshot.extend_pixels != null) {
    const px = parseInt(snapshot.extend_pixels, 10);
    if (!Number.isNaN(px)) {
      extendParams.extend_pixels = Math.min(2048, Math.max(64, px));
    }
  }
}

function onCanvasComposerRestore(snapshot: {
  prompt?: string;
  title?: string;
  model?: string;
  version?: string;
  negative_prompt?: string;
  seed?: string;
  mode?: string;
  reference_path?: string;
  reference_image_paths?: string[];
  control_path?: string;
  controlnet?: string;
  controlnet_strength?: string;
  extend_directions?: string;
  extend_pixels?: string;
  editor_mode?: string;
  edit_asset_path?: string;
  retouch_model_version?: string;
  extend_model_version?: string;
  upscale_model_version?: string;
  upscale_scale?: string;
  upscale_denoise?: string;
  fill_edit_steps?: string;
  fill_edit_guidance?: string;
}) {
  if (!snapshot || typeof snapshot !== 'object') return;
  if (snapshot.prompt != null) params.prompt = snapshot.prompt;
  if (snapshot.title != null) params.title = snapshot.title;
  if (snapshot.negative_prompt != null) params.negative_prompt = snapshot.negative_prompt;
  if (snapshot.seed != null) params.seed = snapshot.seed;
  if (snapshot.mode === 'img2img' || snapshot.mode === 'text2img') {
    imageMode.value = snapshot.mode;
  }
  if (snapshot.model) {
    params.model = snapshot.model;
    if (snapshot.version) {
      params.version = snapshot.version;
      selectedModelVersion.value = `${snapshot.model}|${snapshot.version}`;
    } else {
      selectedModelVersion.value = snapshot.model;
    }
    loadCompatibleAdapters(snapshot.model);
  }
  if (snapshot.reference_path) {
    referenceImage.value = bindComposerImageFromOverlayPath(snapshot.reference_path);
    imageMode.value = 'img2img';
  }
  if (snapshot.reference_image_paths?.length) {
    extraReferenceImages.value = snapshot.reference_image_paths.map((path) =>
      bindComposerImageFromOverlayPath(path),
    );
  } else {
    extraReferenceImages.value = [];
  }
  if (snapshot.control_path) {
    controlImage.value = bindComposerImageFromOverlayPath(snapshot.control_path);
  }
  if (snapshot.controlnet) {
    params.controlnet = snapshot.controlnet;
    const s = parseFloat(String(snapshot.controlnet_strength ?? ''));
    if (!Number.isNaN(s)) params.controlnet_strength = s;
  }
  restoreExtendParamsFromSnapshot(snapshot);
  restoreEditorParamsFromSnapshot(snapshot);
  if (viewMode.value === 'canvas') {
    nextTick(() => syncCompositorOverlaysOnCanvasEnter());
  }
}

function canvasImageRelationType(): string {
  if (showEditorDrawer.value) {
    const m = editorMode.value;
    if (m === 'retouch' || m === 'extend' || m === 'upscale') return m;
  }
  if (imageMode.value === 'img2img' && referenceImage.value) return 'img2img';
  if (controlImage.value) return 'controlnet';
  return 'create';
}

function buildCanvasMeta(extra: Record<string, unknown> = {}): Record<string, unknown> {
  const meta: Record<string, unknown> = { ...extra };
  const sid = imageCanvas.sessionId.value;
  if (sid) meta.canvas_session_id = sid;
  const parentPath =
    (showEditorDrawer.value && editDrawerItem.value?.path) ||
    canvasSelectedItem.value?.path ||
    imageCanvas.activeAssetPath.value ||
    '';
  if (parentPath.startsWith('asset:')) {
    meta.parent_asset_id = parentPath.slice('asset:'.length);
    meta.relation_type = canvasImageRelationType();
  }
  return meta;
}

function shouldAutoAddToCanvas(): boolean {
  return viewMode.value === 'canvas' || canvasAutoAddEnabled('image');
}

/* ------------------------------------------------------------------ */
/*  Reference Image (new: replaces old edit mode tabs)                 */
/* ------------------------------------------------------------------ */

const referenceImage = ref<{ previewUrl: string; path: string; assetId?: string } | null>(null);
const extraReferenceImages = ref<Array<{ previewUrl: string; path: string; assetId?: string }>>([]);
const controlImage = ref<{ previewUrl: string; path: string; assetId?: string } | null>(null);
const showAssetPicker = ref(false);
const assetPickerMode = ref<'reference' | 'extra_reference' | 'control'>('reference');

function openAssetPicker(mode: 'reference' | 'extra_reference' | 'control') {
  assetPickerMode.value = mode;
  showAssetPicker.value = true;
}

function assetIdFromPickPath(path: string): string | undefined {
  return path.startsWith('asset:') ? path.slice('asset:'.length) : undefined;
}

function onReferencePick({ path, previewUrl }: { path: string; previewUrl: string }) {
  if (versionDisallowsReferences()) {
    toast.warning($tt('create.quantizedNoReference'));
    showAssetPicker.value = false;
    return;
  }
  referenceImage.value = { path, previewUrl, assetId: assetIdFromPickPath(path) };
  showAssetPicker.value = false;
  imageMode.value = 'img2img';
}

function modelReferenceParameters(): Record<string, unknown> | undefined {
  return currentModelConfig.value?.parameters as Record<string, unknown> | undefined;
}

function versionDisallowsReferences(): boolean {
  const versions = currentModelConfig.value?.versions as
    | Record<string, { overrides?: Record<string, unknown> }>
    | undefined;
  const block = versions?.[String(params.version || '')];
  return Boolean(block?.overrides?.hidream_quantized_no_edit);
}

const referencesBlocked = computed(() => versionDisallowsReferences());

const generateBlockReason = computed(() => {
  if (!referencesBlocked.value) return '';
  return $tt('create.quantizedNoReference');
});

function onExtraReferencePick({ path, previewUrl }: { path: string; previewUrl: string }) {
  if (versionDisallowsReferences()) {
    toast.warning($tt('create.quantizedNoReference'));
    showAssetPicker.value = false;
    return;
  }
  if (!path.startsWith('asset:')) {
    toast.warning($tt('canvas.controlImageAssetRequired'));
    return;
  }
  const assetId = assetIdFromPickPath(path);
  const primaryId = referenceImage.value?.assetId
    || (referenceImage.value?.path.startsWith('asset:')
      ? referenceImage.value.path.slice('asset:'.length)
      : '');
  if (assetId && assetId === primaryId) {
    showAssetPicker.value = false;
    return;
  }
  if (extraReferenceImages.value.some((img) => img.path === path || (assetId && img.assetId === assetId))) {
    showAssetPicker.value = false;
    return;
  }
  if (extraReferenceImages.value.length >= extraReferenceSlotCap(modelReferenceParameters())) {
    toast.warning($tt('canvas.referenceLimit', { n: referenceImageSlotCap(modelReferenceParameters()) }));
    showAssetPicker.value = false;
    return;
  }
  extraReferenceImages.value = [
    ...extraReferenceImages.value,
    { path, previewUrl, assetId },
  ];
  showAssetPicker.value = false;
}

function removeExtraReferenceImage(index: number) {
  extraReferenceImages.value = extraReferenceImages.value.filter((_, i) => i !== index);
}

function onControlPick({ path, previewUrl }: { path: string; previewUrl: string }) {
  controlImage.value = { path, previewUrl, assetId: assetIdFromPickPath(path) };
  showAssetPicker.value = false;
  syncCanvasControlOverlay(path);
}

function onAssetPickerPick(payload: { path: string; previewUrl: string }) {
  if (assetPickerMode.value === 'control') onControlPick(payload);
  else if (assetPickerMode.value === 'extra_reference') onExtraReferencePick(payload);
  else onReferencePick(payload);
}

function removeControlImage() {
  controlImage.value = null;
  syncCanvasControlOverlay(null);
}

function onCanvasUseAsRef(payload: { path: string; previewUrl: string; quiet?: boolean }) {
  const { path, previewUrl } = payload;
  if (versionDisallowsReferences()) {
    toast.warning($tt('create.quantizedNoReference'));
    return;
  }
  if (payload.quiet) {
    const item = galleryItems.value.find((g) => g.path === path);
    if (item) fillComposerFromGalleryItem(item);
    referenceImage.value = { path, previewUrl, assetId: assetIdFromPickPath(path) };
    extraReferenceImages.value = [];
    imageMode.value = 'img2img';
    return;
  }
  const incomingId = assetIdFromPickPath(path);
  if (!referenceImage.value) {
    referenceImage.value = { path, previewUrl, assetId: incomingId };
    imageMode.value = 'img2img';
    toast.success($tt('canvas.referenceBound'));
    return;
  }
  const primaryId = referenceImage.value.assetId
    || (referenceImage.value.path.startsWith('asset:')
      ? referenceImage.value.path.slice('asset:'.length)
      : '');
  if (path === referenceImage.value.path || (incomingId && incomingId === primaryId)) {
    imageMode.value = 'img2img';
    return;
  }
  const extraCap = extraReferenceSlotCap(modelReferenceParameters());
  if (extraCap <= 0 || extraReferenceImages.value.length >= extraCap) {
    toast.warning($tt('canvas.referenceLimit', { n: referenceImageSlotCap(modelReferenceParameters()) }));
    return;
  }
  if (!path.startsWith('asset:')) {
    toast.warning($tt('canvas.controlImageAssetRequired'));
    return;
  }
  if (extraReferenceImages.value.some((img) => img.path === path || (incomingId && img.assetId === incomingId))) {
    return;
  }
  extraReferenceImages.value = [
    ...extraReferenceImages.value,
    { path, previewUrl, assetId: incomingId },
  ];
  imageMode.value = 'img2img';
  toast.success($tt('canvas.referenceAppended', { n: extraReferenceImages.value.length + 1 }));
}

function onCanvasUseAsControl(payload: { path: string; previewUrl: string; quiet?: boolean }) {
  const { path, previewUrl } = payload;
  if (!path.startsWith('asset:')) {
    toast.warning($tt('canvas.controlImageAssetRequired'));
    return;
  }
  onControlPick({ path, previewUrl });
  syncCanvasControlOverlay(path);
  if (imageMode.value === 'img2img') {
    imageMode.value = 'text2img';
    removeReferenceImage();
  }
  if (!params.controlnet) {
    const key = pickDefaultStructuralControlNet(compatibleControlNets.value);
    if (key) {
      params.controlnet = key;
      applyControlNetRegistryDefaults(key, compatibleControlNets.value, params);
    }
  }
  if (!params.controlnet) {
    toast.warning($tt('canvas.controlNetRequired'));
    return;
  }
  const guideCheck = structuralGuideValidation();
  if (!guideCheck.ok && guideCheck.code === 'controlnet_not_ready') {
    toast.warning($tt('studio.controlnetNotReady'));
    return;
  }
  if (!payload.quiet) toast.success($tt('canvas.controlBound'));
}

function removeReferenceImage() {
  referenceImage.value = null;
  extraReferenceImages.value = [];
}

function syncCanvasControlOverlay(path: string | null) {
  if (!path) {
    imageCanvas.setControlOverlay(null);
    return;
  }
  const item = galleryItems.value.find((g) => g.path === path) ?? null;
  imageCanvas.setControlOverlay(path, item ?? undefined);
}

function bindComposerImageFromOverlayPath(path: string): {
  path: string;
  previewUrl: string;
  assetId?: string;
} {
  const item = galleryItems.value.find((g) => g.path === path);
  return {
    path,
    previewUrl: item ? previewUrlForGalleryItem(item) : api.gallery.getImageUrl(path),
    assetId: assetIdFromPickPath(path),
  };
}

function restoreComposerFromCanvasOverlays() {
  const ref = imageCanvas.overlays.reference;
  if (!referenceImage.value && ref?.path) {
    referenceImage.value = bindComposerImageFromOverlayPath(ref.path);
    imageMode.value = 'img2img';
  }
  if (ref?.path) {
    imageCanvas.clearOverlay('reference');
  }
  const ctrl = imageCanvas.overlays.control;
  if (!controlImage.value && ctrl?.path) {
    controlImage.value = bindComposerImageFromOverlayPath(ctrl.path);
  }
}

function syncCompositorOverlaysOnCanvasEnter() {
  restoreComposerFromCanvasOverlays();
  syncCanvasControlOverlay(controlImage.value?.path ?? null);
}

function onCanvasOverlayCleared(kind: import('@/types').CanvasOverlayKind) {
  if (kind === 'reference') {
    removeReferenceImage();
    if (imageMode.value === 'img2img' && !controlImage.value) {
      imageMode.value = 'text2img';
    }
  } else {
    controlImage.value = null;
  }
}

watch(
  () => referenceImage.value?.path ?? null,
  () => {
    persistComposerSnapshot();
  }
);

watch(
  () => extraReferenceImages.value.map((img) => img.path).join('\n'),
  () => persistComposerSnapshot(),
);

watch(
  () => controlImage.value?.path ?? null,
  (path) => {
    if (viewMode.value === 'canvas') syncCanvasControlOverlay(path);
    persistComposerSnapshot();
  }
);

function onCanvasOpenPreview(item: GalleryItem) {
  previewSourceSessionId.value = canvasSessionId.value || '';
  const list = previewItems.value;
  const idx = list.findIndex((g) => g.path === item.path);
  if (idx >= 0) {
    selectedImageIndex.value = idx;
    previewVisible.value = true;
    return;
  }
  // Node not yet in gallery catalog — fall back to full gallery list.
  const gIdx = galleryItems.value.findIndex((g) => g.path === item.path);
  if (gIdx >= 0) {
    selectedImageIndex.value = gIdx;
    previewVisible.value = true;
  }
}

const { isEnhancing, isReversing, enhance: doEnhance, reversePrompt } = useComposerLlm();
const promptApply = usePromptApplyOffer();
const promptApplyPreview = computed(() => promptApply.pending.value?.result ?? null);

async function onEnhancePrompt(ctx?: { stylePositive?: string }) {
  const prompt = String(params.prompt || '').trim();
  if (!prompt) return;
  const enhanced = await doEnhance(
    prompt,
    ctx?.stylePositive,
    'image_create',
    params.model,
    { quietSuccess: true },
  );
  if (enhanced) {
    promptApply.offer(params.prompt, enhanced, (text) => { params.prompt = text; });
  }
}

async function onReversePromptFromReference() {
  const path = referenceImage.value?.path || '';
  const assetId = referenceImage.value?.assetId || assetIdFromGalleryPath(path);
  if (!assetId) {
    toast.warning($tt('create.reverseNeedAsset'));
    return;
  }
  const prompt = await reversePrompt(assetId, { quietSuccess: true });
  if (prompt) {
    promptApply.offer(params.prompt, prompt, (text) => { params.prompt = text; });
  }
}

function onPromptApplyReplace() {
  promptApply.applyReplace((text) => { params.prompt = text; });
}

function onPromptApplyAppend() {
  promptApply.applyAppend(() => params.prompt, (text) => { params.prompt = text; });
}

/* ------------------------------------------------------------------ */
/*  LoRAs / ControlNets                                                */
/* ------------------------------------------------------------------ */

const compatibleLoras = ref<CompatibleLoraRow[]>([]);
const {
  compatibleControlNets,
  loadCompatibleControlNets,
  clearIfIncompatible: clearControlNetIfIncompatible,
} = useCompatibleControlNets('create');

const controlNetHostRuntimeAvailable = computed(() =>
  isControlNetHostRuntimeAvailable(
    systemInfo?.value?.controlnet_runtime_available,
    compatibleControlNets.value,
    systemInfo?.value,
  ),
);

function syncImageLoraSelection() {
  if (!params.lora) return;
  const row = findCompatibleLora(compatibleLoras.value, params.lora);
  if (!row || row.ready === false) {
    params.lora = '';
    return;
  }
  const bare = String(row.id || params.lora).split(':')[0];
  applyLoraComposeOverrides(
    params as Record<string, unknown>,
    row,
    modelRegistry.value[bare] as Record<string, unknown> | undefined,
  );
}

async function loadCompatibleAdapters(modelKey: string) {
  if (!modelKey) {
    compatibleLoras.value = [];
    compatibleControlNets.value = [];
    params.lora = '';
    return;
  }
  try {
    const loras = await api.settings.getCompatibleLoras(modelKey);
    if (String(params.model || '') !== modelKey) return;
    compatibleLoras.value = ((loras as CompatibleLoraRow[]) || []).map((row) => {
      const bare = String(row.id || '').split(':')[0];
      return withRegistryLoraMeta(
        row,
        modelRegistry.value[bare] as Record<string, unknown> | undefined,
      );
    });
    syncImageLoraSelection();
    await loadCompatibleControlNets(modelKey);
    clearControlNetIfIncompatible(params, () => {
      controlImage.value = null;
      syncCanvasControlOverlay(null);
    });
  } catch (e) {
    console.error('Failed to load compatible adapters:', e);
    compatibleLoras.value = [];
    compatibleControlNets.value = [];
    toast.error($tt('studio.error', { msg: $tt('studio.adapterLoadFailed') }));
  }
}

/* ------------------------------------------------------------------ */
/*  Gallery / Studio Canvas                                            */
/* ------------------------------------------------------------------ */

const {
  galleryItems,
  galleryGroups,
  galleryLoading,
  galleryHasMore,
  groupMode,
  selectedGroupId,
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
  enterGroup,
  exitGroup,
  onCanvasScroll,
  resetGalleryFilters,
  toggleSelect,
  toggleSelectionMode,
  selectAllLoaded,
  deleteItem,
  batchDeleteSelected,
  clearSelection,
} = useStudioGallery('image');

function batchTrainLoraSelected() {
  const selected = galleryItems.value.filter((item) => selectedPaths.value.has(item.path));
  const assetIds = assetIdsFromGalleryItems(selected);
  if (!assetIds.length) {
    toast.warning($tt('loraTrain.needImageAssets'));
    return;
  }
  const ok = navigateToLoraTrainWithAssets(router, assetIds, {
    datasetName: $tt('loraTrain.galleryImportDataset'),
  });
  if (ok) {
    clearSelection();
    toast.success($tt('loraTrain.handoffStarted'));
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

watch(() => params.model, (modelKey) => {
  if (modelKey) loadCompatibleAdapters(String(modelKey));
});

watch(() => params.lora, () => {
  syncImageLoraSelection();
});

watch(
  () => [params.controlnet, params.controlnet_strength] as const,
  () => persistComposerSnapshot(),
);

/* ------------------------------------------------------------------ */
/*  Active Tasks (generating placeholders)                             */
/* ------------------------------------------------------------------ */

const activeImageTasks = computed(() => {
  const running = tasksStore.queueState.running.filter((t: Task) =>
    String(t.kind || '').startsWith('image.')
  );
  const queued = tasksStore.queueState.queued.filter((t: Task) =>
    String(t.kind || '').startsWith('image.')
  );
  return [...running, ...queued].map((t: Task) => {
    const live = tasksStore.liveTaskProgress[t.id];
    return live ? { ...t, ...live } : t;
  });
});

const composerBusy = computed(() => {
  if (generating.value) return true;
  return activeImageTasks.value.some((t) =>
    ACTIVE_COMPOSER_TASK_STATUSES.has(String(t.status || '')),
  );
});

/* ------------------------------------------------------------------ */
/*  Task status helpers                                                */
/* ------------------------------------------------------------------ */

function getStatusType(status: string): string {
  const map: Record<string, string> = {
    pending: 'info',
    queued: 'info',
    submitting: 'info',
    running: 'warning',
    completed: 'success',
    failed: 'danger',
    cancelled: 'info',
  };
  return map[status] || 'info';
}

function getStatusText(status: string): string {
  const map: Record<string, string> = {
    pending: 'studio.pending',
    queued: 'studio.queued',
    submitting: 'studio.submitting',
    running: 'studio.running',
    completed: 'studio.completed',
    failed: 'studio.failed',
    cancelled: 'studio.cancelled',
  };
  const key = map[status] || `studio.${status}`;
  return $tt(key);
}

/* ------------------------------------------------------------------ */
/*  Model registry                                                     */
/* ------------------------------------------------------------------ */

const modelRegistry = ref<Record<string, Record<string, unknown>>>({});
const modelsDetailedStatus = ref<Record<string, { versions?: Record<string, { ready?: boolean; status?: string }> }>>({});

const allVersions = computed(() => {
  const result: Array<Record<string, unknown>> = [];
  for (const [modelKey, config] of Object.entries(modelRegistry.value)) {
    if (!imageModelRow(config)) continue;
    const actions = { ...(config.actions as Record<string, unknown> || {}) };
    const versions = config.versions || { default: { name: 'Default', size: '', default: true } };
    const detailed = modelsDetailedStatus.value[modelKey] || {};
    const versionStatuses = detailed.versions || {};

    for (const [versionKey, versionConfig] of Object.entries(versions as Record<string, Record<string, unknown>>)) {
      const status = versionStatuses[versionKey] || { status: 'not_downloaded', ready: false };
      const size = (versionConfig as Record<string, unknown>).size || '';

      result.push({
        modelKey,
        versionKey,
        name: $mvn(modelKey, config as { name?: string | { zh?: string; en?: string }; name_en?: string }, versionConfig as { name?: string | { zh?: string; en?: string } }),
        description: $md(config as { description?: string | { zh?: string; en?: string }; description_en?: string }),
        size,
        status: status.status,
        ready: status.ready,
        recommended: config.recommended && (versionConfig as Record<string, unknown>).default,
        commercialUseAllowed: config.commercial_use_allowed === true,
        actions,
      });
    }
  }
  return result;
});

const filteredAllVersions = computed(() => {
  return allVersions.value.filter((v) => {
    const row = modelRegistry.value[String(v.modelKey || '')];
    if (isStructuralComposerControlNet(row)) return false;
    const acts = v.actions as Record<string, unknown> || {};
    if (imageMode.value === 'img2img') {
      return hasAction(acts, 'rewrite') || hasAction(acts, 'retouch');
    }
    return imageSupportsCreate(acts);
  });
});

const filteredModelPickerVersions = computed(() => {
  const rows = applyModelVersionFilters(filteredAllVersions.value, {
    installedOnly: true,
    commercialOnly: false, // show all; badge indicates commercial status
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

const modelSelectOptions = computed(() => {
  return filteredModelPickerVersions.value.map((v) => modelSelectOption(v));
});

const currentModelConfig = computed(() => modelRegistry.value[params.model as string] || null);

const successorDismissNonce = ref(0);

const successorHint = computed(() => {
  void successorDismissNonce.value;
  const modelId = String(params.model || '');
  const cfg = currentModelConfig.value;
  const successorId = typeof cfg?.successor === 'string' ? cfg.successor.trim() : '';
  if (!successorId || !modelRegistry.value[successorId]) return null;
  if (isSuccessorHintDismissed(modelId)) return null;
  return {
    successorId,
    successorName: $mn(modelRegistry.value[successorId], successorId),
  };
});

function onSuccessorSwitch() {
  const hint = successorHint.value;
  if (!hint) return;
  const versionKey = pickDefaultVersionKey(
    hint.successorId,
    modelRegistry.value,
    modelsDetailedStatus.value[hint.successorId]?.versions,
  );
  if (!versionKey) return;
  onModelVersionChange(`${hint.successorId}|${versionKey}`);
}

function onSuccessorDismiss() {
  const modelId = String(params.model || '');
  if (!modelId) return;
  dismissSuccessorHint(modelId);
  successorDismissNonce.value += 1;
}
const currentModelDisplayName = computed(() => {
  const c = currentModelConfig.value;
  if (c) {
    return $mn(c as { name?: string | { zh?: string; en?: string }; name_en?: string }, params.model as string);
  }
  return params.model || '';
});

watch(composerDrawerOpen, (open) => {
  if (open) {
    showLineageDrawer.value = false;
    infiniteCanvasRef.value?.closeMutexPanels?.();
  }
});

const selectedModelNotReady = computed(() => {
  if (!params.model || !params.version) return false;
  const detailed = modelsDetailedStatus.value[params.model as string];
  if (!detailed || !detailed.versions) return true;
  const versionStatus = detailed.versions[params.version as string];
  return !versionStatus || !versionStatus.ready;
});

function structuralGuideValidation():
  | { ok: true }
  | { ok: false; code: import('@/composables/useStructuralGuide').StructuralGuideValidationCode } {
  return validateStructuralGuideForCreate({
    controlnet: String(params.controlnet || ''),
    baseModel: String(params.model || ''),
    hasReferenceImage: referenceImage.value != null,
    hasControlImage: controlImage.value != null,
    hostRuntimeAvailable: systemInfo?.value?.controlnet_runtime_available,
    compatibleControlNets: compatibleControlNets.value,
    systemInfo: systemInfo?.value,
  });
}

const canGenerate = computed(() => {
  if (selectedModelNotReady.value) return false;
  if (!String(params.prompt || '').trim()) return false;
  if (referencesBlocked.value && referenceImage.value) return false;
  if (imageMode.value === 'img2img' && !referenceImage.value) return false;
  if (String(params.controlnet || '') && !structuralGuideValidation().ok) return false;
  return true;
});

function parseModelVersionValue(value: string): { modelKey: string; versionKey: string } | null {
  if (!value || typeof value !== 'string') return null;
  const parts = value.split('|');
  if (parts.length !== 2 || !parts[0] || !parts[1]) return null;
  return { modelKey: parts[0], versionKey: parts[1] };
}

function getImageModeStorageKey(mode: string): StorageKey {
  const map: Record<string, StorageKey> = {
    text2img: DQ_STORAGE.IMAGE_MODEL_TEXT2IMG,
    img2img: DQ_STORAGE.IMAGE_MODEL_IMG2IMG,
    retouch: DQ_STORAGE.IMAGE_MODEL_RETOUCH,
    extend: DQ_STORAGE.IMAGE_MODEL_EXTEND,
    upscale: DQ_STORAGE.IMAGE_MODEL_UPSCALE,
  };
  return map[mode] || DQ_STORAGE.IMAGE_MODEL_TEXT2IMG;
}

function isModelAvailable(modelKey: string, versionKey: string): boolean {
  return filteredModelPickerVersions.value.some(
    (v) => v.modelKey === modelKey && v.versionKey === versionKey && v.ready
  );
}

function restoreModelForMode(mode: string) {
  const saved = getItem(getImageModeStorageKey(mode));
  if (saved) {
    const parsed = parseModelVersionValue(saved);
    if (parsed && isModelAvailable(parsed.modelKey, parsed.versionKey)) {
      selectedModelVersion.value = saved;
      params.model = parsed.modelKey;
      params.version = parsed.versionKey;
      loadModelDefaults();
      loadCompatibleAdapters(parsed.modelKey);
      return;
    }
  }
  // fallback: 尝试当前模型是否支持该 mode
  const currentKey = params.model && params.version ? `${params.model}|${params.version}` : '';
  const currentParsed = parseModelVersionValue(currentKey);
  if (currentParsed && isModelAvailable(currentParsed.modelKey, currentParsed.versionKey)) {
    return;
  }
  // fallback: 选第一个可用模型
  if (filteredModelPickerVersions.value.length > 0) {
    const first = filteredModelPickerVersions.value[0];
    selectedModelVersion.value = `${first.modelKey}|${first.versionKey}`;
    params.model = first.modelKey;
    params.version = first.versionKey;
    loadModelDefaults();
    loadCompatibleAdapters(first.modelKey);
  }
}

const editorMode = ref<'retouch' | 'extend' | 'upscale'>('retouch');
const editDrawerItem = ref<GalleryItem | null>(null);
const canvasMaskPreview = ref<import('@/types').CanvasMaskPreviewState | null>(null);

function onEditorMaskPreview(
  payload: { dataUrl: string; width: number; height: number } | null
) {
  if (
    !showEditorDrawer.value ||
    editorMode.value !== 'retouch' ||
    viewMode.value !== 'canvas'
  ) {
    canvasMaskPreview.value = null;
    return;
  }
  canvasMaskPreview.value = payload;
}

watch(showEditorDrawer, (open) => {
  if (!open) canvasMaskPreview.value = null;
});

watch(editorMode, (mode) => {
  if (mode !== 'retouch') canvasMaskPreview.value = null;
});

const canvasExtendPreview = computed((): import('@/types').CanvasExtendPreviewState | null => {
  if (
    !showEditorDrawer.value ||
    editorMode.value !== 'extend' ||
    viewMode.value !== 'canvas' ||
    !editDrawerItem.value
  ) {
    return null;
  }
  const dirs = Array.isArray(extendParams.extend_directions)
    ? extendParams.extend_directions.filter((d: string) =>
        ['top', 'bottom', 'left', 'right'].includes(d)
      )
    : [];
  if (!dirs.length) return null;
  return {
    directions: dirs as import('@/types').CanvasExtendDirection[],
    pixels: Math.min(2048, Math.max(64, Number(extendParams.extend_pixels) || 256)),
  };
});
const showLineageDrawer = ref(false);
const lineageTargetAssetId = ref('');

watch(showLineageDrawer, (open) => {
  if (open) closeComposerDrawer();
});

const canvasAssetIdsOnCanvas = computed(() =>
  Object.keys(imageCanvas.items)
    .filter((p) => p.startsWith('asset:'))
    .map((p) => p.slice('asset:'.length))
);

async function onLineageFocusAsset(assetId: string) {
  if (viewMode.value !== 'canvas') {
    await activateCanvasViewForResults(viewMode, syncCompositorOverlaysOnCanvasEnter);
  }
  infiniteCanvasRef.value?.focusLineageAsset(assetId);
}

const editorDrawerTitle = computed(() => {
  const map: Record<string, string> = {
    retouch: $tt('action.image.retouch'),
    extend: $tt('action.image.extend'),
    upscale: $tt('action.image.upscale'),
  };
  return map[editorMode.value] || '';
});

function onModelVersionChange(value: string) {
  const parsed = parseModelVersionValue(value);
  if (!parsed) return;
  params.model = parsed.modelKey;
  params.version = parsed.versionKey;
  const activeMode = showEditorDrawer.value ? editorMode.value : imageMode.value;
  setItem(getImageModeStorageKey(activeMode), value);
  loadModelDefaults();
  syncResolutionForModel(parsed.modelKey);
  loadCompatibleAdapters(parsed.modelKey);
}

/* ------------------------------------------------------------------ */
/*  Size options                                                       */
/* ------------------------------------------------------------------ */

const sizeOptions = computed(() =>
  buildResolutionSizeOptions(currentModelConfig.value?.parameters as Record<string, unknown> | undefined),
);

function applySelectedSize(val: string) {
  const parsed = parseSizeValue(val);
  if (!parsed) return;
  params.width = parsed.width;
  params.height = parsed.height;
}

function syncResolutionForModel(modelId?: string) {
  if (verificationRecipeActive.value) return;
  const mid = modelId || String(params.model || '');
  if (!mid) return;
  migrateLegacyImageLastSize(mid);
  const pick = pickResolutionForModel(
    currentModelConfig.value?.parameters as Record<string, unknown> | undefined,
    getImageSizeForModel(mid),
  );
  if (!pick) return;
  if (selectedSize.value !== pick) selectedSize.value = pick;
  else applySelectedSize(pick);
}

const imageMode = ref('text2img');

function onModeChange(mode: string) {
  imageMode.value = mode;
  if (mode === 'text2img') {
    removeReferenceImage();
  }
}

watch(imageMode, (newMode, oldMode) => {
  // 保存旧模式模型
  if (oldMode && selectedModelVersion.value) {
    setItem(getImageModeStorageKey(oldMode), selectedModelVersion.value);
  }
  // 恢复新模式模型
  const saved = getItem(getImageModeStorageKey(newMode));
  if (saved) {
    const parsed = parseModelVersionValue(saved);
    if (parsed && isModelAvailable(parsed.modelKey, parsed.versionKey)) {
      selectedModelVersion.value = saved;
      params.model = parsed.modelKey;
      params.version = parsed.versionKey;
      loadModelDefaults();
      syncResolutionForModel(parsed.modelKey);
      loadCompatibleAdapters(parsed.modelKey);
      return;
    }
  }
  // fallback
  if (
    reconcileVersionPickerSelection(filteredModelPickerVersions.value, params, selectedModelVersion)
  ) {
    loadModelDefaults();
    syncResolutionForModel(String(params.model || ''));
    loadCompatibleAdapters(String(params.model || ''));
  }
});

watch(selectedSize, (val) => {
  applySelectedSize(val);
  if (verificationRecipeActive.value) return;
  const mid = String(params.model || '');
  if (mid) setImageSizeForModel(mid, val);
});

watch(sizeOptions, () => {
  syncResolutionForModel();
});

// Drawer 关闭时保存 editor 模型并恢复 composer 模型
watch(showEditorDrawer, (isOpen, wasOpen) => {
  if (wasOpen && !isOpen) {
    // drawer 关闭
    if (selectedModelVersion.value) {
      setItem(getImageModeStorageKey(editorMode.value), selectedModelVersion.value);
    }
    // 恢复 composer 模型
    restoreModelForMode(imageMode.value);
  }
});

const hasCustomParams = computed(() => {
  const config = currentModelConfig.value;
  if (!config || !config.parameters) return false;
  return hasDeviation(config.parameters as Record<string, unknown>, params);
});

/* ------------------------------------------------------------------ */
/*  Presets / Styles                                                   */
/* ------------------------------------------------------------------ */

const presets = ref<Record<string, Record<string, unknown>>>({});

const presetActionFilter = computed(() => new Set(['create']));

const filteredPresets = computed(() => {
  const want = presetActionFilter.value;

  function planPresetShapeOk(preset: Record<string, unknown>) {
    return (
      Array.isArray(preset.applies_to) &&
      (preset.applies_to as unknown[]).length > 0 &&
      preset.media_scope === 'image'
    );
  }

  function matches(preset: Record<string, unknown>) {
    if (!planPresetShapeOk(preset)) return false;
    return (preset.applies_to as string[]).some((k: string) => want.has(k));
  }

  const entries = Object.entries(presets.value)
    .filter(([, preset]) => matches(preset))
    .sort((a, b) => {
      const aCreate = (a[1].applies_to as string[]).includes('create');
      const bCreate = (b[1].applies_to as string[]).includes('create');
      if (aCreate !== bCreate) {
        return aCreate ? -1 : 1;
      }
      return a[0].localeCompare(b[0], 'zh');
    });

  const result: Record<string, Record<string, unknown>> = {};
  for (const [name, preset] of entries) {
    result[name] = preset;
  }
  return result;
});

const loadPresets = async () => {
  try {
    const data = await api.settings.getPresets();
    presets.value = (data as Record<string, Record<string, unknown>>) || {};
  } catch (e) {
    console.error('Failed to load presets:', e);
    presets.value = {};
  }
};

/* ------------------------------------------------------------------ */
/*  Generation logic (simplified: text-to-image + image-to-image)      */
/* ------------------------------------------------------------------ */

let activeGenStream: EventSource | null = null;
const currentTask = ref<Record<string, unknown> | null>(null);

function closeGenStream() {
  if (activeGenStream) {
    activeGenStream.close();
    activeGenStream = null;
  }
}

function assetIdsFromResultPayload(data: unknown): string[] {
  if (!data || typeof data !== 'object') return [];
  const rec = data as Record<string, unknown>;
  const nested =
    rec.result && typeof rec.result === 'object'
      ? (rec.result as Record<string, unknown>)
      : null;
  const raw = rec.asset_ids ?? nested?.asset_ids;
  if (!Array.isArray(raw)) return [];
  return raw.map((id) => String(id || '').trim()).filter(Boolean);
}

async function resolveCompletedAssetIds(taskId: string, seeded: string[]): Promise<string[]> {
  if (seeded.length > 0) return seeded;
  try {
    const task = (await api.gen.getMediaTask(taskId)) as {
      result?: { asset_ids?: unknown; primary_asset_id?: unknown };
    };
    const fromResult = assetIdsFromResultPayload(task?.result);
    if (fromResult.length) return fromResult;
    const primary = String(task?.result?.primary_asset_id || '').trim();
    return primary ? [primary] : [];
  } catch {
    return [];
  }
}

/** Ensure gallery has rows for canvas placement (group filters can miss a fresh asset). */
async function ensureGalleryItemsForCanvas(assetIds: string[]): Promise<void> {
  const missing = () =>
    assetIds.filter((id) => !galleryItems.value.some((g) => g.path === `asset:${id}`));
  if (!missing().length) return;
  // Ungrouped / group-root filters may omit the new row; pull a plain first page.
  try {
    const rows = await api.gallery.listImages(Math.max(40, assetIds.length), 0, {
      kind: 'image',
      exclude_upload_refs: true,
    });
    const byPath = new Map(galleryItems.value.map((g) => [g.path, g]));
    for (const row of rows) {
      byPath.set(row.path, row);
    }
    galleryItems.value = Array.from(byPath.values());
  } catch {
    /* staging add still attempts with whatever gallery has */
  }
  // Still missing → stub so board placement / render never depends on pagination.
  if (missing().length) {
    const byPath = new Map(galleryItems.value.map((g) => [g.path, g]));
    for (const id of missing()) {
      const path = `asset:${id}`;
      byPath.set(path, galleryStubForAssetPath(path));
    }
    galleryItems.value = Array.from(byPath.values());
  }
}

function attachStreamFromSubmit(submitRes: unknown) {
  const tid = taskIdFromSubmitResponse(submitRes);
  if (!tid) {
    generating.value = false;
    return;
  }
  closeGenStream();
  pendingCanvasAssetIds.value = [];
  generating.value = true;
  tasksStore.clearTaskLogs(tid);
  tasksStore.appendTaskLog(tid, $tt('studio.startingGen'), 'info');
  tasksStore.registerPageOwnedStream(tid);
  currentTask.value = {
    id: tid,
    progress: 0,
    step: 0,
    total: 0,
    status: 'queued',
    params: { model: params.model, title: String(params.title || '').trim(), prompt: params.prompt },
  };

  activeGenStream = api.gen.streamMediaTask(tid, {
    onLog: (logData: any) => {
      tasksStore.ingestTaskLog(tid, logData);
    },
    onTrace: (traceData: unknown) => {
      tasksStore.ingestTaskPipelineTrace(tid, traceData);
    },
    onStatus: (statusData: any) => {
      if (currentTask.value) {
        currentTask.value = { ...currentTask.value, ...statusData };
      }
    },
    onProgress: (progressData: any) => {
      tasksStore.ingestTaskProgressLog(tid, progressData);
      tasksStore.patchLiveTaskProgress(tid, {
        progress: progressData.progress,
        step: progressData.step,
        total: progressData.total,
        eta_seconds: progressData.eta_seconds,
        progressMessage: progressData.message ?? progressData.phase,
      });
      if (currentTask.value) {
        currentTask.value = {
          ...currentTask.value,
          progress: progressData.progress ?? currentTask.value.progress,
          step: progressData.step ?? currentTask.value.step,
          total: progressData.total ?? currentTask.value.total,
        };
      }
    },
    onResult: (resultData: any) => {
      const ids = assetIdsFromResultPayload(resultData);
      if (ids.length > 0) pendingCanvasAssetIds.value = ids;
    },
    onDone: async (doneData: any) => {
      generating.value = false;
      tasksStore.unregisterPageOwnedStream(tid);
      if (doneData.status === 'completed') {
        tasksStore.appendTaskLog(tid, $tt('studio.genComplete'), 'success');
        const ids = await resolveCompletedAssetIds(tid, [...pendingCanvasAssetIds.value]);
        pendingCanvasAssetIds.value = [];
        // Session-tagged results always get membership on that canvas (shelf/board),
        // even if the user already returned to the strip list.
        const willPlace = ids.length > 0;
        const willRevealCanvas = willPlace && shouldAutoAddToCanvas();
        if (!willRevealCanvas) {
          toast.success($tt('studio.genComplete'));
        }
        void (async () => {
          // Let SSE close / DB commit settle, then add with a fresh gallery catalog
          // (do not rely on InfiniteCanvas props.items — they lag loadGallery reset).
          await new Promise((r) => setTimeout(r, 400));
          await loadGallery(true);
          await nextTick();
          if (!willPlace) return;
          // Land on the session that submitted the task (user may have switched strips).
          let sid = '';
          try {
            const task = (await api.gen.getMediaTask(tid)) as {
              params?: {
                request?: { metadata?: { canvas_session_id?: string } };
                metadata?: { canvas_session_id?: string };
              };
            };
            const params = task?.params || {};
            sid = String(
              params.request?.metadata?.canvas_session_id ||
                params.metadata?.canvas_session_id ||
                '',
            ).trim();
            if (sid && sid !== unref(imageCanvas.sessionId)) {
              await imageCanvas.switchSession(sid);
            }
          } catch {
            /* keep current session */
          }
          await ensureGalleryItemsForCanvas(ids);
          await nextTick();
          if (willRevealCanvas) {
            await activateCanvasViewForResults(viewMode, syncCompositorOverlaysOnCanvasEnter);
            await nextTick();
          }
          const paths = ids.map((id) => `asset:${id}`);
          // Prefer store placement (works even when InfiniteCanvas is unmounted in grid mode).
          let added = imageCanvas.addPathsFromGallery(paths, galleryItems.value, {
            placement: 'staging',
            selectLast: true,
          });
          if (!added.length && viewMode.value === 'canvas') {
            added = addAssetPathsToCanvas(paths, { placement: 'staging', focusLast: true, fit: true });
          }
          if (!added.length) {
            await ensureGalleryItemsForCanvas(ids);
            await nextTick();
            added = imageCanvas.addPathsFromGallery(paths, galleryItems.value, {
              placement: 'staging',
              selectLast: true,
            });
          }
          if (added.length) {
            imageCanvas.refreshEdges(galleryItems.value);
            await imageCanvas.persist();
            await nextTick();
            if (viewMode.value === 'canvas') {
              infiniteCanvasRef.value?.fitAll?.();
            }
          } else {
            // Fail loud: tagged session must receive the asset; last resort shelf.
            if (sid || unref(imageCanvas.sessionId)) {
              imageCanvas.addToShelf(paths);
              await imageCanvas.persist();
              toast.warning($tt('canvas.autoAddMissed'));
            } else {
              toast.warning($tt('canvas.autoAddMissed'));
            }
          }
        })();
      } else if (doneData.status === 'failed') {
        const updated = await api.gen.getMediaTask(tid) as any;
        tasksStore.appendTaskLog(
          tid,
          $tt('studio.genFailed', { msg: updated.error || updated.error_message || '' }),
          'error'
        );
        toast.error($tt('studio.genFailed', { msg: updated.error || updated.error_message || '' }));
      }
      currentTask.value = null;
      closeGenStream();
    },
    onError: () => {
      generating.value = false;
      tasksStore.unregisterPageOwnedStream(tid);
      tasksStore.appendTaskLog(tid, $tt('studio.connectionLost'), 'warning');
      currentTask.value = null;
      closeGenStream();
    },
  });
}

type GenerationPrepared = {
  modelStr: string;
  adapters: Array<{ id: string; weight: number }>;
  meta: Record<string, unknown>;
  control_asset_id: string | null;
  source_asset_id: string | null;
  hasRef: boolean;
};

function validateGenerationPreconditions(): boolean {
  const detailed = modelsDetailedStatus.value[params.model as string];
  const versionStatus = detailed?.versions?.[params.version as string];
  if (!versionStatus?.ready) {
    toast.warning($tt('studio.modelNotReadyDesc', { name: currentModelDisplayName.value, version: params.version as string }));
    return false;
  }

  const verCfg =
    (currentModelConfig.value &&
      currentModelConfig.value.versions &&
      (currentModelConfig.value.versions as Record<string, Record<string, unknown>>)[params.version as string]) ||
    null;
  const sizeHuman = verCfg && verCfg.size ? String(verCfg.size) : '';
  const minMemRaw = currentModelConfig.value?.parameters?.min_unified_memory_gb;
  const minUnifiedMemoryGb = minMemRaw != null && Number(minMemRaw) > 0 ? Number(minMemRaw) : null;
  warnIfRiskyMemory({
    systemInfo: unref(systemInfo),
    versionSizeHuman: sizeHuman,
    minUnifiedMemoryGb,
    $tt,
  });

  if (versionDisallowsReferences() && referenceImage.value) {
    toast.warning($tt('create.quantizedNoReference'));
    return false;
  }

  const guideValidation = structuralGuideValidation();
  if (!guideValidation.ok) {
    const guideToastKeys = {
      fill_edit_only: 'studio.controlnetFillEditOnly',
      flux_only: 'studio.controlnetFluxOnly',
      no_img2img: 'studio.controlnetNoImg2img',
      missing_control_image: 'canvas.controlImageRequired',
      controlnet_not_ready: 'studio.controlnetNotReady',
      runtime_unavailable: 'studio.controlnetMlxOnly',
    };
    const toastKey = guideToastKeys[guideValidation.code];
    if (toastKey) toast.warning($tt(toastKey));
    return false;
  }

  return true;
}

async function prepareGenerationContext(): Promise<GenerationPrepared | null> {
  if (!validateGenerationPreconditions()) return null;

  const modelStr = params.version ? `${params.model}:${params.version}` : params.model;
  // Primary LoRA only. Companion adapters come from the LoRA's inference metadata
  // on the server (bind_installed_loras → inference_stack) — keep the client neutral.
  const adapters: Array<{ id: string; weight: number }> = [];
  if (params.lora) adapters.push({ id: String(params.lora), weight: Number(params.lora_scale) || 0.8 });
  persistComposerSnapshot();
  const meta = buildCanvasMeta();
  const scheduler = schedulerForModel(
    currentModelConfig.value?.parameters as Record<string, unknown> | undefined,
    params.scheduler,
  );
  if (scheduler) meta.scheduler = scheduler;

  let control_asset_id: string | null = null;
  if (params.controlnet) {
    control_asset_id = resolveControlAssetId(controlImage.value, {
      assetRequired: $tt('canvas.controlImageAssetRequired'),
      required: $tt('canvas.controlImageRequired'),
    });
  }

  const hasRef = referenceImage.value != null;
  let source_asset_id: string | null = null;
  if (hasRef) {
    const rp = referenceImage.value!.path;
    if (typeof rp === 'string' && rp.startsWith('asset:')) {
      source_asset_id = rp.slice('asset:'.length);
    } else {
      const blob = await api.gen.urlToBlob(referenceImage.value!.previewUrl);
      const up = await api.gen.uploadAsset(
        new File([blob], 'ref.png', { type: blob.type || 'image/png' })
      );
      source_asset_id = (up as any).id;
    }
  }

  return {
    modelStr,
    adapters,
    meta,
    control_asset_id,
    source_asset_id,
    hasRef,
  };
}

function collectExtraReferenceAssetIds(sourceAssetId: string | null): string[] {
  if (versionDisallowsReferences()) return [];
  const cap = extraReferenceSlotCap(modelReferenceParameters());
  if (cap <= 0) return [];
  const ids: string[] = [];
  const pushId = (id: string | null | undefined) => {
    const value = String(id || '').trim();
    if (!value || value === sourceAssetId || ids.includes(value)) return;
    ids.push(value);
  };
  for (const img of extraReferenceImages.value) {
    pushId(img.assetId || (img.path.startsWith('asset:') ? img.path.slice('asset:'.length) : ''));
  }
  return ids.slice(0, cap);
}

function resolveSeedForTask(baseSeed: number | null, batchIndex: number): number | null {
  if (baseSeed == null) return null;
  return baseSeed + batchIndex;
}

async function submitImageGenerationTask(
  ctx: GenerationPrepared,
  opts: { prompt: string; title: string; seed: number | null; n: number },
): Promise<unknown> {
  const enhCommon = {
    parameters: currentModelConfig.value?.parameters as Record<string, unknown> | undefined,
    params,
    controlnet: String(params.controlnet || ''),
    controlAssetId: ctx.control_asset_id,
    controlnetStrength: Number(params.controlnet_strength) || 0.8,
    compatibleControlNets: compatibleControlNets.value,
    registry: modelRegistry.value,
  };

  if (ctx.hasRef) {
    const editBody: Record<string, unknown> = {
      model: ctx.modelStr,
      operation: 'rewrite',
      source_asset_id: ctx.source_asset_id,
      title: opts.title,
      prompt: opts.prompt,
      negative_prompt: params.negative_prompt || '',
      n: 1,
      steps: params.steps,
      guidance: params.guidance,
      seed: opts.seed,
      scheduler: schedulerForModel(
        currentModelConfig.value?.parameters as Record<string, unknown> | undefined,
        params.scheduler,
      ),
      scheduler_shift: params.scheduler_shift,
      use_empirical_mu: params.use_empirical_mu === true,
      adapters: ctx.adapters,
      source_fidelity: strengthToSourceFidelity(
        params.strength,
        strengthDefaultFromRegistry(currentModelConfig.value?.parameters as Record<string, unknown> | undefined),
      ),
      size: `${params.width}x${params.height}`,
      metadata: { ...ctx.meta },
      priority: 'normal',
    };
    const editEnh = appendZImageEnhancementFields(editBody, enhCommon);
    if (!editEnh.ok) {
      throw new Error($tt('create.inpaintPairRequired'));
    }
    const extraRefs = collectExtraReferenceAssetIds(ctx.source_asset_id);
    if (extraRefs.length) {
      editBody.reference_asset_ids = extraRefs;
    }
    return api.gen.createImageEdit(editBody);
  }

  const genBody: Record<string, unknown> = {
    model: ctx.modelStr,
    title: opts.title,
    prompt: opts.prompt,
    negative_prompt: params.negative_prompt || '',
    size: `${params.width}x${params.height}`,
    n: opts.n,
    steps: params.steps,
    guidance: params.guidance,
    seed: opts.seed,
    scheduler: schedulerForModel(
      currentModelConfig.value?.parameters as Record<string, unknown> | undefined,
      params.scheduler,
    ),
    scheduler_shift: params.scheduler_shift,
    use_empirical_mu: params.use_empirical_mu === true,
    adapters: ctx.adapters,
    metadata: { ...ctx.meta },
    priority: 'normal',
  };
  const genEnh = appendZImageEnhancementFields(genBody, enhCommon);
  if (!genEnh.ok) {
    throw new Error($tt('create.inpaintPairRequired'));
  }
  return api.gen.createImageGeneration(genBody);
}

async function queuePromptsToServer() {
  const lines = splitComposerPromptLines(params.prompt);
  if (lines.length === 0) {
    toast.warning($tt('studio.enterPrompt'));
    return;
  }
  if (queueSubmitting.value) return;

  queueSubmitting.value = true;
  const baseSeed = seedForSubmit(params.seed);
  const title = String(params.title || '').trim();

  try {
    const ctx = await prepareGenerationContext();
    if (!ctx) return;

    for (let i = 0; i < lines.length; i += 1) {
      await submitImageGenerationTask(ctx, {
        prompt: lines[i],
        title,
        seed: resolveSeedForTask(baseSeed, i),
        n: 1,
      });
    }

    params.prompt = '';
    toast.success(
      lines.length > 1
        ? $tt('create.batchSubmitted', { count: lines.length })
        : $tt('assistant.taskQueued'),
    );
    tasksStore.pollQueueOnce();
  } catch (e) {
    toast.error($tt('studio.error', { msg: (e as Error).message || String(e) }));
  } finally {
    queueSubmitting.value = false;
  }
}

async function onComposerSubmit() {
  if (!String(params.prompt || '').trim()) {
    toast.warning($tt('studio.enterPrompt'));
    return;
  }
  if (composerBusy.value) {
    await queuePromptsToServer();
    return;
  }
  await startGeneration();
}

const startGeneration = async () => {
  if (generating.value) return;
  if (!String(params.prompt || '').trim()) {
    toast.warning($tt('studio.enterPrompt'));
    return;
  }

  generating.value = true;

  try {
    const ctx = await prepareGenerationContext();
    if (!ctx) {
      generating.value = false;
      return;
    }

    const seedNum = seedForSubmit(params.seed);
    const submitRes = await submitImageGenerationTask(ctx, {
      prompt: params.prompt,
      title: String(params.title || '').trim(),
      seed: seedNum,
      n: batchCount.value,
    });

    attachStreamFromSubmit(submitRes);
    tasksStore.pollQueueOnce();
  } catch (e) {
    generating.value = false;
    closeGenStream();
    currentTask.value = null;
    toast.error($tt('studio.error', { msg: (e as Error).message || String(e) }));
  }
};

/* ------------------------------------------------------------------ */
/*  Load model registry                                                */
/* ------------------------------------------------------------------ */

const loadModelRegistry = async () => {
  try {
    const RS = registryStore;
    const regPromise = RS && RS.load
      ? RS.load()
      : api.settings.getModelRegistry().then((r: Record<string, unknown>) => ({ models: (r as Record<string, unknown>).models }));
    const [registryData, statusData, detailedStatusData] = await Promise.all([
      regPromise,
      api.settings.getModelsStatus(),
      api.settings.getModelsDetailedStatus(),
    ]);

    modelRegistry.value = (registryData as Record<string, unknown>).models || {};
    modelsDetailedStatus.value = (detailedStatusData as any) || {};

    const picker = filteredModelPickerVersions.value;
    const inPicker = (key: string) =>
      picker.some((v) => `${v.modelKey}|${v.versionKey}` === key);
    if (!selectedModelVersion.value || !inPicker(selectedModelVersion.value)) {
      const lastModel = getItem(DQ_STORAGE.IMAGE_LAST_MODEL);
      if (lastModel && inPicker(lastModel)) {
        const parsed = parseModelVersionValue(lastModel);
        if (parsed) {
          params.model = parsed.modelKey;
          params.version = parsed.versionKey;
          selectedModelVersion.value = lastModel;
        }
      } else if (picker.length > 0) {
        const first = picker[0];
        params.model = first.modelKey;
        params.version = first.versionKey;
        selectedModelVersion.value = `${first.modelKey}|${first.versionKey}`;
      }
    }

    loadModelDefaults();
    restoreSavedSize();
    if (params.model) loadCompatibleAdapters(String(params.model));
  } catch (e) {
    console.error('Failed to load model registry:', e);
  }
};

const loadModelDefaults = () => {
  if (verificationRecipeActive.value) return;
  const config = currentModelConfig.value;
  if (!config || !config.parameters) return;
  applyDefaults(config.parameters as Record<string, unknown>, params);
};

function restoreSavedSize() {
  syncResolutionForModel(String(params.model || ''));
}

const resetToDefaults = () => {
  loadModelDefaults();
  syncResolutionForModel(String(params.model || ''));
  syncImageLoraSelection();
  toast.success($tt('studio.restoredDefaults'));
};

/* ------------------------------------------------------------------ */
/*  Gallery interactions                                               */
/* ------------------------------------------------------------------ */

const previewVisible = ref(false);
const selectedImageIndex = ref(0);

watch(previewVisible, (open) => {
  if (!open) {
    stripPreviewOverride.value = null;
    previewSourceSessionId.value = '';
  }
});

function getImageUrl(item: GalleryItem) {
  return api.gallery.getImageUrl(item.path);
}

function onGallerySelect(item: GalleryItem) {
  previewSourceSessionId.value = canvasSessionId.value || '';
  const idx = galleryItems.value.findIndex((it) => it.path === item.path);
  selectedImageIndex.value = idx >= 0 ? idx : 0;
  previewVisible.value = true;
}

/* ------------------------------------------------------------------ */
/*  Card actions: retouch / extend / upscale / download / delete       */
/* ------------------------------------------------------------------ */

const upscaleParams = reactive({
  upscale_scale: 2,
  upscale_denoise: 0.3,
  upscale_tile: 1024,
});
const upscaleModelVersion = ref('');
const upscaleModelOptions = computed(() => {
  return allVersions.value
    .filter((v) => {
      const acts = (v.actions as Record<string, unknown>) || {};
      return imageSupportsUpscale(acts);
    })
    .map((v) => modelSelectOption(v));
});
const extendModelVersion = ref('');
const extendModelOptions = computed(() => {
  const rows = allVersions.value
    .filter((v) => {
      const acts = (v.actions as Record<string, unknown>) || {};
      if (!imageSupportsExtend(acts)) return false;
      const row = modelRegistry.value[String(v.modelKey || '')];
      return isStandaloneEditControlNet(row);
    })
    .map((v) => ({
      ...modelSelectOption(v, { unreadySuffix: $tt('common.notReady') }),
      isStandalone: true,
    }));
  rows.sort((a, b) => {
    if (a.disabled !== b.disabled) return a.disabled ? 1 : -1;
    return a.label.localeCompare(b.label);
  });
  return rows;
});

const fillEditParams = reactive({
  steps: 28,
  guidance: 30,
});

function applyFillEditDefaults(modelVersion: string) {
  const [modelKey] = String(modelVersion || '').split('|');
  if (!modelKey) return;
  const row = modelRegistry.value[modelKey];
  if (isStandaloneEditControlNet(row)) {
    const patch = fillModelRegistryDefaultsPatch(modelKey, modelRegistry.value);
    if (patch.guidance != null) fillEditParams.guidance = patch.guidance;
    if (patch.steps != null) fillEditParams.steps = patch.steps;
    return;
  }
  if (isStructuralRetouchControlNet(row)) {
    const raw =
      (row.parameters as Record<string, { default?: number }> | undefined) ||
      (row.ui as { parameters?: Record<string, { default?: number }> } | undefined)?.parameters;
    if (typeof raw?.guidance?.default === 'number') fillEditParams.guidance = raw.guidance.default;
    if (typeof raw?.steps?.default === 'number') fillEditParams.steps = raw.steps.default;
  }
}

function retouchProviderKind(modelKey: string): 'standalone' | 'companion' | '' {
  const row = modelRegistry.value[modelKey];
  if (isStandaloneEditControlNet(row)) return 'standalone';
  if (isStructuralRetouchControlNet(row)) return 'companion';
  return '';
}

const retouchHintText = computed(() => {
  const [modelKey] = String(retouchModelVersion.value || '').split('|');
  const kind = retouchProviderKind(modelKey || '');
  if (kind === 'companion') return $tt('studio.retouchCompanionHint');
  if (kind === 'standalone') return $tt('studio.retouchStandaloneHint');
  return $tt('studio.retouchSelectHint');
});

function defaultBaseVersionForControlNet(controlnetKey: string): string {
  const row = modelRegistry.value[controlnetKey];
  const base = String(row?.base_model || '').trim();
  if (!base) return '';
  const candidates = allVersions.value.filter((v) => String(v.modelKey) === base);
  const ready = candidates.find((v) => v.ready);
  const pick = ready || candidates[0];
  return pick ? `${pick.modelKey}|${pick.versionKey}` : '';
}

function retouchSizeFromItem(item: GalleryItem): string {
  const w = Number(item.width) || 0;
  const h = Number(item.height) || 0;
  if (w > 0 && h > 0) return `${w}x${h}`;
  return `${params.width}x${params.height}`;
}

function isEditModelVersionReady(modelVersion: string): boolean {
  const [modelKey, versionKey] = String(modelVersion || '').split('|');
  if (!modelKey || !versionKey) return false;
  const detailed = modelsDetailedStatus.value[modelKey];
  return Boolean(detailed?.versions?.[versionKey]?.ready);
}

const extendParams = reactive({
  extend_directions: ['right'],
  extend_pixels: 256,
});

watch(
  () => [extendParams.extend_directions, extendParams.extend_pixels],
  () => persistComposerSnapshot(),
  { deep: true }
);

const retouchModelVersion = ref('');
const retouchModelOptions = computed(() => {
  const rows = allVersions.value
    .filter((v) => {
      const acts = (v.actions as Record<string, unknown>) || {};
      if (!imageSupportsRetouch(acts)) return false;
      const row = modelRegistry.value[String(v.modelKey || '')];
      return (
        isStandaloneEditControlNet(row) || isStructuralRetouchControlNet(row)
      );
    })
    .map((v) => {
      const row = modelRegistry.value[String(v.modelKey || '')];
      return {
        ...modelSelectOption(v, { unreadySuffix: $tt('common.notReady') }),
        modelKey: String(v.modelKey || ''),
        isStandalone: isStandaloneEditControlNet(row),
      };
    });
  rows.sort((a, b) => {
    if (a.isStandalone !== b.isStandalone) return a.isStandalone ? -1 : 1;
    if (a.disabled !== b.disabled) return a.disabled ? 1 : -1;
    return a.label.localeCompare(b.label);
  });
  return rows;
});
const imageEditorRef = ref<any>(null);

watch(retouchModelVersion, (value, prev) => {
  if (!value || value === prev) return;
  applyFillEditDefaults(value);
});

watch(extendModelVersion, (value, prev) => {
  if (!value || value === prev) return;
  applyFillEditDefaults(value);
});

watch(
  () => [
    retouchModelVersion.value,
    extendModelVersion.value,
    upscaleModelVersion.value,
    upscaleParams.upscale_scale,
    upscaleParams.upscale_denoise,
    fillEditParams.steps,
    fillEditParams.guidance,
    editorMode.value,
    editDrawerItem.value?.path,
    showEditorDrawer.value,
  ],
  () => persistComposerSnapshot()
);

function onCardAction({ action, item }: { action: string; item: GalleryItem }) {
  switch (action) {
    case 'compose-from-item':
      fillComposerFromGalleryItem(item);
      if (item.path?.startsWith('asset:')) {
        const previewUrl = previewUrlForGalleryItem(item);
        referenceImage.value = { path: item.path, previewUrl };
        imageMode.value = 'img2img';
      }
      openComposerDrawer();
      break;
    case 'retouch':
    case 'extend':
    case 'upscale':
      // 保存当前 composer 模型到对应 imageMode key
      if (selectedModelVersion.value) {
        setItem(getImageModeStorageKey(imageMode.value), selectedModelVersion.value);
      }
      editDrawerItem.value = item;
      editorMode.value = action as 'retouch' | 'extend' | 'upscale';
      // 恢复该 action 上次使用的模型
      restoreModelForMode(action);
      // 同步 drawer 内模型选择器
      if (action === 'upscale') {
        const currentKey = params.model && params.version ? `${params.model}|${params.version}` : '';
        const currentSupportsUpscale = allVersions.value.some((v) => {
          const acts = (v.actions as Record<string, unknown>) || {};
          return `${v.modelKey}|${v.versionKey}` === currentKey && imageSupportsUpscale(acts);
        });
        if (currentSupportsUpscale) {
          upscaleModelVersion.value = currentKey;
        } else {
          const first = upscaleModelOptions.value.find((o) => !o.disabled);
          upscaleModelVersion.value = first ? first.value : '';
        }
      } else if (action === 'extend') {
        const currentKey = params.model && params.version ? `${params.model}|${params.version}` : '';
        const currentSupportsExtend = allVersions.value.some((v) => {
          const acts = (v.actions as Record<string, unknown>) || {};
          return `${v.modelKey}|${v.versionKey}` === currentKey && imageSupportsExtend(acts);
        });
        if (
          currentSupportsExtend &&
          isStandaloneEditControlNet(modelRegistry.value[currentKey.split('|')[0] || ''])
        ) {
          extendModelVersion.value = currentKey;
        } else {
          const first = extendModelOptions.value.find((o) => !o.disabled);
          extendModelVersion.value = first ? first.value : '';
        }
        if (extendModelVersion.value) applyFillEditDefaults(extendModelVersion.value);
      } else if (action === 'retouch') {
        const currentKey = params.model && params.version ? `${params.model}|${params.version}` : '';
        const currentModelKey = currentKey.split('|')[0] || '';
        const currentSupportsRetouch = allVersions.value.some((v) => {
          const acts = (v.actions as Record<string, unknown>) || {};
          return `${v.modelKey}|${v.versionKey}` === currentKey && imageSupportsRetouch(acts);
        });
        const currentKind = retouchProviderKind(currentModelKey);
        if (currentSupportsRetouch && currentKind) {
          retouchModelVersion.value = currentKey;
        } else {
          // Prefer companion matching the create base, else first ready option.
          const baseKey = String(params.model || '');
          const companionForBase = retouchModelOptions.value.find((o) => {
            if (o.disabled || o.isStandalone) return false;
            const row = modelRegistry.value[o.modelKey];
            return String(row?.base_model || '') === baseKey;
          });
          const first = companionForBase || retouchModelOptions.value.find((o) => !o.disabled);
          retouchModelVersion.value = first ? first.value : '';
        }
        if (retouchModelVersion.value) applyFillEditDefaults(retouchModelVersion.value);
      }
      if (viewMode.value === 'canvas') {
        imageCanvas.placeStagingBeside(item.path, galleryItems.value);
      }
      showEditorDrawer.value = true;
      break;
    case 'add-to-canvas':
      void addGalleryItemToCanvas(item, { fit: true });
      break;
    case 'download':
      downloadItem(item);
      break;
    case 'delete':
      deleteItem(item);
      break;
    case 'lineage':
      lineageTargetAssetId.value = item.path.startsWith('asset:')
        ? item.path.slice('asset:'.length)
        : '';
      showLineageDrawer.value = true;
      break;
  }
}

function downloadItem(item: GalleryItem) {
  const url = getImageUrl(item);
  const a = document.createElement('a');
  a.href = url;
  a.download = item.name;
  a.click();
  toast.success($tt('gallery.startDownload'));
}

async function onEditorSubmit() {
  if (!editDrawerItem.value || !imageEditorRef.value) return;
  if (!controlNetHostRuntimeAvailable.value) {
    toast.warning($tt('studio.controlnetMlxOnly'));
    return;
  }
  if (!validateFillEditPrompt(String(params.prompt || ''))) {
    toast.warning($tt('studio.enterPrompt'));
    return;
  }
  try {
    const maskBlob = await imageEditorRef.value.getMaskBlob();
    if (!maskBlob) {
      toast.warning($tt('studio.drawMask'));
      return;
    }

    const path = editDrawerItem.value.path;
    let source_asset_id: string;
    if (path.startsWith('asset:')) {
      source_asset_id = path.slice('asset:'.length);
    } else {
      const blob = await api.gen.urlToBlob(getImageUrl(editDrawerItem.value));
      const up = await api.gen.uploadAsset(
        new File([blob], 'source.png', { type: blob.type || 'image/png' })
      );
      source_asset_id = (up as any).id;
    }

    const mask_asset_id = (
      await api.gen.uploadAsset(new File([maskBlob], 'mask.png', { type: 'image/png' }), {
        ephemeral: true,
        purpose: 'inpaint_mask',
      })
    ).id as string;

    if (!retouchModelVersion.value) {
      toast.warning($tt('studio.selectModel'));
      return;
    }
    const [modelKey, versionKey] = retouchModelVersion.value.split('|');
    const kind = retouchProviderKind(modelKey);
    if (!kind) {
      toast.warning($tt('studio.retouchProviderRequired'));
      return;
    }
    if (!isEditModelVersionReady(retouchModelVersion.value)) {
      toast.warning($tt('studio.modelNotReadyDesc', { name: modelKey, version: versionKey }));
      return;
    }
    const seedNum = seedForSubmit(params.seed);
    const adapters: Array<{ id: string; weight: number }> = [];
    if (params.lora) adapters.push({ id: String(params.lora), weight: Number(params.lora_scale) || 0.8 });

    let submitRes: unknown;
    if (kind === 'standalone') {
      const modelStr = versionKey ? `${modelKey}:${versionKey}` : modelKey;
      submitRes = await api.gen.createImageEdit({
        model: modelStr,
        operation: 'retouch',
        source_asset_id,
        mask_asset_id,
        title: String(params.title || '').trim(),
        prompt: String(params.prompt || '').trim(),
        negative_prompt: params.negative_prompt || '',
        n: 1,
        steps: fillEditParams.steps,
        guidance: fillEditParams.guidance,
        seed: seedNum,
        adapters,
        metadata: buildCanvasMeta(),
        priority: 'normal',
      });
    } else {
      const basePick = defaultBaseVersionForControlNet(modelKey);
      if (!basePick) {
        toast.warning($tt('studio.retouchBaseMissing'));
        return;
      }
      const [baseKey, baseVersion] = basePick.split('|');
      if (!isEditModelVersionReady(basePick)) {
        toast.warning($tt('studio.modelNotReadyDesc', { name: baseKey, version: baseVersion }));
        return;
      }
      const row = modelRegistry.value[modelKey];
      const paramBlock =
        (row?.parameters as Record<string, { default?: number }> | undefined) ||
        (
          row?.ui as { parameters?: Record<string, { default?: number }> } | undefined
        )?.parameters;
      const strengthRaw = paramBlock?.controlnet_strength?.default;
      // Registry Union default is 0.9; never silently fall back to a weaker scale.
      const weight = typeof strengthRaw === 'number' ? strengthRaw : 0.9;
      const genBody: Record<string, unknown> = {
        model: baseVersion ? `${baseKey}:${baseVersion}` : baseKey,
        title: String(params.title || '').trim(),
        prompt: String(params.prompt || '').trim(),
        negative_prompt: params.negative_prompt || '',
        size: retouchSizeFromItem(editDrawerItem.value),
        n: 1,
        steps: fillEditParams.steps,
        guidance: fillEditParams.guidance,
        seed: seedNum,
        adapters,
        metadata: buildCanvasMeta(),
        priority: 'normal',
        structural_guide: buildStructuralGuidePayload(
          modelKey,
          source_asset_id,
          weight,
          { sourceAssetId: source_asset_id, maskAssetId: mask_asset_id },
          { registry: modelRegistry.value, versionKey },
        ),
      };
      submitRes = await api.gen.createImageGeneration(genBody);
    }
    attachStreamFromSubmit(submitRes);
    showEditorDrawer.value = false;
    tasksStore.pollQueueOnce();
  } catch (e) {
    toast.error($tt('studio.error', { msg: (e as Error).message || String(e) }));
  }
}

async function onExtendSubmit() {
  if (!editDrawerItem.value) return;
  if (!controlNetHostRuntimeAvailable.value) {
    toast.warning($tt('studio.controlnetMlxOnly'));
    return;
  }
  if (!validateFillEditPrompt(String(params.prompt || ''))) {
    toast.warning($tt('studio.enterPrompt'));
    return;
  }
  try {
    const dirs = Array.isArray(extendParams.extend_directions)
      ? extendParams.extend_directions.filter((d: string) => ['top', 'bottom', 'left', 'right'].includes(d))
      : [];
    if (!dirs.length) {
      toast.warning($tt('create.extendNeedDirection'));
      return;
    }
    const px = Math.min(2048, Math.max(64, Number(extendParams.extend_pixels) || 256));

    const path = editDrawerItem.value.path;
    let source_asset_id: string;
    if (path.startsWith('asset:')) {
      source_asset_id = path.slice('asset:'.length);
    } else {
      const blob = await api.gen.urlToBlob(getImageUrl(editDrawerItem.value));
      const up = await api.gen.uploadAsset(
        new File([blob], 'source.png', { type: blob.type || 'image/png' })
      );
      source_asset_id = (up as any).id;
    }

    if (!extendModelVersion.value) {
      toast.warning($tt('studio.selectModel'));
      return;
    }
    const [modelKey, versionKey] = extendModelVersion.value.split('|');
    if (!isStandaloneEditControlNet(modelRegistry.value[modelKey])) {
      toast.warning($tt('studio.extendStandaloneRequired'));
      return;
    }
    if (!isEditModelVersionReady(extendModelVersion.value)) {
      toast.warning($tt('studio.modelNotReadyDesc', { name: modelKey, version: versionKey }));
      return;
    }
    const modelStr = versionKey ? `${modelKey}:${versionKey}` : modelKey;
    const seedNum = seedForSubmit(params.seed);
    const adapters: Array<{ id: string; weight: number }> = [];
    if (params.lora) adapters.push({ id: String(params.lora), weight: Number(params.lora_scale) || 0.8 });

    const submitRes = await api.gen.createImageEdit({
      model: modelStr,
      operation: 'extend',
      source_asset_id,
      title: String(params.title || '').trim(),
      prompt: String(params.prompt || '').trim(),
      negative_prompt: params.negative_prompt || '',
      extend: { directions: dirs, pixels: px },
      n: 1,
      steps: fillEditParams.steps,
      guidance: fillEditParams.guidance,
      seed: seedNum,
      adapters,
      metadata: buildCanvasMeta(),
      priority: 'normal',
    });
    attachStreamFromSubmit(submitRes);
    showEditorDrawer.value = false;
    tasksStore.pollQueueOnce();
  } catch (e) {
    toast.error($tt('studio.error', { msg: (e as Error).message || String(e) }));
  }
}

async function onUpscaleSubmit() {
  if (!editDrawerItem.value) return;
  if (!upscaleModelVersion.value) {
    toast.warning($tt('studio.selectModel'));
    return;
  }
  try {
    const path = editDrawerItem.value.path;
    let source_asset_id: string;
    if (path.startsWith('asset:')) {
      source_asset_id = path.slice('asset:'.length);
    } else {
      const blob = await api.gen.urlToBlob(getImageUrl(editDrawerItem.value));
      const up = await api.gen.uploadAsset(
        new File([blob], 'upscale.png', { type: blob.type || 'image/png' })
      );
      source_asset_id = (up as any).id;
    }

    const [modelKey, versionKey] = upscaleModelVersion.value.split('|');
    const modelStr = versionKey ? `${modelKey}:${versionKey}` : modelKey;
    const submitRes = await api.gen.createImageUpscale({
      model: modelStr,
      source_asset_id,
      scale: upscaleParams.upscale_scale,
      denoise: upscaleParams.upscale_denoise,
      tile_size: upscaleParams.upscale_tile,
      metadata: buildCanvasMeta(),
      priority: 'normal',
    });
    attachStreamFromSubmit(submitRes);
    showEditorDrawer.value = false;
    tasksStore.pollQueueOnce();
  } catch (e) {
    toast.error($tt('studio.error', { msg: (e as Error).message || String(e) }));
  }
}

/* ------------------------------------------------------------------ */
/*  Recent images (for asset picker)                                   */
/* ------------------------------------------------------------------ */

const recentImages = ref<Array<Record<string, unknown>>>([]);
const recentGalleryThumbFailed = ref<Record<string, boolean>>({});

const loadRecentImages = async () => {
  try {
    const images = await api.gallery.listImages(24, 0);
    recentGalleryThumbFailed.value = {};
    recentImages.value = (images as Array<Record<string, unknown>>)
      .filter((v: Record<string, unknown>) => {
        const meta = v.metadata as Record<string, unknown> | undefined;
        if (meta?.asset_kind === 'video' || meta?.asset_kind === 'audio') return false;
        const ext = String(v.name || '').split('.').pop()?.toLowerCase() || '';
        return !['mp4', 'mov', 'avi', 'mkv', 'webm', 'wav', 'mp3', 'flac', 'm4a', 'aac', 'opus', 'ogg'].includes(ext);
      })
      .slice(0, 4);
  } catch (e) {
    console.error('Failed to load recent images:', e);
  }
};

/* ------------------------------------------------------------------ */
/*  App settings defaults                                              */
/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/*  Lifecycle                                                          */
/* ------------------------------------------------------------------ */

onMounted(async () => {
  if (route.query.verify === '1') verificationRecipeActive.value = true;
  await loadModelRegistry();
  loadPresets();
  loadRecentImages();
  loadGallery(true);
  try {
    const adopt = await imageCanvas.ensureDefaultAndAdoptOrphans();
    if (adopt.adopted_count > 0) {
      toast.success($tt('canvas.orphansAdopted', { n: adopt.adopted_count }));
      await loadGallery(true);
    }
    await rebuildSessionAssetIndex();
  } catch (e) {
    toast.error($tt('studio.error', { msg: String(e) }));
  }
  tasksStore.ensureQueuePoller();
  window.addEventListener('keydown', onCreatePageKeydown);
  const promptDraft = consumePromptDraft(DQ_STORAGE.IMAGE_CREATE_PROMPT_DRAFT);
  if (promptDraft) {
    params.prompt = applyPromptDraft(params.prompt, promptDraft);
  }
  const q = route.query;
  if (typeof q.model === 'string' && q.model) params.model = q.model;
  if (typeof q.prompt === 'string' && q.prompt) params.prompt = q.prompt;
  if (typeof q.lora === 'string' && q.lora) params.lora = q.lora;
  if (q.verify === '1') {
    // Apply exported compose params; companion stack stays on the LoRA metadata
    // and is expanded server-side. Do not inject recipe-specific adapters here.
    verificationRecipeActive.value = true;
    const numberQuery = (key: string): number | undefined => {
      const value = Number(q[key]);
      return q[key] != null && q[key] !== '' && Number.isFinite(value) ? value : undefined;
    };
    for (const key of ['steps', 'guidance', 'lora_scale', 'width', 'height'] as const) {
      const value = numberQuery(key);
      if (value != null) params[key] = value;
    }
    if (typeof q.seed === 'string') params.seed = q.seed;
    if (typeof q.scheduler === 'string' && q.scheduler) params.scheduler = q.scheduler;
    if (typeof q.version === 'string' && q.version) params.version = q.version;
    const modelKey = String(params.model || '');
    const readyVersions = filteredModelPickerVersions.value.filter(
      (row) => row.modelKey === modelKey && row.ready,
    );
    const version = readyVersions.find((row) => row.versionKey === params.version) || readyVersions[0];
    if (version) {
      params.model = version.modelKey;
      params.version = version.versionKey;
      selectedModelVersion.value = `${version.modelKey}|${version.versionKey}`;
    }
    selectedSize.value = `${params.width}x${params.height}`;
    await loadCompatibleAdapters(modelKey);
    // Prefer LoRA compose_overrides when present; query values re-applied for seed/prompt handoff.
    syncImageLoraSelection();
    for (const key of ['steps', 'guidance', 'lora_scale', 'width', 'height'] as const) {
      const value = numberQuery(key);
      if (value != null) params[key] = value;
    }
    if (typeof q.seed === 'string') params.seed = q.seed;
    if (typeof q.scheduler === 'string' && q.scheduler) params.scheduler = q.scheduler;
    if (typeof q.lora === 'string' && q.lora) params.lora = q.lora;
    selectedSize.value = `${params.width}x${params.height}`;
    await nextTick();
    syncImageLoraSelection();
  }
});

onUnmounted(() => {
  window.removeEventListener('keydown', onCreatePageKeydown);
  closeGenStream();
  tasksStore.releaseQueuePoller();
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
.studio-dialog-center {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
}

.studio-dialog-img {
  max-width: 100%;
  max-height: 70vh;
  border-radius: 8px;
  object-fit: contain;
}

.studio-editor-drawer {
  height: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.studio-extend-panel,
.studio-upscale-panel,
.studio-retouch-panel {
  display: flex;
  flex-direction: column;
  gap: 16px;
  flex: 1;
  min-height: 0;
  overflow: hidden;
  margin-top: 0;
  padding-top: 0;
  border-top: none;
}

.studio-retouch-fill-hint {
  margin: 0 16px 8px;
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
  color: var(--dq-label-tertiary);
}

.studio-retouch-editor-wrap {
  flex: 1;
  min-height: 0;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.studio-drawer-submit {
  margin-top: auto;
}

.image-create-filters {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  min-width: 0;
}

.image-create-filters--list {
  width: 100%;
  max-width: none;
}

.image-create-filters__search {
  flex: 0 1 320px;
  min-width: 160px;
  max-width: 420px;
}

.image-create-filters__spacer {
  flex: 1 1 auto;
  min-width: 8px;
}

.image-create-filters__actions {
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  gap: 8px;
  margin-left: auto;
}

.image-create-filters__session-title {
  font-size: var(--dq-font-size-body);
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 40vw;
}
</style>

<style>
.studio-image-editor-drawer .dq-drawer-body {
  display: flex;
  flex-direction: column;
  padding: 16px 18px 20px;
  overflow: hidden;
}

/* Drawer 内统一分组面板：模型选择 + 参数表单合并为一个视觉卡片 */
.studio-image-editor-drawer .studio-editor-drawer-pref-pane.dq-pref-pane {
  border: 0.5px solid var(--dq-glass-border);
  border-radius: var(--dq-radius-group);
  background: var(--dq-glass-grouped-bg);
  -webkit-backdrop-filter: var(--dq-glass-blur-light);
  backdrop-filter: var(--dq-glass-blur-light);
  overflow: hidden;
}

/* 面板内所有行：统一水平内边距，避免贴边 */
.studio-image-editor-drawer .studio-editor-drawer-pref-pane .dq-pref-row {
  padding-left: 16px;
  padding-right: 16px;
}

/* 第一行：顶部无分割线 */
.studio-image-editor-drawer .studio-editor-drawer-pref-pane .dq-pref-row:first-child {
  border-top: none;
  padding-top: 12px;
}

/* 最后一行：底部留白 */
.studio-image-editor-drawer .studio-editor-drawer-pref-pane .dq-pref-row:last-child {
  padding-bottom: 12px;
}
</style>
