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
        canvas-media="image"
        :composer-busy="composerBusy"
        @update:filter-time="filterTime = $event"
        @update:filter-models="filterModels = $event"
        @update:search-text="searchText = $event"
        @refresh="refreshGallery"
        @toggle-selection-mode="toggleSelectionMode"
        @select-all="selectAllLoaded"
        @batch-delete="batchDeleteSelected"
        @batch-train-lora="batchTrainLoraSelected"
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
        :groups="galleryGroups"
        :active-tasks="activeImageTasks"
        :loading="galleryLoading"
        :has-more="galleryHasMore"
        media="image"
        :group-mode="groupMode"
        :selected-group-id="selectedGroupId"
        :has-active-filters="hasActiveFilters"
        :selection-mode="selectionMode"
        :selected-paths="selectedPaths"
        :all-selected="allLoadedSelected"
        @select="onGallerySelect"
        @enter-group="enterGroup"
        @exit-group="exitGroup"
        @card-action="onCardAction"
        @reset-filters="resetGalleryFilters"
        @load-more="loadGallery(false)"
        @toggle-select="toggleSelect"
        @select-all="selectAllLoaded"
        @batch-delete="batchDeleteSelected"
        @batch-train-lora="batchTrainLoraSelected"
        @clear-selection="clearSelection"
        @open-composer="openComposerDrawer()"
      />
      <InfiniteCanvas
        v-else
        ref="infiniteCanvasRef"
        :items="galleryItems"
        media="image"
        :editing-path="showEditorDrawer && editDrawerItem ? editDrawerItem.path : ''"
        :mask-preview="canvasMaskPreview"
        :extend-preview="canvasExtendPreview"
        @use-as-reference="onCanvasUseAsRef"
        @use-as-control="onCanvasUseAsControl"
        @card-action="onCardAction"
        @download="downloadItem"
        @toggle-grid-view="() => onViewModeChange('grid')"
        @node-selected="onCanvasNodeSelected"
        @session-ready="onCanvasSessionReady"
        @open-preview="onCanvasOpenPreview"
        @composer-restore="onCanvasComposerRestore"
        @overlay-cleared="onCanvasOverlayCleared"
        @open-composer="onCanvasOpenComposer()"
        @request-close-composer="closeComposerDrawer()"
      />
    </template>
  </StudioLayout>

  <StudioComposeFab
    v-if="!composerDrawerOpen"
    media="image"
    :busy="composerBusy"
    @open="openComposerDrawer()"
  />

  <StudioComposerHost
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
        :control-image="controlImage"
        :inpaint-source-image="inpaintSourceImage"
        :inpaint-mask-image="inpaintMaskImage"
        :mode="imageMode"
        :mode-options="imageModeOptions"
        :current-model-config="currentModelConfig"
        :compatible-loras="compatibleLoras"
        :compatible-control-nets="compatibleControlNets"
        :control-net-runtime-available="controlNetHostRuntimeAvailable"
        @update:mode="onModeChange"
        @generate="onComposerSubmit"
        @pick-reference="openAssetPicker('reference')"
        @remove-reference="removeReferenceImage"
        @pick-control="openAssetPicker('control')"
        @remove-control="removeControlImage"
        @pick-inpaint-source="openAssetPicker('inpaint_source')"
        @remove-inpaint-source="removeInpaintSourceImage"
        @pick-inpaint-mask="openAssetPicker('inpaint_mask')"
        @remove-inpaint-mask="removeInpaintMaskImage"
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

  <!-- Preview dialog -->
  <GalleryPreviewDialog
    v-model:open="previewVisible"
    v-model:index="selectedImageIndex"
    :items="galleryItems"
    media="image"
  />

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
            <DqSelect v-model="retouchModelVersion" size="sm" style="width: 100%" :placeholder="$t('studio.selectModel')">
              <DqOption
                v-for="item in retouchModelOptions"
                :key="item.value"
                :label="item.label"
                :value="item.value"
                :disabled="item.disabled"
              >
                <DqTag
                  v-if="item.commercialUseAllowed"
                  size="sm"
                  type="success"
                  class="studio-drawer-model-badge"
                >
                  {{ $t('download.commercialUseBadge') }}
                </DqTag>
              </DqOption>
            </DqSelect>
          </DqPrefRow>
        </DqPrefPane>

        <p class="studio-retouch-fill-hint">{{ $t('studio.retouchFillHint') }}</p>
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
import { applyDefaults, buildResolutionSizeOptions, hasDeviation, parseSizeValue, pickResolutionForModel, strengthDefaultFromRegistry, strengthToSourceFidelity } from '@/utils/registryParamSchema';
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
  type CompatibleLoraRow,
} from '@/utils/loraAdapterMeta';
import { reconcileVersionPickerSelection } from '@/composables/useModelRegistryFilters';
import { applyModelVersionFilters } from '@/utils/modelPickerFilters';
import { pickDefaultVersionKey } from '@/utils/defaultModelSettings';
import { dismissSuccessorHint, isSuccessorHintDismissed } from '@/utils/modelSuccessor';
import { ACTIVE_COMPOSER_TASK_STATUSES, splitComposerPromptLines } from '@/utils/composerQueue';
// Studio components
import StudioLayout from '@/components/studio/StudioLayout.vue';
import StudioCanvas from '@/components/studio/StudioCanvas.vue';
import StudioGalleryFilters from '@/components/studio/StudioGalleryFilters.vue';
import ImageComposer from '@/components/studio/ImageComposer.vue';
import StudioComposerHost from '@/components/studio/StudioComposerHost.vue';
import StudioComposeFab from '@/components/studio/StudioComposeFab.vue';
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
import { previewUrlForGalleryItem } from '@/utils/canvasAssets';
import { useStudioGallery } from '@/composables/useStudioGallery';
import {
  appendZImageEnhancementFields,
  applyControlNetRegistryDefaults,
  fillModelRegistryDefaultsPatch,
  isFillControlNet,
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
const canvasSelectedItem = ref<GalleryItem | null>(null);

const showEditorDrawer = ref(false);
const {
  composerDrawerOpen,
  openComposerDrawer,
  closeComposerDrawer,
} = useComposerDrawer('image', { editorDrawerOpen: showEditorDrawer });

const savedViewMode = getItem(DQ_STORAGE.IMAGE_VIEW_MODE);
const viewMode = ref<'grid' | 'canvas'>(
  savedViewMode === 'canvas' || savedViewMode === 'grid' ? savedViewMode : 'grid'
);

watch(viewMode, (mode) => {
  setItem(DQ_STORAGE.IMAGE_VIEW_MODE, mode);
});

function onViewModeChange(mode: 'grid' | 'canvas') {
  if (viewMode.value === mode) return;
  if (mode === 'canvas' && selectedGroupId.value) {
    exitGroup();
  }
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
      } else if (Object.keys(imageCanvas.items).length > 0) {
        infiniteCanvasRef.value?.fitAll();
      }
    });
  } else {
    clearSelection();
  }
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
  imageCanvas.addPathsFromGallery(paths, galleryItems.value, {
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
const controlImage = ref<{ previewUrl: string; path: string; assetId?: string } | null>(null);
const inpaintSourceImage = ref<{ previewUrl: string; path: string; assetId?: string } | null>(null);
const inpaintMaskImage = ref<{ previewUrl: string; path: string; assetId?: string } | null>(null);
const showAssetPicker = ref(false);
const assetPickerMode = ref<'reference' | 'control' | 'inpaint_source' | 'inpaint_mask'>('reference');

function openAssetPicker(mode: 'reference' | 'control' | 'inpaint_source' | 'inpaint_mask') {
  assetPickerMode.value = mode;
  showAssetPicker.value = true;
}

function assetIdFromPickPath(path: string): string | undefined {
  return path.startsWith('asset:') ? path.slice('asset:'.length) : undefined;
}

function resolveInpaintAssetId(
  ref: { path: string; assetId?: string } | null,
): string {
  if (!ref) return '';
  if (ref.assetId) return ref.assetId;
  if (ref.path.startsWith('asset:')) return ref.path.slice('asset:'.length);
  return '';
}

function onReferencePick({ path, previewUrl }: { path: string; previewUrl: string }) {
  referenceImage.value = { path, previewUrl, assetId: assetIdFromPickPath(path) };
  showAssetPicker.value = false;
  imageMode.value = 'img2img';
}

function onControlPick({ path, previewUrl }: { path: string; previewUrl: string }) {
  controlImage.value = { path, previewUrl, assetId: assetIdFromPickPath(path) };
  showAssetPicker.value = false;
  syncCanvasControlOverlay(path);
}

function onInpaintSourcePick({ path, previewUrl }: { path: string; previewUrl: string }) {
  if (!path.startsWith('asset:')) {
    toast.warning($tt('canvas.controlImageAssetRequired'));
    return;
  }
  inpaintSourceImage.value = { path, previewUrl, assetId: assetIdFromPickPath(path) };
  showAssetPicker.value = false;
}

function onInpaintMaskPick({ path, previewUrl }: { path: string; previewUrl: string }) {
  if (!path.startsWith('asset:')) {
    toast.warning($tt('canvas.controlImageAssetRequired'));
    return;
  }
  inpaintMaskImage.value = { path, previewUrl, assetId: assetIdFromPickPath(path) };
  showAssetPicker.value = false;
}

function onAssetPickerPick(payload: { path: string; previewUrl: string }) {
  if (assetPickerMode.value === 'control') onControlPick(payload);
  else if (assetPickerMode.value === 'inpaint_source') onInpaintSourcePick(payload);
  else if (assetPickerMode.value === 'inpaint_mask') onInpaintMaskPick(payload);
  else onReferencePick(payload);
}

function removeControlImage() {
  controlImage.value = null;
  syncCanvasControlOverlay(null);
}

function removeInpaintSourceImage() {
  inpaintSourceImage.value = null;
}

function removeInpaintMaskImage() {
  inpaintMaskImage.value = null;
}

function onCanvasUseAsRef(payload: { path: string; previewUrl: string; quiet?: boolean }) {
  const { path, previewUrl } = payload;
  if (payload.quiet) {
    const item = galleryItems.value.find((g) => g.path === path);
    if (item) fillComposerFromGalleryItem(item);
  }
  referenceImage.value = { path, previewUrl, assetId: assetIdFromPickPath(path) };
  imageMode.value = 'img2img';
  if (!payload.quiet) toast.success($tt('canvas.referenceBound'));
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
    referenceImage.value = null;
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
    referenceImage.value = null;
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
  () => controlImage.value?.path ?? null,
  (path) => {
    if (viewMode.value === 'canvas') syncCanvasControlOverlay(path);
    persistComposerSnapshot();
  }
);

function onCanvasOpenPreview(item: GalleryItem) {
  const idx = galleryItems.value.findIndex((g) => g.path === item.path);
  if (idx >= 0) {
    selectedImageIndex.value = idx;
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
  if (!row) {
    params.lora = '';
    return;
  }
  applyLoraComposeOverrides(params as Record<string, unknown>, row);
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
    compatibleLoras.value = (loras as CompatibleLoraRow[]) || [];
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
  if (params.controlnet && !supportsStructuralGuideBaseModel(String(modelKey || ''))) {
    params.controlnet = '';
    controlImage.value = null;
    syncCanvasControlOverlay(null);
  }
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
  return filteredModelPickerVersions.value.map((v) => ({
    label: String(v.name || ''),
    value: `${v.modelKey}|${v.versionKey}`,
    disabled: !v.ready,
    commercialUseAllowed: v.commercialUseAllowed as boolean,
  }));
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

const imageModeOptions = computed(() => [
  { label: $tt('action.image.text2img'), value: 'text2img' },
  { label: $tt('action.image.rewrite'), value: 'img2img' },
]);

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

function attachStreamFromSubmit(submitRes: unknown) {
  const tid = taskIdFromSubmitResponse(submitRes);
  if (!tid) {
    generating.value = false;
    return;
  }
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
      const ids = (resultData?.asset_ids as string[] | undefined) || [];
      if (ids.length > 0) pendingCanvasAssetIds.value = ids;
    },
    onDone: async (doneData: any) => {
      generating.value = false;
      tasksStore.unregisterPageOwnedStream(tid);
      if (doneData.status === 'completed') {
        tasksStore.appendTaskLog(tid, $tt('studio.genComplete'), 'success');
        const ids = [...pendingCanvasAssetIds.value];
        const willAutoAdd = shouldAutoAddToCanvas() && ids.length > 0;
        if (!willAutoAdd) {
          toast.success($tt('studio.genComplete'));
        }
        pendingCanvasAssetIds.value = [];
        setTimeout(async () => {
          await loadGallery(true);
          if (willAutoAdd) {
            await activateCanvasViewForResults(viewMode, syncCompositorOverlaysOnCanvasEnter);
            addAssetPathsToCanvas(ids.map((id) => `asset:${id}`), { placement: 'staging' });
          }
        }, 1000);
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
  const adapters: Array<{ id: string; weight: number }> = [];
  if (params.lora) adapters.push({ id: String(params.lora), weight: Number(params.lora_scale) || 0.8 });
  persistComposerSnapshot();
  const meta = buildCanvasMeta();
  if (params.scheduler) meta.scheduler = params.scheduler;

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

function modelSupportsEditMultiReference(): boolean {
  const raw = currentModelConfig.value?.parameters as Record<string, unknown> | undefined;
  return Boolean(raw?.edit_plus_multi_image);
}

function collectExtraReferenceAssetIds(
  sourceAssetId: string | null,
  controlAssetId: string | null,
): string[] {
  if (!modelSupportsEditMultiReference()) return [];
  const raw = currentModelConfig.value?.parameters as Record<string, unknown> | undefined;
  const maxRefs = Number(raw?.edit_max_reference_images ?? 1);
  const cap = Math.max(1, maxRefs) - 1;
  const ids: string[] = [];
  if (controlAssetId && controlAssetId !== sourceAssetId) {
    ids.push(controlAssetId);
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
  const inpSrc = resolveInpaintAssetId(inpaintSourceImage.value);
  const inpMsk = resolveInpaintAssetId(inpaintMaskImage.value);
  const enhCommon = {
    parameters: currentModelConfig.value?.parameters as Record<string, unknown> | undefined,
    params,
    controlnet: String(params.controlnet || ''),
    controlAssetId: ctx.control_asset_id,
    controlnetStrength: Number(params.controlnet_strength) || 0.8,
    inpaintSourceId: inpSrc,
    inpaintMaskId: inpMsk,
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
      adapters: ctx.adapters,
      source_fidelity: strengthToSourceFidelity(
        params.strength,
        strengthDefaultFromRegistry(currentModelConfig.value?.parameters as Record<string, unknown> | undefined),
      ),
      metadata: { ...ctx.meta },
      priority: 'normal',
    };
    const editEnh = appendZImageEnhancementFields(editBody, enhCommon);
    if (!editEnh.ok) {
      throw new Error($tt('create.inpaintPairRequired'));
    }
    const extraRefs = collectExtraReferenceAssetIds(ctx.source_asset_id, ctx.control_asset_id);
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
  const baseSeed = params.seed ? parseInt(String(params.seed), 10) : null;
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

    const seedNum = params.seed ? parseInt(String(params.seed), 10) : null;
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

    if (!selectedModelVersion.value) {
      // 尝试从本地存储恢复上次选择的模型
      const lastModel = getItem(DQ_STORAGE.IMAGE_LAST_MODEL);
      if (lastModel) {
        const parsed = parseModelVersionValue(lastModel);
        if (parsed) {
          const detailed = (detailedStatusData as Record<string, Record<string, unknown>>)[parsed.modelKey] || {};
          const versions = detailed.versions || {};
          if (versions[parsed.versionKey]?.ready) {
            params.model = parsed.modelKey;
            params.version = parsed.versionKey;
            selectedModelVersion.value = lastModel;
            loadModelDefaults();
            restoreSavedSize();
            loadCompatibleAdapters(parsed.modelKey);
            return;
          }
        }
      }

      let found = false;
      for (const [modelKey, config] of Object.entries(modelRegistry.value)) {
        if (config.recommended) {
          const detailed = (detailedStatusData as Record<string, Record<string, unknown>>)[modelKey] || {};
          const versions = detailed.versions || {};
          const defaultVersionKey = Object.keys(config.versions || {}).find((k) => (config.versions as Record<string, Record<string, unknown>>)[k]?.default) || Object.keys(config.versions || {})[0];

          if (defaultVersionKey && versions[defaultVersionKey]?.ready) {
            params.model = modelKey;
            params.version = defaultVersionKey;
            selectedModelVersion.value = modelKey + '|' + defaultVersionKey;
            found = true;
            break;
          }
        }
      }

      if (!found) {
        for (const [modelKey, config] of Object.entries(modelRegistry.value)) {
          const detailed = (detailedStatusData as Record<string, Record<string, unknown>>)[modelKey] || {};
          const versions = detailed.versions || {};
          for (const versionKey of Object.keys(config.versions || {})) {
            if (versions[versionKey]?.ready) {
              params.model = modelKey;
              params.version = versionKey;
              selectedModelVersion.value = modelKey + '|' + versionKey;
              found = true;
              break;
            }
          }
          if (found) break;
        }
      }

      if (!found) {
        const firstModel = Object.keys(modelRegistry.value)[0];
        if (firstModel) {
          const firstVersion = Object.keys(modelRegistry.value[firstModel].versions || {})[0] || 'default';
          params.model = firstModel;
          params.version = firstVersion;
          selectedModelVersion.value = firstModel + '|' + firstVersion;
        }
      }
    }

    loadModelDefaults();
    restoreSavedSize();
  } catch (e) {
    console.error('Failed to load model registry:', e);
  }
};

const loadModelDefaults = () => {
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
  toast.success($tt('studio.restoredDefaults'));
};

/* ------------------------------------------------------------------ */
/*  Gallery interactions                                               */
/* ------------------------------------------------------------------ */

const previewVisible = ref(false);
const selectedImageIndex = ref(0);

function getImageUrl(item: GalleryItem) {
  return api.gallery.getImageUrl(item.path);
}

function onGallerySelect(item: GalleryItem) {
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
    .map((v) => ({
      label: String(v.name || ''),
      value: `${v.modelKey}|${v.versionKey}`,
      disabled: !v.ready,
      commercialUseAllowed: v.commercialUseAllowed as boolean,
    }));
});
const extendModelVersion = ref('');
const extendModelOptions = computed(() => {
  const rows = allVersions.value
    .filter((v) => {
      const acts = (v.actions as Record<string, unknown>) || {};
      return imageSupportsExtend(acts);
    })
    .map((v) => ({
      label: String(v.name || '') + (!v.ready ? ` (${$tt('common.notReady')})` : ''),
      value: `${v.modelKey}|${v.versionKey}`,
      disabled: !v.ready,
      commercialUseAllowed: v.commercialUseAllowed as boolean,
      isFill: isFillControlNet(String(v.modelKey || '')),
    }));
  rows.sort((a, b) => {
    if (a.isFill !== b.isFill) return a.isFill ? -1 : 1;
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
  const patch = fillModelRegistryDefaultsPatch(modelKey, modelRegistry.value);
  if (patch.guidance != null) fillEditParams.guidance = patch.guidance;
  if (patch.steps != null) fillEditParams.steps = patch.steps;
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
      return imageSupportsRetouch(acts);
    })
    .map((v) => ({
      label: String(v.name || '') + (!v.ready ? ` (${$tt('common.notReady')})` : ''),
      value: `${v.modelKey}|${v.versionKey}`,
      disabled: !v.ready,
      commercialUseAllowed: v.commercialUseAllowed as boolean,
      modelKey: String(v.modelKey || ''),
      isFill: isFillControlNet(String(v.modelKey || '')),
    }));
  rows.sort((a, b) => {
    if (a.isFill !== b.isFill) return a.isFill ? -1 : 1;
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
        if (currentSupportsExtend && isFillControlNet(currentKey.split('|')[0] || '')) {
          extendModelVersion.value = currentKey;
        } else {
          const first = extendModelOptions.value.find((o) => !o.disabled);
          extendModelVersion.value = first ? first.value : '';
        }
        if (extendModelVersion.value) applyFillEditDefaults(extendModelVersion.value);
      } else if (action === 'retouch') {
        const currentKey = params.model && params.version ? `${params.model}|${params.version}` : '';
        const currentSupportsRetouch = allVersions.value.some((v) => {
          const acts = (v.actions as Record<string, unknown>) || {};
          return `${v.modelKey}|${v.versionKey}` === currentKey && imageSupportsRetouch(acts);
        });
        if (currentSupportsRetouch && isFillControlNet(currentKey.split('|')[0] || '')) {
          retouchModelVersion.value = currentKey;
        } else {
          const first = retouchModelOptions.value.find((o) => !o.disabled);
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
      await api.gen.uploadAsset(new File([maskBlob], 'mask.png', { type: 'image/png' }))
    ).id as string;

    if (!retouchModelVersion.value) {
      toast.warning($tt('studio.selectModel'));
      return;
    }
    const [modelKey, versionKey] = retouchModelVersion.value.split('|');
    if (!isFillControlNet(modelKey)) {
      toast.warning($tt('studio.retouchFillModelRequired'));
      return;
    }
    if (!isEditModelVersionReady(retouchModelVersion.value)) {
      toast.warning($tt('studio.modelNotReadyDesc', { name: modelKey, version: versionKey }));
      return;
    }
    const modelStr = versionKey ? `${modelKey}:${versionKey}` : modelKey;
    const seedNum = params.seed ? parseInt(String(params.seed), 10) : null;
    const adapters: Array<{ id: string; weight: number }> = [];
    if (params.lora) adapters.push({ id: String(params.lora), weight: Number(params.lora_scale) || 0.8 });

    const submitRes = await api.gen.createImageEdit({
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
    if (!isFillControlNet(modelKey)) {
      toast.warning($tt('studio.extendFillModelRequired'));
      return;
    }
    if (!isEditModelVersionReady(extendModelVersion.value)) {
      toast.warning($tt('studio.modelNotReadyDesc', { name: modelKey, version: versionKey }));
      return;
    }
    const modelStr = versionKey ? `${modelKey}:${versionKey}` : modelKey;
    const seedNum = params.seed ? parseInt(String(params.seed), 10) : null;
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
  await loadModelRegistry();
  loadPresets();
  loadRecentImages();
  loadGallery(true);
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

<style>
.studio-drawer-model-label {
  flex: 1 1 auto;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.studio-drawer-model-badge {
  flex-shrink: 0;
  margin-left: 6px;
}
</style>
