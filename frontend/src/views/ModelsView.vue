<!-- @ts-nocheck -->
<template>
  <div class="models-page">
    <!-- Left sidebar: category navigation -->
    <div class="models-page__sidebar">
      <DqSurfaceCard class="models-page__sidebar-card">
        <div class="card-title">
          <DqIcon><box /></DqIcon>
          {{ $t('download.modelLibrary') }}
        </div>
        <div class="models-page__sidebar-intro">
          {{ $t('models.pageSubtitle') }}
        </div>

        <ModelsCategoryNav
          :active-category="activeCategory"
          :total-model-count="totalModelCount"
          :active-download-count="activeDownloadCount"
          @select="handleCategorySelect"
        />


        <!-- Disk space -->
        <div v-if="diskSpace" class="disk-space-panel">
          <div class="disk-space-title">
            <DqIcon><monitor /></DqIcon>
            {{ $t('download.diskSpace') }}
          </div>
          <div class="disk-space-item">
            <div class="disk-space-label">
              <span>{{ $t('download.modelLabel') }}</span>
              <span class="disk-space-value">{{ diskSpace.models?.size_human }}</span>
            </div>
            <DqProgress
              :percentage="getDiskPercent('models')"
              :show-text="false"
              :stroke-width="4"
            />
          </div>
          <div class="disk-space-item">
            <div class="disk-space-label">
              <span>{{ $t('download.loraLabel') }}</span>
              <span class="disk-space-value">{{ diskSpace.loras?.size_human }}</span>
            </div>
            <DqProgress
              :percentage="getDiskPercent('loras')"
              :show-text="false"
              :stroke-width="4"
              color="var(--dq-success)"
            />
          </div>
          <div class="disk-space-footer">
            {{ $t('download.free') }}: {{ diskSpace.models?.free_human }}
          </div>
        </div>
      </DqSurfaceCard>
    </div>

    <!-- Right content area -->
    <div class="models-page__main">
      <!-- Model grid (category browsing) -->
      <div
        v-if="
          [
            'all',
            'image_models',
            'video_models',
            'music_models',
            'llm_models',
            'vlm_models',
            'controlnets',
            'upscalers',
            'tools',
            'loras',
          ].includes(activeCategory)
        "
      >
        <!-- Category title -->
        <div class="page-header models-page__page-header">
          <h2 class="page-title models-page__category-title">
            <DqIcon v-if="categoryPageIcon" class="models-page__title-icon" aria-hidden="true">
              <component :is="categoryPageIcon" />
            </DqIcon>
            <span>{{ categoryTitleText }}</span>
          </h2>
          <div
            v-if="
              [
                'all',
                'image_models',
                'video_models',
                'music_models',
                'llm_models',
                'vlm_models',
                'controlnets',
                'upscalers',
                'tools',
                'loras',
              ].includes(activeCategory)
            "
            class="models-page__header-center"
          >
            <DqInput
              v-model="filterQuery"
              :placeholder="$t('download.searchModel')"
              class="models-page__search-input"
              clearable
            >
              <template #prefix>
                <DqIcon><search /></DqIcon>
              </template>
            </DqInput>
            <ModelPickerFilters
              v-model:installed-only="modelFilterInstalledOnly"
              v-model:commercial-only="modelFilterCommercialOnly"
              v-model:current-models-only="modelFilterCurrentModelsOnly"
            />
          </div>
          <div class="page-actions models-page__actions">
            <DqButton
              v-if="
                activeCategory !== 'loras' && activeCategory !== 'installed'
              "
              class="models-toolbar-btn models-page__import-btn"
              @click="showImportDialog"
            >
              <DqIcon class="models-toolbar-btn__icon"><upload /></DqIcon>
              <span class="models-toolbar-btn__label">{{ $t('download.importLocal') }}</span>
            </DqButton>
            <DqButton
              class="models-toolbar-btn models-page__refresh-btn"
              :loading="refreshing"
              @click="refreshStatus"
            >
              <DqIcon class="models-toolbar-btn__icon"><refresh /></DqIcon>
              <span class="models-toolbar-btn__label">{{ $t('gallery.refresh') }}</span>
            </DqButton>
          </div>
        </div>

        <ModelsImportDialog
          v-model:open="importDialogVisible"
          v-model:import-model-name="importModelName"
          v-model:import-model-path="importModelPath"
          v-model:import-model-type="importModelType"
          :importing="importing"
          @submit="importLocalModel"
          @cancel="importDialogVisible = false"
        />

        <ZImageMergePanel
          v-if="activeCategory === 'tools'"
          class="models-page__merge-panel"
          @merged-complete="refreshUserMergedModels"
        />

        <div
          v-if="activeCategory === 'tools' && userMergedModels.length"
          class="models-page__merged-list"
        >
          <h3 class="models-page__merged-title">{{ $t('tools.mergedModelsTitle') }}</h3>
          <ul class="models-page__merged-items">
            <li v-for="m in userMergedModels" :key="m.id" class="models-page__merged-item">
              <span class="models-page__merged-name">{{ m.name || m.id }}</span>
              <span class="models-page__merged-id">{{ m.id }}</span>
            </li>
          </ul>
        </div>

        <!-- Model card grid -->
        <DqRow :gutter="16" class="model-grid model-grid--fluid">
          <DqCol
            v-for="model in filteredModels"
            :key="model.id"
            :xs="24"
            :sm="12"
            :md="12"
            :lg="8"
            :xl="8"
            class="models-page__col-mb"
          >
            <DqSurfaceCard
              class="model-card"
              :class="{ 'model-ready': model.ready }"
            >
              <!-- Card header: icon/preview + status -->
              <div class="model-card-header">
                <div class="model-icon">
                  {{ getModelInitials(model) }}
                </div>
                <div
                  class="dq-status-dot model-card-status"
                  :class="{
                    'dq-status-dot--success': modelsDetailedStatus[model.id]?.status === 'ready',
                    'dq-status-dot--danger': modelsDetailedStatus[model.id]?.status === 'incomplete',
                    'dq-status-dot--warning': !modelsDetailedStatus[model.id]?.status
                      || modelsDetailedStatus[model.id]?.status === 'missing',
                  }"
                  :title="modelStatusTitle(model.id)"
                />
              </div>

              <!-- Card content -->
              <div class="model-card-content">
                <DqTooltip :content="$mn(model)" placement="top">
                  <div class="model-card-name">
                    {{ $mn(model) }}
                  </div>
                </DqTooltip>
                <DqTooltip
                  :content="$md(model)"
                  placement="top"
                >
                  <div class="model-card-desc">
                    {{ $md(model) }}
                  </div>
                </DqTooltip>

                <div class="model-card-attrs">
                  <ModelLicenseBadges
                    :recommended="model.recommended"
                    :commercial-use-allowed="model.commercial_use_allowed"
                    effect="plain"
                    size="sm"
                  />
                  <ModelVersionSourceBadge
                    v-if="modelCardSource(model)"
                    :source="modelCardSource(model)"
                  />
                </div>

                <ModelCatalogRelation
                  v-if="modelDistilledRelation(model)"
                  v-bind="modelDistilledRelation(model)!"
                  @navigate="focusRelatedModel"
                />
                <ModelCatalogRelation
                  v-if="modelSuccessorRelation(model)"
                  v-bind="modelSuccessorRelation(model)!"
                  @navigate="focusRelatedModel"
                />

                <div
                  v-if="model.size || model.base_model"
                  class="model-card-meta"
                >
                  <DqTag
                    v-if="model.size"
                    type="info"
                    effect="plain"
                  >
                    {{ model.size }}
                  </DqTag>
                  <DqTag
                    v-if="model.base_model"
                    type="success"
                    effect="plain"
                  >
                    {{ model.base_model }}
                  </DqTag>
                </div>

                <ModelCardVersions
                  v-if="model.versions"
                  :model="model"
                  :uniform-source="uniformDownloadSource(model)"
                  :loading-keys="downloadingModels"
                  :can-download="canDownload(model)"
                  :get-version-status="getVersionStatus"
                  :bundle-components-for="(verKey) => versionBundleComponents(model.id, verKey)"
                  @download="downloadVersion(model, $event)"
                  @delete="deleteVersion(model, $event)"
                  @quantize="quantizeVersion(model, $event)"
                />

              </div>
            </DqSurfaceCard>
          </DqCol>
        </DqRow>

        <DqEmpty
          v-if="filteredModels.length === 0"
          :description="$t('download.noModelsInCategory')"
        />
      </div>

      <!-- Downloaded LoRAs (remote search) -->
      <div v-if="activeCategory === 'downloaded_loras'" class="trained-loras-page">
        <div class="page-header models-page__page-header">
          <h2 class="page-title">{{ $t('download.downloadedLoras') }}</h2>
          <div class="page-actions models-page__actions">
            <DqButton
              size="sm"
              type="secondary"
              :loading="downloadedLorasRefreshing"
              @click="refreshDownloadedLoras"
            >
              {{ $t('gallery.refresh') }}
            </DqButton>
            <DqButton type="primary" @click="activeCategory = 'lora_search'">
              {{ $t('download.loraSearch') }}
            </DqButton>
          </div>
        </div>
        <p class="models-page__section-hint">{{ $t('download.downloadedLorasHint') }}</p>

        <DqRow v-if="downloadedLoras.length" :gutter="16" class="model-grid">
          <DqCol
            v-for="ul in downloadedLoras"
            :key="ul.id"
            :xs="24"
            :sm="12"
            :md="8"
            :lg="8"
            class="models-page__col-mb"
          >
            <UserLoraCard
              :lora="ul"
              :base-model-label="userLoraBaseModelLabel(ul.base_model)"
              @verify="verifyUserLora(ul)"
              @delete="deleteDownloadedLora(ul)"
            />
          </DqCol>
        </DqRow>
        <div v-else class="trained-loras-page__empty">
          <DqEmpty :description="$t('download.noDownloadedLoras')" />
          <DqButton type="primary" @click="activeCategory = 'lora_search'">
            {{ $t('download.loraSearch') }}
          </DqButton>
        </div>
      </div>

      <!-- User-trained LoRAs -->
      <div v-if="activeCategory === 'trained_loras'" class="trained-loras-page">
        <div class="page-header models-page__page-header">
          <h2 class="page-title">{{ $t('download.myTrainedLoras') }}</h2>
          <div class="page-actions models-page__actions">
            <DqButton
              size="sm"
              type="secondary"
              :loading="userLorasRefreshing"
              @click="refreshUserLoras"
            >
              {{ $t('loraTrain.refreshHistory') }}
            </DqButton>
            <DqButton type="primary" @click="goToLoraTrain">
              {{ $t('loraTrain.startTraining') }}
            </DqButton>
          </div>
        </div>

        <DqRow v-if="userLoras.length" :gutter="16" class="model-grid">
          <DqCol
            v-for="ul in userLoras"
            :key="ul.id"
            :xs="24"
            :sm="12"
            :md="8"
            :lg="8"
            class="models-page__col-mb"
          >
            <UserLoraCard
              :lora="ul"
              :base-model-label="userLoraBaseModelLabel(ul.base_model)"
              @verify="verifyUserLora(ul)"
              @view-run="openTrainingRun"
              @delete="deleteUserLora(ul)"
            />
          </DqCol>
        </DqRow>
        <div v-else class="trained-loras-page__empty">
          <DqEmpty :description="$t('download.noUserLoras')" />
          <DqButton type="primary" @click="goToLoraTrain">
            {{ $t('loraTrain.startTraining') }}
          </DqButton>
        </div>
      </div>

      <!-- LoRA remote search (list view, separate from registry cards) -->
      <LoraSearchPanel
        v-if="activeCategory === 'lora_search'"
        :connect-progress="connectProgressSSE"
      />

      <!-- CivitAI search removed — use LoRA category search panel -->
      <!-- Installed -->
      <div v-if="activeCategory === 'installed'">
        <div class="page-header">
          <h2 class="page-title">{{ $t('download.installedLabel') }}</h2>
        </div>

        <div v-if="installedModels.length" class="models-installed-list" role="list">
          <article
            v-for="(row, idx) in installedModels"
            :key="row.path || row.name || idx"
            class="models-installed-row"
            role="listitem"
          >
            <div class="models-installed-row__main">
              <span class="models-installed-row__name">{{ row.name }}</span>
              <DqTag :type="getModelTypeTagType(row.type)">
                {{ row.type || 'unknown' }}
              </DqTag>
            </div>
            <span class="models-installed-row__size">{{ row.size_human }}</span>
            <span class="models-installed-row__path" :title="row.path">{{ row.path }}</span>
          </article>
        </div>
        <DqEmpty v-else :description="$t('download.noModels')" />
      </div>

      <!-- Downloading -->
      <div v-if="activeCategory === 'downloading'">
        <div class="page-header">
          <h2 class="page-title">
            {{ $t('download.downloadingTab') }} ({{ activeDownloadCount }})
          </h2>
        </div>

        <DqSurfaceCard v-if="activeDownloadCount === 0" class="studio-surface-card">
          <DqEmpty :description="$t('download.noTasks')" />
        </DqSurfaceCard>

        <DqSurfaceCard v-else class="studio-surface-card">
          <div
            v-for="(item, taskId) in activeDownloads"
            :key="taskId"
            class="models-download-task"
          >
            <div class="models-download-task-head">
              <span class="models-download-task-name">{{ formatDownloadDisplayName(item.name) }}</span>
              <div class="models-download-task-meta">
                <span class="models-download-progress-text">
                  <span v-if="item.status === 'failed' && item.error">
                    {{ $t('download.failed') }}
                  </span>
                  <span v-else-if="item.total_size > 0">
                    {{ Math.round(item.progress * 100) }}%
                    <span v-if="item.speed">({{ item.speed }})</span>
                  </span>
                  <span v-else-if="item.downloaded_size > 0">
                    {{ formatBytes(item.downloaded_size) }}
                    <span v-if="item.speed">({{ item.speed }})</span>
                  </span>
                  <span v-else>{{ $t('download.preparing') }}</span>
                </span>
                <DqButton size="sm"
                  v-if="item.status === 'paused'"
                  class="model-ver-btn model-ver-btn--download"
                  @click="resumeDownload(taskId)"
                >
                  <DqIcon class="model-ver-btn__icon"><video-play /></DqIcon>
                  <span class="model-ver-btn__label">{{ $t('download.resume') }}</span>
                </DqButton>
                <DqButton size="sm"
                  v-else-if="item.status === 'running'"
                  class="model-ver-btn model-ver-btn--neutral"
                  @click="cancelDownload(taskId)"
                >
                  <DqIcon class="model-ver-btn__icon"><close /></DqIcon>
                  <span class="model-ver-btn__label">{{ $t('download.cancelDownload') }}</span>
                </DqButton>
                <template v-else-if="item.status === 'failed'">
                  <DqButton size="sm"
                    class="model-ver-btn model-ver-btn--download"
                    @click="resumeDownload(taskId)"
                  >
                    <DqIcon class="model-ver-btn__icon"><video-play /></DqIcon>
                    <span class="model-ver-btn__label">{{ $t('download.retryDownload') }}</span>
                  </DqButton>
                  <DqButton size="sm"
                    class="model-ver-btn model-ver-btn--delete"
                    @click="deleteDownload(taskId)"
                  >
                    <DqIcon class="model-ver-btn__icon"><delete /></DqIcon>
                    <span class="model-ver-btn__label">{{ $t('download.deleteTask') }}</span>
                  </DqButton>
                </template>
              </div>
            </div>
            <DqProgress
              :percentage="
                item.total_size > 0 ? Math.round(item.progress * 100) : 0
              "
              :status="item.status === 'failed' ? 'exception' : ''"
              :stroke-width="8"
              :show-text="item.total_size > 0"
            />
            <div
              v-if="item.error"
              class="models-download-error"
            >
              {{ item.error }}
            </div>
          </div>
        </DqSurfaceCard>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
// @ts-nocheck
import { ref, reactive, computed, onMounted, onUnmounted, watch } from 'vue';
import { onBeforeRouteLeave, useRouter } from 'vue-router';
import { useI18n } from 'vue-i18n';
import { toast, confirm } from '@/utils/feedback';
import { api } from '@/utils/api';
import { $tt, $mn, $md, $mvn } from '@/utils/i18n';
import { openLoraTrainingRun } from '@/utils/loraTrainHandoff';
import { modelInitialsFromName } from '@/utils/modelInitials';
import { useRegistryStore } from '@/stores/registry';
import { DQ_STORAGE, consumeStringDraft, getItem, setItem } from '@/utils/storage';
import ModelLicenseBadges from '@/components/model/ModelLicenseBadges.vue';
import ModelCatalogRelation from '@/components/model/ModelCatalogRelation.vue';
import ModelPickerFilters from '@/components/model/ModelPickerFilters.vue';
import ModelsImportDialog from '@/components/models/ModelsImportDialog.vue';
import ModelsCategoryNav from '@/components/models/ModelsCategoryNav.vue';
import LoraSearchPanel from '@/components/models/LoraSearchPanel.vue';
import UserLoraCard from '@/components/lora/UserLoraCard.vue';
import ModelCardVersions from '@/components/models/ModelCardVersions.vue';
import ModelVersionSourceBadge from '@/components/models/ModelVersionSourceBadge.vue';
import ZImageMergePanel from '@/components/tools/ZImageMergePanel.vue';
import { uniformDownloadSource } from '@/utils/modelVersionLayout';
import { formatDownloadDisplayName } from '@/utils/registryLabel';
import { useModelRegistryFilters } from '@/composables/useModelRegistryFilters';
import { modelPassesRegistryFilters } from '@/utils/modelPickerFilters';

/* ───── Types ───── */

interface ModelVersion {
  name: string | { zh?: string; en?: string };
  size?: string;
  source?: string;
  source_type?: string;
  from_version?: string;
}

interface ModelConfig {
  media?: string;
  actions?: Record<string, unknown>;
  name?: string | { zh?: string; en?: string };
  name_en?: string;
  description?: string | { zh?: string; en?: string };
  description_en?: string;
  category?: string;
  size?: string;
  source?: string;
  base_model?: string;
  recommended?: boolean;
  commercial_use_allowed?: boolean | null;
  successor?: string;
  distilled_from?: string;
  distilled_variant?: string;
  versions?: Record<string, ModelVersion>;
  ready?: boolean;
}

interface ModelRow extends ModelConfig {
  id: string;
  ready: boolean;
}

interface DownloadItem {
  name: string;
  progress: number;
  status: string;
  speed: string;
  error: string;
  total_size: number;
  downloaded_size: number;
  kind?: string;
}

interface DiskInfo {
  size_human?: string;
  free_human?: string;
  exists?: boolean;
  total?: number;
  size?: number;
}

interface DiskSpaceData {
  models?: DiskInfo;
  loras?: DiskInfo;
}

/* ───── State ───── */

const activeCategory = ref('all');
const userLoras = ref<any[]>([]);
const downloadedLoras = ref<any[]>([]);
const userMergedModels = ref<any[]>([]);
const userLorasRefreshing = ref(false);
const downloadedLorasRefreshing = ref(false);
const router = useRouter();
const { locale } = useI18n();
const modelRegistry = ref<Record<string, ModelConfig>>({});
const modelsStatus = ref<Record<string, boolean>>({});
const modelsDetailedStatus = ref<Record<string, any>>({});
const categories = ref<Record<string, any>>({});
const filterQuery = ref('');
const { installedOnly: modelFilterInstalledOnly, commercialOnly: modelFilterCommercialOnly, currentModelsOnly: modelFilterCurrentModelsOnly } =
  useModelRegistryFilters();
const refreshing = ref(false);

const registryStore = useRegistryStore();

const downloadingModels = ref<Record<string, boolean>>({});
const activeDownloads = ref<Record<string, DownloadItem>>({});
const sseConnections = ref<Record<string, EventSource>>({});

const installedModels = ref<any[]>([]);
const diskSpace = ref<DiskSpaceData | null>(null);

const importDialogVisible = ref(false);
const importModelName = ref('');
const importModelPath = ref('');
const importModelType = ref('base');
const importing = ref(false);

/* ───── Helpers ───── */

/** Icons aligned with ModelsCategoryNav sidebar items. */
const categoryPageIcon = computed(() => {
  const icons: Record<string, string> = {
    all: 'Grid',
    image_models: 'PictureFilled',
    video_models: 'VideoCamera',
    music_models: 'Headset',
    llm_models: 'Document',
    vlm_models: 'PictureFilled',
    controlnets: 'Aim',
    upscalers: 'ZoomIn',
    tools: 'Tools',
    loras: 'MagicStick',
    lora_search: 'Search',
  };
  return icons[activeCategory.value] ?? null;
});

const categoryTitleText = computed(() => {
  const titles: Record<string, string> = {
    all: $tt('download.allModels'),
    image_models: $tt('download.imageModels'),
    video_models: $tt('download.videoModels'),
    music_models: $tt('download.audioModels'),
    llm_models: $tt('download.llmModels'),
    vlm_models: $tt('download.vlmModels'),
    controlnets: $tt('download.controlNet'),
    upscalers: $tt('download.upscalers'),
    tools: $tt('download.tools'),
    loras: $tt('download.loraModels'),
    lora_search: $tt('download.loraSearch'),
  };
  return titles[activeCategory.value] || $tt('download.title');
});

function modelSearchBlob(m: ModelRow): string {
  const parts = [$mn(m, m.id), $md(m, '')];
  for (const relId of [m.successor, m.distilled_from, m.distilled_variant]) {
    if (!relId) continue;
    const target = modelRegistry.value[relId];
    if (target) parts.push($mn(target, relId));
  }
  return parts.join(' ').toLowerCase();
}

function relatedModelName(modelId?: string): string {
  if (!modelId) return '';
  const target = modelRegistry.value[modelId];
  return target ? $mn(target, modelId) : modelId;
}

interface CatalogRelationProps {
  roleLabel: string;
  navLabel: string;
  targetName: string;
  targetId: string;
  roleTagType?: 'info' | 'success' | 'warning';
}

function modelDistilledRelation(model: ModelRow): CatalogRelationProps | null {
  if (model.distilled_from) {
    return {
      roleLabel: $tt('download.modelRoleDistilled'),
      navLabel: $tt('download.viewBaseModel'),
      targetName: relatedModelName(model.distilled_from),
      targetId: model.distilled_from,
      roleTagType: 'warning',
    };
  }
  if (model.distilled_variant) {
    return {
      roleLabel: $tt('download.modelRoleBase'),
      navLabel: $tt('download.viewDistilledVariant'),
      targetName: relatedModelName(model.distilled_variant),
      targetId: model.distilled_variant,
      roleTagType: 'success',
    };
  }
  return null;
}

function modelSuccessorRelation(model: ModelRow): CatalogRelationProps | null {
  const sid = model.successor;
  if (!sid) return null;
  return {
    roleLabel: $tt('download.successorBadge'),
    navLabel: $tt('download.viewSuccessor'),
    targetName: relatedModelName(sid),
    targetId: sid,
    roleTagType: 'info',
  };
}

function focusRelatedModel(modelId: string): void {
  if (!modelId) return;
  filterQuery.value = relatedModelName(modelId) || modelId;
}

const filteredModels = computed(() => {
  const list: ModelRow[] = [];
  for (const [id, config] of Object.entries(modelRegistry.value)) {
    const model: ModelRow = {
      id,
      ...config,
      ready: modelsStatus.value[id] || false,
    };

    if (activeCategory.value !== 'all') {
      const regCat = model.category;
      if (activeCategory.value === 'image_models') {
        if (regCat !== 'base_models') continue;
      } else if (activeCategory.value !== regCat) {
        continue;
      }
    }

    if (filterQuery.value) {
      const query = filterQuery.value.toLowerCase();
      if (!modelSearchBlob(model).includes(query)) {
        continue;
      }
    }

    if (
      !modelPassesRegistryFilters(model, {
        installedOnly: modelFilterInstalledOnly.value,
        commercialOnly: modelFilterCommercialOnly.value,
        currentModelsOnly: modelFilterCurrentModelsOnly.value,
      })
    ) {
      continue;
    }

    list.push(model);
  }

  return list.sort((a, b) => $mn(a, a.id).localeCompare($mn(b, b.id)));
});

const totalModelCount = computed(() => Object.keys(modelRegistry.value).length);

const activeDownloadCount = computed(() => {
  return Object.values(activeDownloads.value).filter(
    (item) =>
      item.status === 'running' ||
      item.status === 'paused' ||
      item.status === 'failed'
  ).length;
});

/* ───── Loaders ───── */

async function loadModelRegistry() {
  try {
    const [registryResult, statusResult, detailedResult] = await Promise.allSettled([
      registryStore.load(),
      api.settings.getModelsStatus(),
      api.settings.getModelsDetailedStatus(),
    ]);
    if (registryResult.status !== 'fulfilled' || !registryResult.value) {
      throw registryResult.status === 'rejected' ? registryResult.reason : new Error('registry empty');
    }
    const reg = registryResult.value || { models: {}, categories: {} };
    modelRegistry.value = reg.models || {};
    categories.value = reg.categories || {};
    if (statusResult.status === 'fulfilled') {
      modelsStatus.value = (statusResult.value as Record<string, boolean>) || {};
    } else {
      console.error('Failed to load model status:', statusResult.reason);
      modelsStatus.value = {};
    }
    if (detailedResult.status === 'fulfilled') {
      modelsDetailedStatus.value = (detailedResult.value as typeof modelsDetailedStatus.value) || {};
    } else {
      console.error('Failed to load detailed model status:', detailedResult.reason);
      modelsDetailedStatus.value = {};
    }
  } catch (e) {
    console.error('Failed to load model registry:', e);
  }
}

async function refreshStatus() {
  refreshing.value = true;
  try {
    const [statusData, detailedStatusData] = await Promise.all([
      api.settings.getModelsStatus(),
      api.settings.getModelsDetailedStatus(),
    ]);
    modelsStatus.value = statusData || {};
    modelsDetailedStatus.value = detailedStatusData || {};
    await loadInstalled();
    await loadDiskSpace();
  } catch (e) {
    console.error('Refresh failed:', e);
  } finally {
    refreshing.value = false;
  }
}

async function loadInstalled() {
  try {
    const models = await api.settings.listModels();
    installedModels.value = (models as any[]) || [];
  } catch (e) {
    console.error('Failed to load installed:', e);
  }
}

async function loadDownloadedLoras() {
  try {
    const res = (await api.loras.listDownloadedAdapters()) as { items?: any[] };
    downloadedLoras.value = res.items || [];
  } catch {
    downloadedLoras.value = [];
  }
}

async function refreshDownloadedLoras() {
  downloadedLorasRefreshing.value = true;
  try {
    await loadDownloadedLoras();
  } finally {
    downloadedLorasRefreshing.value = false;
  }
}

async function loadUserLoras() {
  try {
    const res = (await api.loras.listUserAdapters()) as { items?: any[] };
    userLoras.value = res.items || [];
  } catch {
    userLoras.value = [];
  }
}

async function refreshUserLoras() {
  userLorasRefreshing.value = true;
  try {
    await loadUserLoras();
  } finally {
    userLorasRefreshing.value = false;
  }
}

function verifyUserLora(ul: { id?: string; base_model?: string }) {
  router.push({
    name: 'image_create',
    query: {
      model: String(ul.base_model || 'flux1-dev'),
      lora: String(ul.id || ''),
    },
  });
}

function userLoraBaseModelLabel(modelId: string | undefined): string {
  const id = String(modelId || '').trim();
  if (!id) return '—';
  const model = modelRegistry.value[id];
  return model ? $mn(model, id) : id;
}

function goToLoraTrain() {
  void router.push({ name: 'lora_train' });
}

function openTrainingRun(taskId: string) {
  openLoraTrainingRun(router, String(taskId || ''));
}

async function deleteDownloadedLora(ul: { id?: string; name?: string }) {
  try {
    await confirm(
      $tt('download.deleteDownloadedLoraMessage', { name: ul.name || ul.id }),
      $tt('download.deleteDownloadedLoraTitle'),
      {
        confirmButtonText: $tt('download.deleteConfirmBtn'),
        cancelButtonText: $tt('download.deleteCancelBtn'),
        type: 'warning',
      }
    );
  } catch (e) {
    if (e !== 'cancel') console.error('Delete downloaded LoRA confirm failed:', e);
    return;
  }
  try {
    await api.loras.deleteUserAdapter(String(ul.id), true);
    await loadDownloadedLoras();
    toast.success($tt('download.downloadedLoraDeleted'));
  } catch (e: any) {
    toast.error(e?.message || String(e));
  }
}

async function deleteUserLora(ul: { id?: string; name?: string }) {
  try {
    await confirm(
      $tt('download.deleteUserLoraMessage', { name: ul.name || ul.id }),
      $tt('download.deleteUserLoraTitle'),
      {
        confirmButtonText: $tt('download.deleteConfirmBtn'),
        cancelButtonText: $tt('download.deleteCancelBtn'),
        type: 'warning',
      }
    );
  } catch (e) {
    if (e !== 'cancel') {
      console.error('Delete user LoRA confirm failed:', e);
    }
    return;
  }
  try {
    await api.loras.deleteUserAdapter(String(ul.id), true);
    await loadUserLoras();
    toast.success($tt('download.userLoraDeleted'));
  } catch (e: any) {
    toast.error(e?.message || String(e));
  }
}

watch(activeCategory, (cat) => {
  if (cat === 'trained_loras') void refreshUserLoras();
  if (cat === 'downloaded_loras') void refreshDownloadedLoras();
  if (cat === 'tools') void refreshUserMergedModels();
});

async function refreshUserMergedModels() {
  try {
    const data = await api.tools.listUserMergedZImageModels();
    userMergedModels.value = data.items || [];
  } catch {
    userMergedModels.value = [];
  }
}

async function loadDiskSpace() {
  try {
    const data = await api.settings.getDiskSpace();
    diskSpace.value = data as DiskSpaceData;
  } catch (e) {
    console.error('Failed to load disk space:', e);
  }
}

async function loadActiveDownloads() {
  try {
    const tasks = await api.download.listDownloads();
    if (!Array.isArray(tasks)) return;
    for (const task of tasks as any[]) {
      if (
        task.status === 'running' ||
        task.status === 'pending' ||
        task.status === 'paused' ||
        task.status === 'failed'
      ) {
        activeDownloads.value[task.id] = {
          name: formatDownloadDisplayName(task.filename || task.url || ''),
          progress: task.progress || 0,
          status: task.status,
          speed: '',
          error: task.error_message || '',
          total_size: task.total_size || 0,
          downloaded_size: task.downloaded_size || 0,
        };
        if (task.status === 'running') {
          connectProgressSSE(
            task.id,
            formatDownloadDisplayName(task.filename || task.url || '')
          );
        }
      }
    }
  } catch (e) {
    console.error('Failed to load active downloads:', e);
  }
}

/* ───── Utilities ───── */

function getDiskPercent(type: 'models' | 'loras'): number {
  if (!diskSpace.value || !diskSpace.value[type]) return 0;
  const info = diskSpace.value[type];
  if (!info || !info.exists || info.total === 0) return 0;
  return Math.round((info.size! / info.total) * 100);
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function isModelReady(modelId: string): boolean {
  return modelsStatus.value[modelId] || false;
}

function getModelName(modelId: string): string {
  const cfg = modelRegistry.value[modelId];
  if (!cfg) return modelId;
  const row: ModelRow = { id: modelId, ...cfg, ready: false };
  return $mn(row, modelId);
}

function canDownload(_model: ModelRow): boolean {
  return true;
}

function modelStatusTitle(modelId: string): string {
  const status = modelsDetailedStatus.value[modelId]?.status;
  if (status === 'ready') return $tt('download.readyTag');
  if (status === 'incomplete') return $tt('download.incompleteTag');
  return $tt('download.notDownloadedTag');
}

function modelCardSource(model: ModelRow): string {
  if (model.versions) return uniformDownloadSource(model);
  return model.source || '';
}

function getVersionStatus(modelId: string, versionKey: string): string {
  const detail = modelsDetailedStatus.value[modelId];
  if (!detail || !detail.versions) return 'missing';

  const verStatus = detail.versions[versionKey];
  if (!verStatus) return 'missing';

  if (verStatus.ready) return 'ready';

  const model = modelRegistry.value[modelId];
  const ver = model?.versions?.[versionKey];
  if (ver && ver.source_type === 'derived' && ver.from_version) {
    const parentVer = ver.from_version;
    const parentSt = parentVer ? detail.versions[parentVer] : null;
    if (!parentSt || !parentSt.ready) {
      return 'parent_missing';
    }
    return 'quantize';
  }

  return 'missing';
}

const BUNDLE_COMPONENT_ORDER = ['transformer', 'text_encoder', 'vae', 'tokenizer'] as const;

function versionBundleComponents(modelId: string, verKey: string) {
  const detail = modelsDetailedStatus.value[modelId];
  const verStatus = detail?.versions?.[verKey];
  const components = verStatus?.bundle_components?.components;
  if (!components || typeof components !== 'object') return [];
  return BUNDLE_COMPONENT_ORDER.filter((name) => name in components).map((name) => ({
    name,
    ok: Boolean(components[name]),
  }));
}

/* ───── Download / Quantize / Delete ───── */

async function downloadVersion(
  model: ModelRow,
  versionKey: string,
  opts: { uiLoadingKey?: string } = {}
) {
  const uiKey =
    opts.uiLoadingKey != null ? opts.uiLoadingKey : `${model.id}-${versionKey}`;
  if (downloadingModels.value[uiKey]) return;

  try {
    const tasks = (await api.download.listDownloads()) as Array<{
      id: string;
      status: string;
      model_name?: string;
      version?: string | null;
    }>;
    const active = tasks.find(
      (t) =>
        t.model_name === model.id &&
        (t.version ?? null) === (versionKey || null) &&
        (t.status === 'running' || t.status === 'paused' || t.status === 'failed')
    );
    if (active?.id) {
      const version = model.versions?.[versionKey];
      const label = version
        ? $mvn(model.id, model, version)
        : `${$mn(model, model.id)} ${versionKey}`;
      downloadingModels.value[uiKey] = true;
      if (active.status === 'failed') {
        await resumeDownload(active.id);
      } else {
        connectProgressSSE(active.id, label, uiKey);
      }
      return;
    }
  } catch {
    /* fall through to install */
  }

  downloadingModels.value[uiKey] = true;

  try {
    const version = model.versions?.[versionKey];
    const data = (await api.models.install(model.id, { version: versionKey })) as any;
    const label = version
      ? $mvn(model.id, model, version)
      : `${$mn(model, model.id)} ${versionKey}`;
    connectProgressSSE(data.task_id, label, uiKey);
  } catch (e: any) {
    console.error('Download failed:', e);
    toast.error($tt('download.downloadFailed', { msg: e.message }));
    delete downloadingModels.value[uiKey];
  }
}

async function quantizeVersion(model: ModelRow, versionKey: string) {
  const ver = model.versions?.[versionKey];
  const fromV = ver?.from_version;
  if (!fromV) {
    toast.error($tt('download.versionConfigError'));
    return;
  }
  const uiKey = `${model.id}-${versionKey}`;
  if (downloadingModels.value[uiKey]) return;

  downloadingModels.value[uiKey] = true;
  try {
    const data = (await api.download.startConvert({
      model_name: model.id,
      from_version: fromV,
      to_version: versionKey,
    })) as any;
    const taskId = data.task_id;
    const label = ver
      ? $mvn(model.id, model, ver)
      : `${$mn(model, model.id)} ${versionKey}`;
    connectConversionSSE(taskId, label, uiKey);
  } catch (e: any) {
    console.error('Quantize failed:', e);
    const detail = e?.response?.data?.detail;
    const msg =
      typeof detail === 'string'
        ? detail
        : Array.isArray(detail)
          ? detail.map((d: any) => d?.msg || String(d)).join('; ')
          : e.message || String(e);
    toast.error($tt('download.quantizeFailed', { msg }));
    delete downloadingModels.value[uiKey];
  }
}

async function deleteVersion(model: ModelRow, versionKey: string) {
  try {
    const version = model.versions?.[versionKey];
    await confirm(
      $tt('download.deleteConfirm', {
        name: version ? $mvn(model.id, model, version) : `${$mn(model, model.id)} ${versionKey}`,
      }),
      $tt('download.deleteConfirmTitle'),
      {
        confirmButtonText: $tt('download.deleteConfirmBtn'),
        cancelButtonText: $tt('download.deleteCancelBtn'),
        type: 'warning',
      }
    );
    const result = (await api.models.deleteVersion(model.id, versionKey)) as any;
    if (result.success) {
      toast.success(
        $tt('download.deletedMsg', {
          name: version ? $mvn(model.id, model, version) : `${$mn(model, model.id)} ${versionKey}`,
        })
      );
    } else {
      toast.error(result.error || $tt('download.deleteFailed'));
    }
    refreshStatus();
  } catch (e: any) {
    if (e !== 'cancel') {
      console.error('Delete failed:', e);
      toast.error($tt('download.deleteFailed') + ': ' + (e.message || e));
    }
  }
}

/* ───── SSE connections ───── */

function connectProgressSSE(
  taskId: string,
  name: string,
  downloadingKey: string | null = null
) {
  const dmKey = downloadingKey != null ? downloadingKey : taskId;
  if (sseConnections.value[taskId]) {
    sseConnections.value[taskId].close();
  }

  const eventSource = new EventSource(
    api.download.installProgressStreamUrl(taskId)
  );
  sseConnections.value[taskId] = eventSource;

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      activeDownloads.value[taskId] = {
        name,
        progress: data.progress || 0,
        status: data.status,
        speed: data.speed || '',
        error: data.error_message || '',
        total_size: data.total_size,
        downloaded_size: data.downloaded_size,
      };

      if (data.status === 'completed') {
        eventSource.close();
        delete sseConnections.value[taskId];
        setTimeout(() => {
          delete activeDownloads.value[taskId];
          delete downloadingModels.value[dmKey];
        }, 2000);
        toast.success($tt('download.downloadComplete', { name }));
        refreshStatus();
        void loadDownloadedLoras();
      } else if (data.status === 'failed') {
        eventSource.close();
        delete sseConnections.value[taskId];
        delete downloadingModels.value[dmKey];
        toast.error(
          $tt('download.downloadFailed', { name, msg: data.error_message })
        );
      } else if (data.status === 'cancelled') {
        eventSource.close();
        delete sseConnections.value[taskId];
        delete activeDownloads.value[taskId];
        delete downloadingModels.value[dmKey];
        toast.info($tt('download.downloadCancelled', { name }));
      }
    } catch (e) {
      console.error('SSE parse error:', e);
    }
  };

  eventSource.onerror = (error) => {
    console.error('SSE error:', error);
    eventSource.close();
    delete sseConnections.value[taskId];
    delete downloadingModels.value[dmKey];
  };
}

function connectConversionSSE(
  taskId: string,
  name: string,
  dmKey: string | null = null
) {
  const rowKey = dmKey != null ? dmKey : taskId;
  if (sseConnections.value[taskId]) {
    try {
      sseConnections.value[taskId].close();
    } catch (e) {}
  }

  const eventSource = new EventSource(
    api.download.convertProgressStreamUrl(taskId)
  );
  sseConnections.value[taskId] = eventSource;

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      activeDownloads.value[taskId] = {
        kind: 'quantize',
        name,
        progress: typeof data.progress === 'number' ? data.progress : 0,
        status: data.status || 'running',
        speed: '',
        total_size: 1,
        downloaded_size: 0,
        error: data.error_message || '',
      };

      if (data.status === 'completed') {
        eventSource.close();
        delete sseConnections.value[taskId];
        setTimeout(() => {
          delete activeDownloads.value[taskId];
          delete downloadingModels.value[rowKey];
        }, 2000);
        toast.success($tt('download.quantizeComplete', { name }));
        refreshStatus();
      } else if (data.status === 'failed') {
        eventSource.close();
        delete sseConnections.value[taskId];
        delete downloadingModels.value[rowKey];
        toast.error(
          $tt('download.quantizeFailed', { msg: data.error_message || '' })
        );
      } else if (data.status === 'cancelled') {
        eventSource.close();
        delete sseConnections.value[taskId];
        delete activeDownloads.value[taskId];
        delete downloadingModels.value[rowKey];
        toast.info($tt('download.genCancelled', { name }));
      }
    } catch (e) {
      console.error('Conversion SSE parse error:', e);
    }
  };

  eventSource.onerror = () => {
    eventSource.close();
    delete sseConnections.value[taskId];
    delete downloadingModels.value[rowKey];
  };
}

/* ───── Batch / Cancel / Resume ───── */

async function cancelDownload(taskId: string) {
  const item = activeDownloads.value[taskId];
  if (item && item.kind === 'quantize') {
    try {
      await api.download.cancelConversion(taskId);
    } catch (e) {
      console.error('Cancel conversion failed:', e);
    }
    return;
  }
  try {
    await api.download.cancel(taskId);
  } catch (e) {
    console.error('Cancel failed:', e);
  }
}

async function resumeDownload(taskId: string) {
  try {
    const item = activeDownloads.value[taskId];
    if (!item) return;

    await api.download.resume(taskId);

    item.status = 'running';
    connectProgressSSE(taskId, item.name);

    toast.success($tt('download.resumeStart', { name: item.name }));
  } catch (e: any) {
    console.error('Resume failed:', e);
    toast.error($tt('download.resumeFailed', { msg: e.message }));
  }
}

async function deleteDownload(taskId: string) {
  try {
    const item = activeDownloads.value[taskId];
    if (!item || item.kind !== 'quantize') {
      await api.download.delete(taskId);
    }
    delete activeDownloads.value[taskId];
    toast.success($tt('download.deleteSuccess'));
  } catch (e) {
    console.error('Delete download failed:', e);
    toast.error($tt('download.deleteFailed'));
  }
}

/* ───── Format / Helpers ───── */

function showImportDialog() {
  importModelName.value = '';
  importModelPath.value = '';
  importModelType.value = 'base';
  importDialogVisible.value = true;
}

async function importLocalModel() {
  if (!importModelName.value || !importModelPath.value) {
    toast.warning($tt('download.importWarn'));
    return;
  }

  importing.value = true;
  try {
    let importedModels: unknown[] = [];
    try {
      const raw = getItem(DQ_STORAGE.IMPORTED_MODELS);
      importedModels = raw ? JSON.parse(raw) : [];
      if (!Array.isArray(importedModels)) importedModels = [];
    } catch {
      importedModels = [];
    }
    importedModels.push({
      name: importModelName.value,
      path: importModelPath.value,
      type: importModelType.value,
      importedAt: new Date().toISOString(),
    });
    setItem(DQ_STORAGE.IMPORTED_MODELS, JSON.stringify(importedModels));

    toast.success(
      $tt('download.importSuccess', { name: importModelName.value })
    );
    importDialogVisible.value = false;
  } catch (e) {
    console.error('Import failed:', e);
    toast.error($tt('download.importFailed'));
  } finally {
    importing.value = false;
  }
}

function getModelInitials(model: ModelRow): string {
  return modelInitialsFromName($mn(model, model.id), 'M');
}

function getModelTypeTagType(type: string): string {
  const typeMap: Record<string, string> = {
    diffusion: 'primary',
    controlnet: 'warning',
    upscaler: 'success',
    tool: 'info',
    lora: 'danger',
    video: 'success',
  };
  return typeMap[type] || 'info';
}

function handleCategorySelect(index: string) {
  activeCategory.value = index;
}

function closeAllSSE() {
  Object.values(sseConnections.value).forEach((es) => {
    try {
      es.close();
    } catch (e) {}
  });
  sseConnections.value = {};
}

onBeforeRouteLeave(() => {
  closeAllSSE();
});

/* ───── Lifecycle ───── */

onMounted(() => {
  const jumpCategory = consumeStringDraft(DQ_STORAGE.MODELS_CATEGORY);
  if (jumpCategory) {
    activeCategory.value = jumpCategory;
  }
  loadModelRegistry();
  loadInstalled();
  loadDiskSpace();
  loadActiveDownloads();
  if (activeCategory.value === 'trained_loras') void refreshUserLoras();
  if (activeCategory.value === 'tools') void refreshUserMergedModels();
});

onUnmounted(() => {
  closeAllSSE();
});
</script>

<style scoped>
.models-page__merge-panel {
  margin-bottom: 20px;
}
.models-page__merged-list {
  margin-bottom: 20px;
  padding: 12px 16px;
  border: 1px solid var(--dq-border-subtle);
  border-radius: var(--dq-radius-md, 8px);
  background: var(--dq-surface-inset);
}
.models-page__merged-title {
  margin: 0 0 10px;
  font-size: var(--dq-font-size-body);
  font-weight: 600;
}
.models-page__merged-items {
  margin: 0;
  padding: 0;
  list-style: none;
}
.models-page__merged-item {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 6px 0;
  font-size: var(--dq-font-size-body);
  border-bottom: 1px solid var(--dq-border-subtle);
}
.models-page__merged-item:last-child {
  border-bottom: none;
}
.models-page__merged-id {
  color: var(--dq-label-secondary);
  font-family: var(--dq-font-mono, monospace);
  font-size: var(--dq-font-size-caption);
}
.trained-loras-page__empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  padding: 12px 0 24px;
}
</style>
