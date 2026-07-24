<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { toast } from '@/utils/feedback';
import { api } from '@/utils/api';
import { $tt } from '@/utils/i18n';
import { Search, Download, PictureFilled } from '@danqing/dq-shell';
import ModelVersionSourceBadge from '@/components/models/ModelVersionSourceBadge.vue';

export interface LoraSearchResult {
  id: string;
  source: string;
  name: string;
  description?: string;
  preview_url?: string;
  base_model_label?: string;
  hub_base_model?: string;
  tags?: string[];
  downloads?: number;
  likes?: number;
  nsfw?: boolean;
  creator?: string;
  repo_id?: string;
  filename?: string;
  download_url?: string;
  civitai_model_id?: number;
  civitai_version_id?: number;
  versions?: Array<{
    id: number;
    name: string;
    base_model?: string;
    files?: Array<{ name: string; download_url?: string; primary?: boolean }>;
    images?: Array<{ url?: string }>;
  }>;
}

const props = defineProps<{
  connectProgress: (taskId: string, name: string, downloadingKey?: string | null) => void;
}>();

const { t } = useI18n();

const baseModels = ref<Array<{ id: string; name: string }>>([]);
const searchScopeModel = ref('');
const searchQuery = ref('');
const searchSource = ref<'all' | 'modelscope' | 'huggingface' | 'civitai'>('modelscope');
const searching = ref(false);
const hasSearched = ref(false);
const searchResults = ref<LoraSearchResult[]>([]);
const selectedVersions = ref<Record<string, number>>({});
const downloadBaseByItem = ref<Record<string, string>>({});
const downloading = ref<Record<string, boolean>>({});
const previewLoadFailed = ref<Record<string, boolean>>({});

const sourceOptions = computed(() => [
  { value: 'all', label: t('download.loraSearchSourceAll') },
  { value: 'modelscope', label: t('download.sourceModelscope') },
  { value: 'huggingface', label: t('download.sourceHuggingface') },
  { value: 'civitai', label: t('download.sourceCivitai') },
]);

function hubBaseLabel(item: LoraSearchResult): string {
  if (item.source === 'civitai') {
    const versionId = selectedVersions.value[item.id];
    const version = (item.versions || []).find((v) => v.id === versionId) || item.versions?.[0];
    return (version?.base_model || item.hub_base_model || item.base_model_label || '').trim();
  }
  return (item.hub_base_model || item.base_model_label || '').trim();
}

async function loadBaseModels() {
  try {
    const rows = (await api.download.listLoraBaseModels()) as Array<{ id: string; name: string }>;
    baseModels.value = rows || [];
    if (!searchScopeModel.value && baseModels.value.length) {
      const preferred =
        baseModels.value.find((m) => m.id === 'flux1-dev') ||
        baseModels.value.find((m) => m.id === 'z-image-turbo') ||
        baseModels.value[0];
      searchScopeModel.value = preferred.id;
    }
  } catch (e) {
    console.error('Failed to load LoRA base models:', e);
  }
}

function formatNumber(num: number): string {
  if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(1)}M`;
  if (num >= 1_000) return `${(num / 1_000).toFixed(1)}K`;
  return String(num);
}

function previewUrl(item: LoraSearchResult): string {
  if (item.preview_url) return item.preview_url;
  const versionId = selectedVersions.value[item.id];
  const version = (item.versions || []).find((v) => v.id === versionId) || item.versions?.[0];
  return version?.images?.[0]?.url || '';
}

function onPreviewError(itemId: string) {
  previewLoadFailed.value = { ...previewLoadFailed.value, [itemId]: true };
}

function itemSubtitle(item: LoraSearchResult): string {
  const parts: string[] = [];
  if (item.repo_id) parts.push(item.repo_id);
  else if (item.creator) parts.push(item.creator);
  return parts.join(' · ');
}

function repoPageUrl(item: LoraSearchResult): string {
  if (item.source === 'modelscope' && item.repo_id) {
    return `https://www.modelscope.cn/models/${encodeURI(item.repo_id)}`;
  }
  if (item.source === 'huggingface' && item.repo_id) {
    return `https://huggingface.co/${encodeURI(item.repo_id)}`;
  }
  if (item.source === 'civitai' && item.civitai_model_id) {
    return `https://civitai.com/models/${item.civitai_model_id}`;
  }
  return '';
}

function repoLinkLabel(item: LoraSearchResult): string {
  if (item.repo_id) return item.repo_id;
  if (item.source === 'civitai' && item.civitai_model_id) {
    return `civitai:${item.civitai_model_id}`;
  }
  return item.creator || item.name;
}

function initDownloadBases(items: LoraSearchResult[]) {
  const next: Record<string, string> = { ...downloadBaseByItem.value };
  for (const item of items) {
    next[item.id] = next[item.id] || searchScopeModel.value;
  }
  downloadBaseByItem.value = next;
}

function itemBindBase(itemId: string): string {
  return downloadBaseByItem.value[itemId] || searchScopeModel.value;
}

function setItemBindBase(itemId: string, value: string) {
  downloadBaseByItem.value = { ...downloadBaseByItem.value, [itemId]: value };
}

async function searchLoras() {
  if (!searchScopeModel.value) {
    toast.warning($tt('download.loraSearchSelectScope'));
    return;
  }
  if (searching.value) return;

  searching.value = true;
  hasSearched.value = true;
  previewLoadFailed.value = {};

  try {
    const data = (await api.download.searchLoras({
      q: searchQuery.value,
      base_model: searchScopeModel.value,
      source: searchSource.value,
      limit: '500',
    })) as { items?: LoraSearchResult[] };
    const items = data.items || [];
    searchResults.value = items;
    initDownloadBases(items);
    const nextVersions: Record<string, number> = { ...selectedVersions.value };
    for (const item of items) {
      if (item.source === 'civitai' && item.versions?.length && !nextVersions[item.id]) {
        nextVersions[item.id] = item.versions[0].id;
      }
    }
    selectedVersions.value = nextVersions;
  } catch (e) {
    console.error('LoRA search failed:', e);
    toast.error($tt('download.searchFailed'));
  } finally {
    searching.value = false;
  }
}

async function downloadItem(item: LoraSearchResult) {
  const bindBase = itemBindBase(item.id);
  if (!bindBase) {
    toast.warning($tt('download.loraSearchSelectBindBase'));
    return;
  }
  downloading.value = { ...downloading.value, [item.id]: true };
  try {
    const common = {
      base_model: bindBase,
      display_name: item.name,
    };
    let body: Record<string, unknown> = { source: item.source, ...common };
    if (item.source === 'civitai') {
      const versionId = selectedVersions.value[item.id] || item.civitai_version_id;
      const version = (item.versions || []).find((v) => v.id === versionId);
      const primaryFile =
        version?.files?.find((f) => f.primary) || version?.files?.[0] || null;
      body = {
        source: 'civitai',
        url: primaryFile?.download_url || item.download_url,
        filename: primaryFile?.name || item.filename,
        civitai_version_id: versionId,
        ...common,
      };
      if (!body.url && !body.civitai_version_id) {
        toast.error($tt('download.noDownloadableFile'));
        return;
      }
    } else if (item.source === 'huggingface' || item.source === 'modelscope') {
      body = {
        source: item.source,
        repo_id: item.repo_id,
        filename: item.filename || undefined,
        ...common,
      };
      if (!body.repo_id) {
        toast.error($tt('download.noDownloadableFile'));
        return;
      }
    } else {
      toast.error($tt('download.noDownloadableFile'));
      return;
    }

    const data = (await api.download.startLoraHubDownload(body)) as {
      task_id?: string;
      error_message?: string;
    };
    if (!data?.task_id) {
      toast.error($tt('download.downloadFailed', { msg: data?.error_message || 'missing task_id' }));
      return;
    }
    props.connectProgress(data.task_id, item.name, item.id);
    toast.info($tt('download.downloadStarted', { name: item.name }));
  } catch (e: unknown) {
    const msg =
      (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
      (e instanceof Error ? e.message : String(e));
    console.error('LoRA download failed:', e);
    toast.error($tt('download.downloadFailed', { msg }));
  } finally {
    downloading.value = { ...downloading.value, [item.id]: false };
  }
}

onMounted(() => {
  void loadBaseModels();
});

watch(searchScopeModel, () => {
  if (hasSearched.value) void searchLoras();
});
</script>

<template>
  <div class="lora-search-page">
    <div class="page-header models-page__page-header lora-search-page__header">
      <h2 class="page-title models-page__category-title">
        <DqIcon class="models-page__title-icon" aria-hidden="true"><Search /></DqIcon>
        <span>{{ $t('download.loraSearch') }}</span>
      </h2>
    </div>

    <DqSurfaceCard class="studio-surface-card lora-search-page__toolbar">
      <div class="lora-search-page__controls">
        <div class="lora-search-page__field">
          <span class="lora-search-page__label">{{ $t('download.loraSearchScope') }}</span>
          <DqSelect
            v-model="searchScopeModel"
            :placeholder="$t('download.loraSearchScope')"
          >
            <DqOption
              v-for="model in baseModels"
              :key="model.id"
              :label="model.name"
              :value="model.id"
            />
          </DqSelect>
        </div>

        <div class="lora-search-page__field lora-search-page__field--source">
          <span class="lora-search-page__label">{{ $t('download.loraSearchSource') }}</span>
          <DqSelect v-model="searchSource">
            <DqOption
              v-for="opt in sourceOptions"
              :key="opt.value"
              :label="opt.label"
              :value="opt.value"
            />
          </DqSelect>
        </div>

        <div class="lora-search-page__field lora-search-page__field--query">
          <span class="lora-search-page__label">{{ $t('download.search') }}</span>
          <div class="lora-search-page__query-row">
            <DqInput
              v-model="searchQuery"
              class="lora-search-page__query-input"
              :placeholder="$t('download.loraSearchPlaceholder')"
              clearable
              @keyup.enter="searchLoras"
            >
              <template #prefix>
                <DqIcon><Search /></DqIcon>
              </template>
            </DqInput>
            <DqButton
              type="primary"
              size="sm"
              class="models-toolbar-btn models-toolbar-btn--primary lora-search-page__submit"
              :loading="searching"
              :disabled="!searchScopeModel"
              @click="searchLoras"
            >
              <DqIcon class="models-toolbar-btn__icon"><Search /></DqIcon>
              <span class="models-toolbar-btn__label">{{ $t('download.search') }}</span>
            </DqButton>
          </div>
        </div>
      </div>
    </DqSurfaceCard>

    <DqSurfaceCard v-if="searchResults.length > 0" class="studio-surface-card lora-search-page__results">
      <div class="lora-search-list" role="list">
        <article
          v-for="item in searchResults"
          :key="item.id"
          class="lora-search-row"
          role="listitem"
        >
          <div class="lora-search-row__thumb">
            <img
              v-if="previewUrl(item) && !previewLoadFailed[item.id]"
              :src="previewUrl(item)"
              loading="lazy"
              :alt="item.name"
              @error="onPreviewError(item.id)"
            />
            <div v-else class="lora-search-row__thumb-placeholder">
              <DqIcon><PictureFilled /></DqIcon>
            </div>
          </div>

          <div class="lora-search-row__main">
            <div class="lora-search-row__title">{{ item.name }}</div>
            <div v-if="repoPageUrl(item)" class="lora-search-row__meta">
              <a
                class="lora-search-row__repo-link"
                :href="repoPageUrl(item)"
                target="_blank"
                rel="noopener noreferrer"
                :title="$t('download.loraSearchOpenRepo')"
              >
                {{ repoLinkLabel(item) }}
              </a>
            </div>
            <div v-else-if="itemSubtitle(item)" class="lora-search-row__meta">{{ itemSubtitle(item) }}</div>
            <div v-if="item.description" class="lora-search-row__desc">{{ item.description }}</div>
            <div class="lora-search-row__tags">
              <DqTag
                v-if="hubBaseLabel(item)"
                size="sm"
                type="warning"
                effect="plain"
              >
                {{ $t('download.loraSearchHubBase', { name: hubBaseLabel(item) }) }}
              </DqTag>
              <DqTag v-else size="sm" type="info" effect="plain">
                {{ $t('download.loraSearchHubBaseUnknown') }}
              </DqTag>
              <DqTag
                v-for="tag in item.tags || []"
                :key="`${item.id}-${tag}`"
                size="sm"
                effect="plain"
              >
                {{ tag }}
              </DqTag>
              <ModelVersionSourceBadge :source="item.source" />
              <DqTag v-if="item.nsfw" type="danger" size="sm">{{ $t('download.nsfwTag') }}</DqTag>
              <DqTag v-if="item.likes" type="success" size="sm">
                {{ $t('download.loraSearchLikes', { count: formatNumber(item.likes) }) }}
              </DqTag>
              <DqTag type="info" size="sm">
                <DqIcon><Download /></DqIcon>
                {{ formatNumber(item.downloads || 0) }}
              </DqTag>
            </div>
          </div>

          <div class="lora-search-row__actions">
            <DqSelect
              v-if="item.source === 'civitai' && item.versions?.length"
              v-model="selectedVersions[item.id]"
              class="lora-search-row__version"
              :placeholder="$t('download.selectVersion')"
            >
              <DqOption
                v-for="v in item.versions"
                :key="v.id"
                :label="v.name"
                :value="v.id"
              />
            </DqSelect>
            <label class="lora-search-row__bind">
              <span class="lora-search-row__bind-label">{{ $t('download.loraSearchBindBase') }}</span>
              <DqSelect
                :model-value="itemBindBase(item.id)"
                class="lora-search-row__bind-select"
                :placeholder="$t('download.loraSearchBindBase')"
                @update:model-value="setItemBindBase(item.id, $event)"
              >
                <DqOption
                  v-for="model in baseModels"
                  :key="`${item.id}-${model.id}`"
                  :label="model.name"
                  :value="model.id"
                />
              </DqSelect>
            </label>
            <DqButton
              size="sm"
              class="model-ver-btn model-ver-btn--download"
              :loading="downloading[item.id]"
              @click="downloadItem(item)"
            >
              <DqIcon class="model-ver-btn__icon"><Download /></DqIcon>
              <span class="model-ver-btn__label">{{ $t('download.download_') }}</span>
            </DqButton>
          </div>
        </article>
      </div>
    </DqSurfaceCard>

    <DqSurfaceCard v-else-if="!searching && hasSearched" class="studio-surface-card">
      <DqEmpty :description="$t('download.noResults')" />
    </DqSurfaceCard>
    <p class="lora-search-page__footer-hint">{{ $t('download.loraSearchViewDownloadedHint') }}</p>
  </div>
</template>
