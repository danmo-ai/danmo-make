<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import {
  Aim,
  Document,
  Download,
  FolderChecked,
  Grid,
  Headset,
  MagicStick,
  PictureFilled,
  Search,
  Tools,
  VideoCamera,
  ZoomIn,
} from '@danqing/dq-shell';

const props = defineProps<{
  activeCategory: string;
  totalModelCount: number;
  activeDownloadCount: number;
}>();

const emit = defineEmits<{
  select: [category: string];
}>();

const { t } = useI18n();

type NavEntry =
  | { kind: 'section'; label: string }
  | { kind: 'divider' }
  | {
      kind: 'item';
      id: string;
      icon: object | string;
      label: string;
      count?: number;
      badge?: number;
      badgeType?: 'info' | 'primary';
    };

const entries = computed<NavEntry[]>(() => [
  {
    kind: 'item',
    id: 'all',
    icon: Grid,
    label: t('download.allModels'),
    count: props.totalModelCount,
    badgeType: 'info',
  },
  { kind: 'section', label: t('models.navBaseModels') },
  { kind: 'item', id: 'image_models', icon: PictureFilled, label: t('download.imageModels') },
  { kind: 'item', id: 'video_models', icon: VideoCamera, label: t('download.videoModels') },
  { kind: 'item', id: 'music_models', icon: Headset, label: t('download.audioModels') },
  { kind: 'item', id: 'llm_models', icon: Document, label: t('download.llmModels') },
  { kind: 'item', id: 'vlm_models', icon: PictureFilled, label: t('download.vlmModels') },
  { kind: 'section', label: t('models.navAdapters') },
  { kind: 'item', id: 'controlnets', icon: Aim, label: t('download.controlNet') },
  { kind: 'item', id: 'upscalers', icon: ZoomIn, label: t('download.upscalers') },
  { kind: 'item', id: 'loras', icon: MagicStick, label: t('download.loraModels') },
  { kind: 'item', id: 'lora_search', icon: Search, label: t('download.loraSearch') },
  { kind: 'item', id: 'downloaded_loras', icon: Download, label: t('download.downloadedLoras') },
  { kind: 'item', id: 'trained_loras', icon: 'Wand2', label: t('download.myTrainedLoras') },
  { kind: 'section', label: t('models.navTools') },
  { kind: 'item', id: 'tools', icon: Tools, label: t('download.tools') },
  { kind: 'divider' },
  { kind: 'section', label: t('models.navStatus') },
  {
    kind: 'item',
    id: 'downloading',
    icon: Download,
    label: t('download.downloadingTab'),
    badge: props.activeDownloadCount,
    badgeType: 'primary',
  },
  { kind: 'item', id: 'installed', icon: FolderChecked, label: t('download.installed') },
]);
</script>

<template>
  <nav class="dq-download-menu models-page__menu" role="navigation" :aria-label="$t('download.modelLibrary')">
    <template v-for="(entry, idx) in entries" :key="entry.kind === 'divider' ? `div-${idx}` : entry.kind === 'section' ? `sec-${idx}-${entry.label}` : entry.id">
      <hr v-if="entry.kind === 'divider'" class="models-page__menu-divider" />
      <div v-else-if="entry.kind === 'section'" class="models-page__menu-section">
        {{ entry.label }}
      </div>
      <button
        v-else
        type="button"
        class="dq-download-menu__item"
        :class="{ 'is-active': activeCategory === entry.id }"
        @click="emit('select', entry.id)"
      >
        <DqIcon class="dq-download-menu__icon"><component :is="entry.icon" /></DqIcon>
        <span class="dq-download-menu__label">{{ entry.label }}</span>
        <DqTag
          v-if="entry.count !== undefined"
          size="sm"
          :type="entry.badgeType ?? 'info'"
          class="dq-menu-end-tag"
        >
          {{ entry.count }}
        </DqTag>
        <DqTag
          v-else-if="entry.badge && entry.badge > 0"
          size="sm"
          :type="entry.badgeType ?? 'primary'"
          class="dq-menu-end-tag"
        >
          {{ entry.badge }}
        </DqTag>
      </button>
    </template>
  </nav>
</template>
