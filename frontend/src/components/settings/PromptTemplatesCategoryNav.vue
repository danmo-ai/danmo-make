<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import {
  Grid,
  PictureFilled,
  VideoCamera,
  Picture,
  Document,
  Brush,
  ArrowRight,
  ZoomIn,
} from '@danqing/dq-shell';

const props = defineProps<{
  activeCategory: string;
  activeAction: string;
  totalCount: number;
}>();

const emit = defineEmits<{
  selectCategory: [category: string];
  selectAction: [action: string];
}>();

const { t } = useI18n();

type NavEntry =
  | { kind: 'divider' }
  | {
      kind: 'item';
      id: string;
      icon: object;
      label: string;
      count?: number;
      isAction?: boolean;
    };

const mediaEntries = computed<NavEntry[]>(() => [
  {
    kind: 'item',
    id: 'all',
    icon: Grid,
    label: t('settings.allTemplates'),
    count: props.totalCount,
  },
  {
    kind: 'item',
    id: 'image',
    icon: PictureFilled,
    label: t('settings.presetMediaImage'),
  },
  {
    kind: 'item',
    id: 'video',
    icon: VideoCamera,
    label: t('settings.presetMediaVideo'),
  },
]);

const ACTION_ICONS: Record<string, object> = {
  create: Picture,
  rewrite: Document,
  retouch: Brush,
  extend: ArrowRight,
  upscale: ZoomIn,
  animate: VideoCamera,
};

const actionEntries = computed<NavEntry[]>(() => [
  { kind: 'item', id: 'create', icon: ACTION_ICONS.create, label: t('action.image.create'), isAction: true },
  { kind: 'item', id: 'rewrite', icon: ACTION_ICONS.rewrite, label: t('action.image.rewrite'), isAction: true },
  { kind: 'item', id: 'retouch', icon: ACTION_ICONS.retouch, label: t('action.image.retouch'), isAction: true },
  { kind: 'item', id: 'extend', icon: ACTION_ICONS.extend, label: t('action.image.extend'), isAction: true },
  { kind: 'item', id: 'upscale', icon: ACTION_ICONS.upscale, label: t('action.image.upscale'), isAction: true },
  { kind: 'item', id: 'animate', icon: ACTION_ICONS.animate, label: t('action.video.animate'), isAction: true },
]);

function isActive(entry: NavEntry): boolean {
  if (entry.kind === 'divider') return false;
  if (entry.isAction) {
    return props.activeAction === entry.id;
  }
  return props.activeCategory === entry.id && props.activeAction === '';
}

function handleClick(entry: NavEntry) {
  if (entry.kind === 'divider') return;
  if (entry.isAction) {
    emit('selectAction', entry.id);
  } else {
    emit('selectCategory', entry.id);
  }
}
</script>

<template>
  <nav class="dq-download-menu templates-page__menu" role="navigation" :aria-label="$t('settings.promptTemplates')">
    <template v-for="(entry, idx) in mediaEntries" :key="entry.kind === 'divider' ? `div-${idx}` : entry.id">
      <button
        v-if="entry.kind === 'item'"
        type="button"
        class="dq-download-menu__item"
        :class="{ 'is-active': isActive(entry) }"
        @click="handleClick(entry)"
      >
        <DqIcon class="dq-download-menu__icon"><component :is="entry.icon" /></DqIcon>
        <span class="dq-download-menu__label">{{ entry.label }}</span>
        <DqTag
          v-if="entry.count !== undefined"
          size="sm"
          type="info"
          class="dq-menu-end-tag"
        >
          {{ entry.count }}
        </DqTag>
      </button>
    </template>

    <hr class="models-page__menu-divider" />

    <div class="templates-page__menu-section-title">
      {{ $t('settings.filterByAction') }}
    </div>

    <template v-for="(entry, idx) in actionEntries" :key="entry.kind === 'divider' ? `div-${idx}` : entry.id">
      <button
        v-if="entry.kind === 'item'"
        type="button"
        class="dq-download-menu__item"
        :class="{ 'is-active': isActive(entry) }"
        @click="handleClick(entry)"
      >
        <DqIcon class="dq-download-menu__icon"><component :is="entry.icon" /></DqIcon>
        <span class="dq-download-menu__label">{{ entry.label }}</span>
      </button>
    </template>
  </nav>
</template>
