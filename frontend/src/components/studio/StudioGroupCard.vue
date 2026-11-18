<template>
  <article class="studio-card studio-card--group" :class="{ 'studio-card--selected': selected }">
    <div class="studio-card__media">
      <button
        type="button"
        class="studio-card__open"
        :aria-label="group.title"
        @click="$emit('click', $event)"
      />
      <img
        v-if="coverItem && !thumbFailed"
        :src="coverItem.thumbnail"
        :alt="group.title"
        loading="lazy"
        @error="thumbFailed = true"
      />
      <div v-else class="studio-card__fallback"><DqIcon :size="32"><Picture /></DqIcon></div>
      <span v-if="assetCount > 1" class="studio-card__group-badge">{{ countLabel }}</span>
    </div>
    <div class="studio-card__footer">
      <span class="studio-card__prompt" :title="group.title">{{ group.title }}</span>
      <span class="studio-card__meta">{{ countLabel }}</span>
    </div>
  </article>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import { Picture } from '@danqing/dq-shell';
import type { GalleryGroup, GalleryItem } from '@/types';

const props = defineProps<{
  group: GalleryGroup;
  selected?: boolean;
}>();

defineEmits<{
  (e: 'click', event: MouseEvent): void;
}>();

const { t: $t } = useI18n();
const thumbFailed = ref(false);
const coverItem = computed((): GalleryItem | null =>
  (props.group.preview_assets?.[0] as GalleryItem | undefined) ?? null,
);
const assetCount = computed(() => props.group.asset_count || 0);
const countLabel = computed(() => $t('gallery.groupAssetCount', { count: assetCount.value }));
</script>

<style scoped>
.studio-card {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.studio-card__media {
  position: relative;
  aspect-ratio: 1 / 1;
  overflow: hidden;
  border: 1px solid var(--dq-border-subtle);
  border-radius: var(--dq-radius-control);
  background: var(--dq-surface-inset);
  transition: border-color 140ms ease, box-shadow 140ms ease;
}

.studio-card:hover .studio-card__media,
.studio-card:focus-within .studio-card__media {
  border-color: var(--dq-border);
  box-shadow: var(--dq-shadow-sm);
}

.studio-card--selected .studio-card__media {
  border-color: var(--dq-accent);
  box-shadow: 0 0 0 2px color-mix(in srgb, var(--dq-accent) 20%, transparent);
}

.studio-card__open {
  position: absolute;
  inset: 0;
  z-index: 1;
  width: 100%;
  height: 100%;
  padding: 0;
  border: 0;
  background: transparent;
  cursor: pointer;
}

.studio-card__media img {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.studio-card__fallback {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  color: var(--dq-label-tertiary);
}

.studio-card__group-badge {
  position: absolute;
  right: 8px;
  bottom: 8px;
  z-index: 2;
  padding: 3px 8px;
  border: 1px solid var(--dq-border-subtle);
  border-radius: var(--dq-radius-pill);
  color: var(--dq-label-primary);
  background: var(--dq-bg-base);
  box-shadow: var(--dq-shadow-sm);
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  pointer-events: none;
}

.studio-card__footer {
  display: flex;
  flex-direction: column;
  gap: 3px;
  padding: 0 2px;
}

.studio-card__prompt,
.studio-card__meta {
  overflow: hidden;
  font-size: var(--dq-font-size-caption);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.studio-card__prompt {
  color: var(--dq-label-primary);
}

.studio-card__meta {
  color: var(--dq-label-tertiary);
}
</style>
