<template>
  <div
    class="studio-card studio-card--group"
    :class="{ 'studio-card--selected': selected }"
    @mouseenter="isHovered = true"
    @mouseleave="isHovered = false"
    @click="$emit('click', $event)"
  >
    <div class="studio-card__media">
      <template v-if="coverItem">
        <img
          v-if="!thumbFailed"
          :src="coverItem.thumbnail"
          :alt="group.title"
          loading="lazy"
          @error="thumbFailed = true"
        />
        <div v-else class="studio-card__fallback">
          <DqIcon :size="36"><Picture /></DqIcon>
        </div>
      </template>
      <div v-else class="studio-card__fallback">
        <DqIcon :size="36"><Picture /></DqIcon>
      </div>

      <span v-if="assetCount > 1" class="studio-card__group-badge">{{ countLabel }}</span>

      <div v-if="isHovered" class="studio-card__overlay">
        <div class="studio-card__overlay-bg" />
        <div class="studio-card__overlay-info">
          <span class="studio-card__overlay-res">{{ countLabel }}</span>
        </div>
      </div>
    </div>

    <div class="studio-card__footer">
      <span class="studio-card__prompt" :title="group.title">
        {{ group.title }}
      </span>
    </div>
  </div>
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
const isHovered = ref(false);
const thumbFailed = ref(false);

const coverItem = computed((): GalleryItem | null => {
  const items = props.group.preview_assets || [];
  return (items[0] as GalleryItem | undefined) ?? null;
});

const assetCount = computed(() => props.group.asset_count || 0);

const countLabel = computed(() => $t('gallery.groupAssetCount', { count: assetCount.value }));
</script>

<style scoped>
.studio-card {
  cursor: pointer;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.studio-card--selected .studio-card__media {
  border-color: var(--dq-accent);
  box-shadow: 0 0 0 2px color-mix(in srgb, var(--dq-accent) 28%, transparent);
}

.studio-card__media {
  position: relative;
  aspect-ratio: 1 / 1;
  overflow: hidden;
  border-radius: var(--dq-radius-group);
  border: 0.5px solid var(--dq-glass-border);
  background: var(--dq-surface-inset);
  box-shadow: var(--dq-shadow-sm);
  transition: box-shadow 0.2s ease, border-color 0.2s ease;
}

.studio-card:hover .studio-card__media {
  border-color: var(--dq-glass-border-strong);
  box-shadow: var(--dq-shadow-md);
}

.studio-card__media img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.studio-card__fallback {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--dq-label-tertiary);
}

.studio-card__group-badge {
  position: absolute;
  right: 8px;
  bottom: 8px;
  z-index: 2;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  color: var(--dq-color-white);
  background: rgba(0, 0, 0, 0.55);
  pointer-events: none;
}

.studio-card__overlay {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
  padding: 10px;
}

.studio-card__overlay-bg {
  position: absolute;
  inset: 0;
  background: linear-gradient(
    to bottom,
    transparent 0%,
    transparent 50%,
    var(--dq-overlay-gradient-end) 100%
  );
  pointer-events: none;
}

.studio-card__overlay-info {
  position: relative;
  z-index: 2;
}

.studio-card__overlay-res {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  color: var(--dq-color-white);
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.4);
}

.studio-card__footer {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-height: 1.2em;
}

.studio-card__prompt {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
