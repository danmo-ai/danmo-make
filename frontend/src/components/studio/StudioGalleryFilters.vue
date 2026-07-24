<template>
  <DqSelect
    :model-value="filterTime"
    size="sm"
    class="studio-gallery-filters__time"
    @update:model-value="$emit('update:filterTime', $event)"
  >
    <DqOption
      v-for="opt in timeOptions"
      :key="opt.value"
      :label="opt.label"
      :value="opt.value"
    />
  </DqSelect>

  <DqSelect
    v-if="showModelFilter !== false"
    :model-value="filterModels"
    size="sm"
    multiple
    collapse-tags
    class="studio-gallery-filters__models"
    :placeholder="$t('gallery.filterModel')"
    @update:model-value="$emit('update:filterModels', $event)"
  >
    <DqOption v-for="m in modelOptions" :key="m" :label="m" :value="m" />
  </DqSelect>

  <DqInput
    v-if="showSearch"
    :model-value="searchText"
    size="sm"
    class="studio-gallery-filters__search"
    :placeholder="searchPlaceholder || $t('gallery.searchPlaceholder')"
    clearable
    @update:model-value="$emit('update:searchText', $event)"
  />

  <DqButton
    type="primary"
    size="sm"
    class="studio-gallery-filters__compose"
    @click="$emit('open-composer')"
  >
    <span v-if="composerBusy" class="studio-gallery-filters__compose-dot" aria-hidden="true" />
    {{ $t('studio.openComposer') }}
  </DqButton>

  <div class="studio-gallery-filters__spacer" />

  <div
    v-if="viewMode && supportsCanvas !== false && !(selectedCount ?? 0)"
    class="studio-gallery-filters__view-mode"
    :aria-label="$t('gallery.viewMode')"
  >
    <span class="studio-gallery-filters__view-label">{{ $t('gallery.viewMode') }}</span>
    <StudioViewModeSwitch
      :model-value="viewMode"
      @update:model-value="onViewModeChange"
    />
  </div>

  <div class="studio-gallery-filters__actions">
    <StudioGalleryBatchActions
      v-if="!viewMode || viewMode === 'grid'"
      :selection-mode="selectionMode"
      :selected-count="selectedCount ?? 0"
      :all-selected="allSelected"
      @toggle-selection-mode="$emit('toggle-selection-mode')"
      @select-all="$emit('select-all')"
      @batch-delete="$emit('batch-delete')"
      @batch-train-lora="$emit('batch-train-lora')"
      @clear-selection="$emit('clear-selection')"
    />
    <StudioCanvasSessionControls
      v-else-if="canvasMedia"
      :media="canvasMedia"
      @composer-restore="$emit('composer-restore', $event)"
    />
  </div>

  <DqIconButton
    type="text"
    size="sm"
    :label="$t('common.refresh')"
    class="studio-gallery-filters__refresh"
    @click="$emit('refresh')"
  >
    <DqIcon><Refresh /></DqIcon>
  </DqIconButton>
</template>

<script setup lang="ts">
import { Refresh } from '@danqing/dq-shell';
import StudioViewModeSwitch from '@/components/studio/StudioViewModeSwitch.vue';
import StudioGalleryBatchActions from '@/components/studio/StudioGalleryBatchActions.vue';
import StudioCanvasSessionControls from '@/components/studio/StudioCanvasSessionControls.vue';
import type { CanvasMedia } from '@/composables/useCanvasStore';

const props = withDefaults(
  defineProps<{
    filterTime: string;
    filterModels: string[];
    searchText?: string;
    timeOptions: { label: string; value: string }[];
    modelOptions: string[];
    selectionMode?: boolean;
    selectedCount?: number;
    allSelected?: boolean;
    viewMode?: 'grid' | 'canvas';
    supportsCanvas?: boolean;
    canvasMedia?: CanvasMedia;
    composerBusy?: boolean;
    showModelFilter?: boolean;
    showSearch?: boolean;
    searchPlaceholder?: string;
  }>(),
  {
    showSearch: true,
  },
);

const emit = defineEmits<{
  (e: 'update:filterTime', value: string): void;
  (e: 'update:filterModels', value: string[]): void;
  (e: 'update:searchText', value: string): void;
  (e: 'refresh'): void;
  (e: 'toggle-selection-mode'): void;
  (e: 'select-all'): void;
  (e: 'batch-delete'): void;
  (e: 'batch-train-lora'): void;
  (e: 'clear-selection'): void;
  (e: 'update:viewMode', value: 'grid' | 'canvas'): void;
  (e: 'composer-restore', snapshot: import('@/types').CanvasComposerSnapshot): void;
  (e: 'open-composer'): void;
}>();

function onViewModeChange(mode: 'grid' | 'canvas') {
  if (props.viewMode === mode) return;
  emit('update:viewMode', mode);
}
</script>

<style scoped>
.studio-gallery-filters__time {
  width: 112px;
  flex-shrink: 0;
}

.studio-gallery-filters__models {
  flex: 1 1 auto;
  min-width: 0;
  max-width: 280px;
}

.studio-gallery-filters__search {
  flex: 1 1 auto;
  min-width: 0;
  max-width: 320px;
}

.studio-gallery-filters__search :deep(.dq-input) {
  font-size: var(--dq-font-size-caption);
}

.studio-gallery-filters__compose {
  flex-shrink: 0;
  position: relative;
}

.studio-gallery-filters__compose-dot {
  position: absolute;
  top: 4px;
  right: 4px;
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--dq-accent);
}

.studio-gallery-filters__spacer {
  flex: 1 1 auto;
  min-width: 8px;
}

.studio-gallery-filters__actions {
  display: flex;
  align-items: center;
  flex: 0 0 auto;
  flex-shrink: 0;
  justify-content: flex-end;
}

.studio-gallery-filters__refresh {
  flex-shrink: 0;
}

.studio-gallery-filters__view-mode {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
}

.studio-gallery-filters__view-label {
  font-size: var(--dq-font-size-caption);
  font-weight: 500;
  color: var(--dq-label-secondary);
  white-space: nowrap;
}
</style>
