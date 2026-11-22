<template>
  <div class="studio-gallery-filters" :class="{ 'is-selecting': selectedCount }">
    <template v-if="selectedCount">
      <StudioGalleryBatchActions
        :selection-mode="selectionMode"
        :selected-count="selectedCount"
        :all-selected="allSelected"
        @toggle-selection-mode="$emit('toggle-selection-mode')"
        @select-all="$emit('select-all')"
        @batch-delete="$emit('batch-delete')"
        @batch-train-lora="$emit('batch-train-lora')"
        @clear-selection="$emit('clear-selection')"
      />
    </template>

    <template v-else>
      <div class="studio-gallery-filters__query">
        <DqSelect
          :model-value="filterTime"
          size="sm"
          variant="ghost"
          class="studio-gallery-filters__time"
          :aria-label="$t('gallery.filterDate')"
          @update:model-value="$emit('update:filterTime', $event)"
        >
          <DqOption v-for="opt in timeOptions" :key="opt.value" :label="opt.label" :value="opt.value" />
        </DqSelect>

        <DqSelect
          v-if="showModelFilter !== false"
          :model-value="filterModels"
          size="sm"
          variant="ghost"
          multiple
          collapse-tags
          class="studio-gallery-filters__models"
          :placeholder="$t('gallery.filterModel')"
          :aria-label="$t('gallery.filterModel')"
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
        >
          <template #prefix><DqIcon><Search /></DqIcon></template>
        </DqInput>
      </div>

      <div class="studio-gallery-filters__spacer" />

      <div class="studio-gallery-filters__actions">
        <div
          v-if="viewMode && supportsCanvas !== false"
          class="studio-gallery-filters__view-mode"
          :aria-label="$t('gallery.viewMode')"
        >
          <StudioViewModeSwitch
            :model-value="viewMode"
            @update:model-value="onViewModeChange"
          />
        </div>

        <StudioGalleryBatchActions
          v-if="!viewMode || viewMode === 'grid'"
          :selection-mode="selectionMode"
          :selected-count="0"
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

        <DqIconButton
          type="text"
          size="sm"
          :label="$t('common.refresh')"
          class="studio-gallery-filters__refresh"
          @click="$emit('refresh')"
        >
          <DqIcon><Refresh /></DqIcon>
        </DqIconButton>

        <DqButton
          type="primary"
          size="sm"
          class="studio-gallery-filters__compose"
          @click="$emit('open-composer')"
        >
          <span v-if="composerBusy" class="studio-gallery-filters__compose-dot" aria-hidden="true" />
          {{ $t('studio.openComposer') }}
        </DqButton>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { Refresh, Search } from '@danqing/dq-shell';
import StudioViewModeSwitch from '@/components/studio/StudioViewModeSwitch.vue';
import StudioGalleryBatchActions from '@/components/studio/StudioGalleryBatchActions.vue';
import StudioCanvasSessionControls from '@/components/studio/StudioCanvasSessionControls.vue';
import type { CanvasMedia } from '@/composables/useCanvasStore';

withDefaults(
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
    selectedCount: 0,
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
  emit('update:viewMode', mode);
}
</script>

<style scoped>
.studio-gallery-filters,
.studio-gallery-filters__query,
.studio-gallery-filters__actions {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}

.studio-gallery-filters {
  width: 100%;
}

.studio-gallery-filters__query {
  flex: 1 1 auto;
}

.studio-gallery-filters__time {
  width: 104px;
  flex-shrink: 0;
}

.studio-gallery-filters__models {
  flex: 0 1 200px;
  min-width: 120px;
}

.studio-gallery-filters__search {
  flex: 1 1 220px;
  min-width: 160px;
  max-width: 320px;
}

.studio-gallery-filters__spacer {
  flex: 1 1 auto;
  min-width: 4px;
}

.studio-gallery-filters__actions {
  flex: 0 0 auto;
  justify-content: flex-end;
}

.studio-gallery-filters__compose,
.studio-gallery-filters__refresh {
  flex-shrink: 0;
}

.studio-gallery-filters__compose {
  position: relative;
  min-width: 68px;
}

.studio-gallery-filters__compose-dot {
  position: absolute;
  top: 4px;
  right: 4px;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--dq-on-accent);
}

.studio-gallery-filters__view-mode {
  display: flex;
  align-items: center;
  flex-shrink: 0;
}

.is-selecting {
  justify-content: flex-end;
}

@media (max-width: 1100px) {
  .studio-gallery-filters {
    flex-wrap: wrap;
  }

  .studio-gallery-filters__query {
    flex-basis: 100%;
    order: 2;
  }

  .studio-gallery-filters__actions {
    width: 100%;
    justify-content: flex-end;
    order: 1;
  }

  .studio-gallery-filters__spacer {
    display: none;
  }
}

@media (max-width: 640px) {
  .studio-gallery-filters__query {
    flex-wrap: wrap;
  }

  .studio-gallery-filters__search {
    flex-basis: 100%;
    max-width: none;
  }
}
</style>
