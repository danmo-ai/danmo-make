<template>
  <div class="model-version-list">
    <div
      v-for="block in layout.fullBlocks"
      :key="block.verKey"
      class="model-version-entry"
      :class="{
        'is-expanded': isParentExpanded(block),
        'is-ready': block.vstatus === 'ready',
      }"
    >
      <ModelVersionDownloadRow
        :ver-key="block.verKey"
        :ver="block.ver"
        :vstatus="block.vstatus"
        :model-source="model.source"
        :hide-source="Boolean(uniformSource)"
        :expandable="block.derivedVariants.length > 0"
        :expanded="isParentExpanded(block)"
        :has-local-quant="block.derivedVariants.length > 0"
        :can-download="canDownload"
        :dependency-hint="dependencyHint"
        :loading="Boolean(loadingKeys[`${model.id}-${block.verKey}`])"
        :bundle-components="bundleComponents(block.verKey)"
        @download="$emit('download', block.verKey)"
        @delete="$emit('delete', block.verKey)"
        @toggle-expand="toggleParent(block.verKey)"
      />
      <ModelVersionLocalQuantPanel
        v-if="block.derivedVariants.length && isParentExpanded(block)"
        :parent-ready="block.parentReady"
        :derived-variants="block.derivedVariants"
        @delete-derived="$emit('delete', $event)"
      />
    </div>

    <ModelVersionDownloadRow
      v-for="row in layout.lightweightRows"
      :key="row.verKey"
      :ver-key="row.verKey"
      :ver="row.ver"
      :vstatus="row.vstatus"
      :model-source="model.source"
      :is-prequantized="true"
      :hide-source="Boolean(uniformSource)"
      :can-download="canDownload"
      :dependency-hint="dependencyHint"
      :loading="Boolean(loadingKeys[`${model.id}-${row.verKey}`])"
      :bundle-components="bundleComponents(row.verKey)"
      @download="$emit('download', row.verKey)"
      @delete="$emit('delete', row.verKey)"
    />
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import ModelVersionDownloadRow from '@/components/models/ModelVersionDownloadRow.vue';
import ModelVersionLocalQuantPanel from '@/components/models/ModelVersionLocalQuantPanel.vue';
import {
  buildModelVersionLayout,
  type FullVersionBlock,
  type ModelVersionLayoutInput,
} from '@/utils/modelVersionLayout';

const props = defineProps<{
  model: ModelVersionLayoutInput;
  uniformSource?: string;
  loadingKeys: Record<string, boolean>;
  canDownload: boolean;
  dependencyHint: string;
  getVersionStatus: (modelId: string, verKey: string) => string;
  bundleComponentsFor: (verKey: string) => Array<{ name: string; ok: boolean }>;
}>();

defineEmits<{
  download: [verKey: string];
  delete: [verKey: string];
}>();

const expandedParents = ref<Set<string>>(new Set());

const layout = computed(() =>
  buildModelVersionLayout(props.model, props.getVersionStatus),
);

watch(
  () => props.model.id,
  () => {
    expandedParents.value = new Set();
  },
);

function bundleComponents(verKey: string) {
  return props.bundleComponentsFor(verKey);
}

function isParentExpanded(block: FullVersionBlock): boolean {
  return expandedParents.value.has(block.verKey);
}

function toggleParent(verKey: string) {
  const block = layout.value.fullBlocks.find((item) => item.verKey === verKey);
  if (!block || block.derivedVariants.length === 0) return;
  const next = new Set(expandedParents.value);
  if (next.has(verKey)) next.delete(verKey);
  else next.add(verKey);
  expandedParents.value = next;
}
</script>
