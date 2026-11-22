<template>
  <div class="model-version-list">
    <div
      v-for="block in layout.fullBlocks"
      :key="block.verKey"
      class="model-version-entry"
      :class="{ 'is-ready': block.vstatus === 'ready' }"
    >
      <ModelVersionDownloadRow
        :ver-key="block.verKey"
        :ver="block.ver"
        :vstatus="block.vstatus"
        :model-source="model.source"
        :hide-source="Boolean(uniformSource)"
        :can-download="canDownload"
        :dependency-hint="dependencyHint"
        :loading="Boolean(loadingKeys[`${model.id}-${block.verKey}`])"
        :bundle-components="bundleComponents(block.verKey)"
        @download="$emit('download', block.verKey)"
        @delete="$emit('delete', block.verKey)"
      />
    </div>

    <div
      v-if="layout.lightweightRows.length"
      class="model-version-list__derived"
    >
      <p class="model-version-list__derived-hint">
        {{ $t('download.derivedHint') }}
      </p>
      <ModelVersionDownloadRow
        v-for="row in layout.lightweightRows"
        :key="row.verKey"
        :ver-key="row.verKey"
        :ver="row.ver"
        :vstatus="row.vstatus"
        :model-source="model.source"
        :is-prequantized="row.ver.source_type === 'prequantized'"
        :is-derived="isDerivedVersion(row.ver)"
        :hide-source="Boolean(uniformSource)"
        :can-download="canDownload"
        :dependency-hint="dependencyHint"
        :loading="Boolean(loadingKeys[`${model.id}-${row.verKey}`])"
        :bundle-components="bundleComponents(row.verKey)"
        @download="$emit('download', row.verKey)"
        @delete="$emit('delete', row.verKey)"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import ModelVersionDownloadRow from '@/components/models/ModelVersionDownloadRow.vue';
import {
  buildModelVersionLayout,
  isDerivedVersion,
  type ModelVersionLayoutInput,
} from '@/utils/modelVersionLayout';

const props = withDefaults(
  defineProps<{
    model: ModelVersionLayoutInput;
    uniformSource?: string;
    loadingKeys: Record<string, boolean>;
    canDownload?: boolean;
    dependencyHint?: string;
    getVersionStatus: (modelId: string, verKey: string) => string;
    bundleComponentsFor: (verKey: string) => Array<{ name: string; ok: boolean }>;
  }>(),
  {
    canDownload: true,
    dependencyHint: '',
  },
);

defineEmits<{
  download: [verKey: string];
  delete: [verKey: string];
}>();

const layout = computed(() =>
  buildModelVersionLayout(props.model, props.getVersionStatus),
);

function bundleComponents(verKey: string) {
  return props.bundleComponentsFor(verKey);
}
</script>
