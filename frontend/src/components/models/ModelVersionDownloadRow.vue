<template>
  <div
    class="model-version-row"
    :class="{
      'model-version-row--prequantized': isPrequantized || isDerived,
      'is-ready': vstatus === 'ready',
      'is-pending': vstatus !== 'ready',
    }"
  >
    <div class="model-version-row__info">
      <div class="model-version-row__head">
        <span class="model-version-name">{{ displayName }}</span>
      </div>
      <div v-if="hasTags" class="model-version-row__tags">
        <DqTag v-if="ver.size" type="info" effect="plain">{{ ver.size }}</DqTag>
        <ModelVersionSourceBadge
          v-if="!hideSource && resolvedSource"
          :source="resolvedSource"
        />
        <DqTag v-if="vstatus === 'ready'" type="success" effect="plain">
          {{ $t('studio.ready') }}
        </DqTag>
        <DqTag
          v-for="comp in visibleBundleComponents"
          :key="`${verKey}-${comp.name}`"
          :type="comp.ok ? 'success' : 'warning'"
          effect="plain"
        >
          {{ $t(`download.component.${comp.name}`) }}
        </DqTag>
      </div>
    </div>
    <div class="model-version-row__actions" @click.stop>
      <DqTag v-if="isDerived" type="info" effect="plain">
        {{ $t('download.derived') }}
      </DqTag>
      <ModelVersionDownloadActions
        :vstatus="vstatus"
        :can-download="canDownload"
        :dependency-hint="dependencyHint"
        :loading="loading"
        :derived="isDerived"
        @download="$emit('download')"
        @delete="$emit('delete')"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import ModelVersionDownloadActions from '@/components/models/ModelVersionDownloadActions.vue';
import ModelVersionSourceBadge from '@/components/models/ModelVersionSourceBadge.vue';
import { $vn } from '@/utils/i18n';
import { simplifyPrequantizedName, type BilingualVersionName } from '@/utils/modelVersionLayout';

const props = defineProps<{
  verKey: string;
  ver: { name: BilingualVersionName; size?: string; source_type?: string; source?: string };
  vstatus: string;
  modelSource?: string;
  isPrequantized?: boolean;
  isDerived?: boolean;
  hideSource?: boolean;
  canDownload: boolean;
  dependencyHint: string;
  loading: boolean;
  bundleComponents: Array<{ name: string; ok: boolean }>;
}>();

defineEmits<{
  download: [];
  delete: [];
}>();

const resolvedSource = computed(
  () => props.ver.source || props.modelSource || '',
);

const localizedName = computed(() => $vn(props.ver, props.verKey));

const displayName = computed(() => {
  if (!props.isPrequantized) return localizedName.value;
  return simplifyPrequantizedName(localizedName.value, resolvedSource.value);
});

const hasTags = computed(() => {
  if (props.ver.size) return true;
  if (!props.hideSource && resolvedSource.value) return true;
  if (props.vstatus === 'ready') return true;
  return visibleBundleComponents.value.length > 0;
});

const visibleBundleComponents = computed(() => {
  if (props.bundleComponents.length === 0) return [];
  if (props.vstatus !== 'ready') return [];
  return props.bundleComponents.filter((comp) => !comp.ok);
});
</script>
