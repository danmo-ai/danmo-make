<script setup lang="ts">
withDefaults(
  defineProps<{
    /** 模型下载页显示「仅已下载」；创作页固定只列已就绪，不展示该选项 */
    showInstalledFilter?: boolean;
  }>(),
  { showInstalledFilter: true },
);

const installedOnly = defineModel<boolean>('installedOnly', { default: false });
const commercialOnly = defineModel<boolean>('commercialOnly', { required: true });
const currentModelsOnly = defineModel<boolean>('currentModelsOnly', { default: false });
</script>

<template>
  <div class="model-picker-filters" role="group" :aria-label="$t('studio.modelFilterGroup')">
    <DqCheckbox
      v-if="showInstalledFilter"
      v-model="installedOnly"
      size="sm"
      class="model-picker-filters__item"
    >
      {{ $t('download.filterReadyOnly') }}
    </DqCheckbox>
    <DqCheckbox v-model="commercialOnly" size="sm" class="model-picker-filters__item">
      {{ $t(showInstalledFilter ? 'download.filterCommercialOnly' : 'studio.filterCommercialOnly') }}
    </DqCheckbox>
    <DqCheckbox
      v-if="showInstalledFilter"
      v-model="currentModelsOnly"
      size="sm"
      class="model-picker-filters__item"
    >
      {{ $t('download.filterCurrentModelsOnly') }}
    </DqCheckbox>
  </div>
</template>
