<script setup lang="ts">
// @ts-nocheck
import ModelSelect from '@/components/model/ModelSelect.vue';

defineProps<{
  params: Record<string, unknown>;
  modelValue?: string;
  modelOptions?: Array<{ label: string; value: string; disabled?: boolean; commercialUseAllowed?: boolean; description?: string }>;
}>();
</script>

<template>
  <DqPrefPane class="studio-create-pref-pane studio-editor-drawer-pref-pane">
    <DqPrefRow :label="$t('studio.model')">
      <ModelSelect
        :model-value="modelValue"
        :options="modelOptions || []"
        :placeholder="$t('studio.selectModel')"
        @update:model-value="$emit('update:modelValue', $event)"
      />
    </DqPrefRow>

    <DqPrefRow :label="$t('create.extendDirections')">
      <DqCheckboxGroup v-model="params.extend_directions">
        <DqCheckbox label="top">{{ $t('create.extendTop') }}</DqCheckbox>
        <DqCheckbox label="bottom">{{ $t('create.extendBottom') }}</DqCheckbox>
        <DqCheckbox label="left">{{ $t('create.extendLeft') }}</DqCheckbox>
        <DqCheckbox label="right">{{ $t('create.extendRight') }}</DqCheckbox>
      </DqCheckboxGroup>
    </DqPrefRow>
    <DqPrefRow :label="$t('create.extendPixels')">
      <DqInputNumber
        v-model="params.extend_pixels"
        :min="64"
        :max="2048"
        :step="64"
        size="sm"
        style="width: 120px"
      />
    </DqPrefRow>
  </DqPrefPane>
</template>
