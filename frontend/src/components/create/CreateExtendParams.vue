<script setup lang="ts">
// @ts-nocheck
defineProps<{
  params: Record<string, unknown>;
  modelValue?: string;
  modelOptions?: Array<{ label: string; value: string; disabled?: boolean; commercialUseAllowed?: boolean }>;
}>();
</script>

<template>
  <DqPrefPane class="studio-create-pref-pane studio-editor-drawer-pref-pane">
    <DqPrefRow :label="$t('studio.model')">
      <DqSelect :model-value="modelValue" @update:model-value="$emit('update:modelValue', $event)" size="sm" style="width: 100%" :placeholder="$t('studio.selectModel')">
        <DqOption
          v-for="item in (modelOptions || [])"
          :key="item.value"
          :label="item.label"
          :value="item.value"
          :disabled="item.disabled"
        >
          <DqTag
            v-if="item.commercialUseAllowed"
            size="sm"
            type="success"
            class="studio-drawer-model-badge"
          >
            {{ $t('download.commercialUseBadge') }}
          </DqTag>
        </DqOption>
      </DqSelect>
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
