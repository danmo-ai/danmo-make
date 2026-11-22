<template>
  <div class="studio-model-field">
    <DqSelect
      :model-value="modelValue"
      size="sm"
      :variant="variant"
      :placeholder="selectPlaceholder"
      :filterable="filterable || undefined"
      :disabled="selectDisabled"
      class="studio-model-field__select"
      @update:model-value="onUpdate"
      @change="onChange"
    >
      <DqOption
        v-for="item in options"
        :key="item.value"
        :label="item.label"
        :value="item.value"
        :disabled="item.disabled"
      >
        <DqTag
          v-if="item.commercialUseAllowed"
          size="sm"
          type="success"
          class="studio-model-option__badge"
        >
          {{ $t('download.commercialUseBadge') }}
        </DqTag>
      </DqOption>
    </DqSelect>
    <ComposerIconTip v-if="description" :content="description" popper-class="studio-model-tip">
      <DqIconButton
        type="text"
        size="sm"
        class="studio-model-field__tip"
        :aria-label="description"
      >
        <span class="studio-model-field__tip-mark" aria-hidden="true">i</span>
      </DqIconButton>
    </ComposerIconTip>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import ComposerIconTip from '@/components/studio/ComposerIconTip.vue';
import type { ModelSelectOption } from '@/utils/modelSelectOption';

const props = withDefaults(
  defineProps<{
    modelValue?: string;
    options: ModelSelectOption[];
    placeholder?: string;
    variant?: string;
    filterable?: boolean;
    disabled?: boolean;
  }>(),
  {
    modelValue: '',
    options: () => [],
    placeholder: '',
    filterable: false,
  },
);

const emit = defineEmits<{
  (e: 'update:modelValue', value: string): void;
  (e: 'change', value: string): void;
}>();

const { t } = useI18n();
const selectPlaceholder = computed(() => props.placeholder || t('studio.selectModel'));
const selectDisabled = computed(() => props.disabled || props.options.length === 0);

const description = computed(() => {
  const row = props.options.find((item) => item.value === props.modelValue);
  return String(row?.description || '').trim();
});

function onUpdate(value: string) {
  emit('update:modelValue', value);
}

function onChange(value: string) {
  emit('change', value);
}
</script>
