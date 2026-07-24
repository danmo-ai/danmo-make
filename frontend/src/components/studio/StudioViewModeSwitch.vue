<template>
  <DqSegmented
    :model-value="modelValue"
    size="sm"
    class="studio-view-mode-switch"
    :options="options"
    @update:model-value="$emit('update:modelValue', $event as 'grid' | 'canvas')"
  />
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';

defineProps<{
  modelValue: 'grid' | 'canvas';
}>();

defineEmits<{
  (e: 'update:modelValue', value: 'grid' | 'canvas'): void;
}>();

const { t: $t } = useI18n();

const options = computed(() => [
  { label: $t('gallery.viewModeGrid'), value: 'grid' },
  { label: $t('gallery.viewModeCanvas'), value: 'canvas' },
]);
</script>

<style scoped>
.studio-view-mode-switch {
  flex-shrink: 0;
}
</style>
