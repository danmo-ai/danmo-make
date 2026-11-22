<template>
  <DqStack wrap :gap="6" justify="end" align="center">
    <template v-if="vstatus === 'ready'">
      <DqButton
        size="sm"
        class="model-ver-btn model-ver-btn--force"
        @click="$emit('download')"
      >
        <DqIcon class="model-ver-btn__icon"><download /></DqIcon>
        <span class="model-ver-btn__label">{{ $t('download.forceDownload') }}</span>
      </DqButton>
      <DqButton
        size="sm"
        class="model-ver-btn model-ver-btn--delete"
        @click="$emit('delete')"
      >
        <DqIcon class="model-ver-btn__icon"><delete /></DqIcon>
        <span class="model-ver-btn__label">{{ $t('common.delete') }}</span>
      </DqButton>
    </template>
    <template v-else>
      <DqTooltip v-if="!canDownload" :content="dependencyHint" placement="top">
        <span>
          <DqButton size="sm" class="model-ver-btn model-ver-btn--download" disabled>
            <DqIcon class="model-ver-btn__icon"><download /></DqIcon>
            <span class="model-ver-btn__label">{{ actionLabel }}</span>
          </DqButton>
        </span>
      </DqTooltip>
      <DqButton
        v-else
        size="sm"
        class="model-ver-btn model-ver-btn--download"
        :loading="loading"
        @click="$emit('download')"
      >
        <DqIcon class="model-ver-btn__icon"><download /></DqIcon>
        <span class="model-ver-btn__label">{{ actionLabel }}</span>
      </DqButton>
    </template>
  </DqStack>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';

const props = defineProps<{
  vstatus: string;
  canDownload: boolean;
  dependencyHint: string;
  loading: boolean;
  derived?: boolean;
}>();

defineEmits<{
  download: [];
  delete: [];
}>();

const { t } = useI18n();

// A derived version is built on this machine from the full-precision base, so
// the action says "quantize" rather than "download".
const actionLabel = computed(() => {
  if (!props.derived) return t('download.downloadVersion');
  return props.loading ? t('download.converting') : t('download.convert');
});
</script>
