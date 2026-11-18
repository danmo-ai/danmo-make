<template>
  <div v-if="derivedVariants.length" class="model-version-local-panel">
    <p v-if="!parentReady" class="model-version-local-panel__hint">
      {{ $t('download.localQuantNeedFullShort') }}
    </p>
    <div class="model-version-local-panel__list">
      <div
        v-for="dv in derivedVariants"
        :key="dv.verKey"
        class="model-local-variant"
        :class="{ 'is-ready': dv.vstatus === 'ready' }"
      >
        <div class="model-local-variant__info">
          <span class="model-local-variant__name">{{ shortLabel(dv) }}</span>
          <span v-if="dv.ver.size" class="model-local-variant__size">{{ dv.ver.size }}</span>
        </div>
        <div class="model-local-variant__action">
          <template v-if="dv.vstatus === 'ready'">
            <DqTag type="success" effect="plain">{{ $t('download.quantChipReady') }}</DqTag>
            <DqButton
              size="sm"
              class="model-local-variant__btn model-ver-btn model-ver-btn--delete"
              @click.stop="$emit('delete-derived', dv.verKey)"
            >
              <DqIcon class="model-ver-btn__icon"><delete /></DqIcon>
            </DqButton>
          </template>
          <span v-else class="model-local-variant__waiting">{{ $t('download.localQuantFromFull') }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { $vn } from '@/utils/i18n';
import { shortDerivedLabel, type BilingualVersionName } from '@/utils/modelVersionLayout';

defineProps<{
  parentReady: boolean;
  derivedVariants: Array<{
    verKey: string;
    ver: { name: BilingualVersionName; size?: string };
    vstatus: string;
  }>;
}>();

defineEmits<{
  'delete-derived': [verKey: string];
}>();

function shortLabel(dv: { verKey: string; ver: { name: BilingualVersionName } }) {
  return shortDerivedLabel(dv.verKey, $vn(dv.ver, dv.verKey));
}
</script>
