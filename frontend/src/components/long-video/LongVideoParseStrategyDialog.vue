<template>
  <DqDialog
    v-model:open="open"
    class="lv-parse-strategy-dialog"
    :title="$tt('video.longVideoParseStrategyTitle')"
    width="min(480px, 94vw)"
    @update:open="onOpenChange"
  >
    <p class="lv-parse-strategy-dialog__lead">{{ $tt('video.longVideoParseStrategyLead') }}</p>
    <p class="lv-parse-strategy-dialog__note">{{ $tt('video.longVideoParseStrategyManualNote') }}</p>

    <div class="lv-parse-strategy-dialog__options">
      <button type="button" class="lv-parse-strategy-dialog__option" @click="emit('choose', 'replace')">
        <span class="lv-parse-strategy-dialog__option-kicker">{{ $tt('video.longVideoParseStrategyReplaceKicker') }}</span>
        <span class="lv-parse-strategy-dialog__option-title">{{ $tt('video.longVideoParseStrategyReplaceTitle') }}</span>
        <span class="lv-parse-strategy-dialog__option-desc">{{ $tt('video.longVideoParseStrategyReplaceDesc') }}</span>
      </button>
      <button type="button" class="lv-parse-strategy-dialog__option" @click="emit('choose', 'new_project')">
        <span class="lv-parse-strategy-dialog__option-kicker">{{ $tt('video.longVideoParseStrategyNewKicker') }}</span>
        <span class="lv-parse-strategy-dialog__option-title">{{ $tt('video.longVideoParseStrategyNewTitle') }}</span>
        <span class="lv-parse-strategy-dialog__option-desc">{{ $tt('video.longVideoParseStrategyNewDesc') }}</span>
      </button>
    </div>
  </DqDialog>
</template>

<script setup lang="ts">
import { useI18n } from 'vue-i18n';

export type LongVideoParseStrategy = 'replace' | 'new_project';

const open = defineModel<boolean>('open', { required: true });

const emit = defineEmits<{
  (e: 'choose', strategy: LongVideoParseStrategy): void;
  (e: 'cancel'): void;
}>();

const { t: $tt } = useI18n();

function onOpenChange(value: boolean) {
  if (!value) emit('cancel');
}
</script>

<style scoped>
.lv-parse-strategy-dialog__lead {
  margin: 0 0 8px;
  font-size: var(--dq-font-size-body);
  line-height: 1.55;
  color: var(--dq-label-primary);
}

.lv-parse-strategy-dialog__note {
  margin: 0 0 16px;
  font-size: var(--dq-font-size-caption);
  line-height: 1.5;
  color: var(--dq-label-tertiary);
}

.lv-parse-strategy-dialog__options {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.lv-parse-strategy-dialog__option {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 4px;
  width: 100%;
  padding: 14px 16px;
  border-radius: 12px;
  border: 1px solid var(--dq-border-subtle);
  background: color-mix(in srgb, var(--dq-surface-elevated) 45%, transparent);
  text-align: left;
  cursor: pointer;
  transition: border-color 0.15s ease, background 0.15s ease;
}

.lv-parse-strategy-dialog__option:hover {
  border-color: color-mix(in srgb, var(--dq-accent) 45%, var(--dq-border-subtle));
  background: color-mix(in srgb, var(--dq-accent) 8%, transparent);
}

.lv-parse-strategy-dialog__option-kicker {
  font-size: var(--dq-font-size-caption);
  font-weight: 700;
  letter-spacing: var(--dq-tracking-widest);
  text-transform: uppercase;
  color: var(--dq-accent);
}

.lv-parse-strategy-dialog__option-title {
  font-size: var(--dq-font-size-body);
  font-weight: 650;
  color: var(--dq-label-primary);
}

.lv-parse-strategy-dialog__option-desc {
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
  color: var(--dq-label-secondary);
}
</style>
