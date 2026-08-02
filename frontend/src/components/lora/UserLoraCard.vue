<template>
  <DqSurfaceCard class="model-card model-ready user-lora-model-card">
    <div class="model-card-header">
      <div class="model-icon">{{ initials }}</div>
      <div
        class="dq-status-dot dq-status-dot--success model-card-status"
        :title="$t('loraTrain.registeredBadge')"
      />
    </div>

    <div class="model-card-content">
      <DqTooltip :content="lora.name || ''" placement="top">
        <div class="model-card-name">{{ lora.name }}</div>
      </DqTooltip>

      <div v-if="lora.trigger_word" class="model-card-desc">
        {{ $t('loraTrain.triggerWord') }}: {{ lora.trigger_word }}
      </div>
      <div v-else-if="lora.repo_id" class="model-card-desc" :title="lora.repo_id">
        {{ lora.repo_id }}
      </div>
      <div v-else-if="formattedDate" class="model-card-desc">
        {{ formattedDate }}
      </div>

      <div class="model-card-meta">
        <DqTag size="sm" type="info" effect="plain">{{ displayBaseModel }}</DqTag>
        <DqTag v-if="lora.lora_rank" size="sm" type="success" effect="plain">
          Rank {{ lora.lora_rank }}
        </DqTag>
        <DqTag v-if="hubSourceLabel" size="sm" type="warning" effect="plain">
          {{ hubSourceLabel }}
        </DqTag>
      </div>

      <div class="model-card-actions">
        <DqStack wrap :gap="6" align="center">
          <DqButton size="sm" class="model-ver-btn model-ver-btn--download" @click="emit('verify')">
            <DqIcon class="model-ver-btn__icon"><picture-filled /></DqIcon>
            <span class="model-ver-btn__label">{{ $t('loraTrain.verifyGenerate') }}</span>
          </DqButton>
          <DqButton
            v-if="lora.task_id"
            size="sm"
            class="model-ver-btn model-ver-btn--neutral"
            @click="emit('view-run', String(lora.task_id))"
          >
            <DqIcon class="model-ver-btn__icon"><document /></DqIcon>
            <span class="model-ver-btn__label">{{ $t('loraTrain.viewTrainingRun') }}</span>
          </DqButton>
          <DqButton size="sm" class="model-ver-btn model-ver-btn--delete" @click="emit('delete')">
            <DqIcon class="model-ver-btn__icon"><delete /></DqIcon>
            <span class="model-ver-btn__label">{{ $t('common.delete') }}</span>
          </DqButton>
        </DqStack>
      </div>
    </div>
  </DqSurfaceCard>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import { modelInitialsFromName } from '@/utils/modelInitials';

export interface UserLoraItem {
  id?: string;
  name?: string;
  base_model?: string;
  trigger_word?: string;
  lora_rank?: number;
  task_id?: string;
  created_at?: string;
  repo_id?: string;
  remote_hub_source?: string;
  local_path?: string;
}

const props = defineProps<{
  lora: UserLoraItem;
  baseModelLabel?: string;
}>();

const emit = defineEmits<{
  (e: 'verify'): void;
  (e: 'view-run', taskId: string): void;
  (e: 'delete'): void;
}>();

const { locale, t } = useI18n();

const hubSourceLabel = computed(() => {
  const src = String(props.lora.remote_hub_source || '').trim();
  if (src === 'modelscope') return t('download.sourceModelscope');
  if (src === 'huggingface') return t('download.sourceHuggingface');
  if (src === 'civitai') return t('download.sourceCivitai');
  return '';
});

const displayBaseModel = computed(
  () => props.baseModelLabel || String(props.lora.base_model || '').trim() || '—'
);

const initials = computed(() => modelInitialsFromName(String(props.lora.name || ''), 'Lo'));

const formattedDate = computed(() => {
  const raw = props.lora.created_at;
  if (!raw) return '';
  try {
    return new Date(String(raw)).toLocaleDateString(locale.value === 'zh' ? 'zh-CN' : undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  } catch {
    return String(raw);
  }
});
</script>

