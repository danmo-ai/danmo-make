<template>
  <DqSurfaceCard class="model-card user-lora-card">
    <div class="user-lora-card__head">
      <span class="user-lora-card__mark" aria-hidden="true">{{ initials }}</span>
      <div class="user-lora-card__identity">
        <div class="user-lora-card__name" :title="lora.name || ''">{{ lora.name }}</div>
        <p v-if="lora.trigger_word" class="user-lora-card__trigger">{{ lora.trigger_word }}</p>
        <p v-else-if="lora.repo_id" class="user-lora-card__trigger" :title="lora.repo_id">{{ lora.repo_id }}</p>
      </div>
      <em v-if="schemeLabel" class="user-lora-card__badge">{{ schemeLabel }}</em>
    </div>

    <p class="user-lora-card__meta">
      <span>{{ displayBaseModel }}</span>
      <span v-if="lora.lora_rank">rank {{ lora.lora_rank }}</span>
      <span v-if="hubSourceLabel">{{ hubSourceLabel }}</span>
      <span v-if="formattedDate">{{ formattedDate }}</span>
    </p>
    <p v-if="recipeLine" class="user-lora-card__recipe">{{ recipeLine }}</p>

    <div class="user-lora-card__actions">
      <DqButton size="sm" type="primary" @click="emit('verify')">
        {{ $t('loraTrain.verifyGenerate') }}
      </DqButton>
      <DqButton
        v-if="lora.task_id"
        size="sm"
        type="secondary"
        @click="emit('view-run', String(lora.task_id))"
      >
        {{ $t('loraTrain.viewTrainingRun') }}
      </DqButton>
      <DqButton size="sm" type="text" class="user-lora-card__delete" @click="emit('delete')">
        {{ $t('common.delete') }}
      </DqButton>
    </div>
  </DqSurfaceCard>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import { modelInitialsFromName } from '@/utils/modelInitials';

export interface UserLoraInference {
  scheme?: string;
  model?: string;
  steps?: number;
  guidance?: number;
  lora_weight?: number;
  scheduler?: string;
  extra_adapters?: string[];
}

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
  inference?: UserLoraInference;
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

const schemeLabel = computed(() => {
  const scheme = String(props.lora.inference?.scheme || '').trim();
  if (scheme === 'scheme4') return t('loraTrain.preset.scheme4');
  return '';
});

const recipeLine = computed(() => {
  const inference = props.lora.inference;
  if (!inference || inference.steps == null) return '';
  return t('loraTrain.adapterRecipe', {
    steps: inference.steps,
    guidance: inference.guidance ?? 0,
    weight: inference.lora_weight ?? 1,
  });
});
</script>

<style scoped>
.user-lora-card.model-card {
  border: 0.5px solid var(--dq-border-subtle);
  border-radius: 16px;
  background: var(--dq-bg-base);
  box-shadow: none;
  overflow: hidden;
}

.user-lora-card.model-card:hover {
  border-color: color-mix(in srgb, var(--dq-accent) 45%, var(--dq-border));
  box-shadow: 0 10px 28px color-mix(in srgb, var(--dq-accent) 10%, transparent);
}

.user-lora-card :deep(.dq-surface-card__body) {
  display: flex;
  flex-direction: column;
  gap: 12px;
  height: 100%;
  padding: 16px;
  box-sizing: border-box;
}

.user-lora-card__head {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  min-width: 0;
}

.user-lora-card__mark {
  flex-shrink: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  border-radius: 12px;
  font-size: var(--dq-font-size-caption);
  font-weight: 700;
  letter-spacing: var(--dq-tracking-tight);
  color: var(--dq-accent);
  background: color-mix(in srgb, var(--dq-accent) 12%, var(--dq-fill-secondary));
  border: 0.5px solid color-mix(in srgb, var(--dq-accent) 28%, transparent);
}

.user-lora-card__identity {
  flex: 1;
  min-width: 0;
}

.user-lora-card__name {
  font-size: var(--dq-font-size-body);
  font-weight: 600;
  line-height: 1.35;
  color: var(--dq-label-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.user-lora-card__trigger {
  margin: 2px 0 0;
  font-size: var(--dq-font-size-caption);
  line-height: 1.4;
  color: var(--dq-label-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.user-lora-card__badge {
  flex-shrink: 0;
  font-style: normal;
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  color: var(--dq-accent);
}

.user-lora-card__meta,
.user-lora-card__recipe {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
  color: var(--dq-label-tertiary);
}

.user-lora-card__meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0;
}

.user-lora-card__meta span + span::before {
  content: '·';
  margin: 0 6px;
  color: var(--dq-label-quaternary);
}

.user-lora-card__recipe {
  color: var(--dq-label-secondary);
}

.user-lora-card__actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin-top: auto;
}

.user-lora-card__delete {
  margin-left: auto;
  color: var(--dq-label-tertiary);
}

.user-lora-card__delete:hover {
  color: var(--dq-danger);
}
</style>
