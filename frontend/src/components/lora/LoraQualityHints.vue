<template>
  <div v-if="report && visibleHints.length" class="lora-quality-hints" :class="`is-${report.level}`">
    <DqAlert :type="alertType" :closable="false" :title="titleText">
      <ul class="lora-quality-hints__list">
        <li
          v-for="(hint, idx) in visibleHints"
          :key="`${hint.code}-${idx}`"
          :class="`is-${hint.severity}`"
        >
          <DqTag
            v-if="hint.source === 'vlm'"
            size="sm"
            type="info"
            effect="plain"
            class="lora-quality-hints__tag"
          >
            VLM
          </DqTag>
          {{ hintText(hint) }}
        </li>
      </ul>
      <p v-if="statsLine" class="lora-quality-hints__stats">{{ statsLine }}</p>
      <p v-if="vlmScoreLine" class="lora-quality-hints__stats">{{ vlmScoreLine }}</p>

      <details v-if="perImageSamples.length && mode !== 'dataset'" class="lora-quality-hints__per-image">
        <summary>{{ $t('loraTrain.quality.vlmPerImage', { count: perImageSamples.length }) }}</summary>
        <ul class="lora-quality-hints__per-image-list">
          <li
            v-for="sample in perImageSamples"
            :key="sample.file"
            :class="{ 'is-weak': (sample.score ?? 5) < 3.5 }"
          >
            <span class="lora-quality-hints__per-image-name">{{ sample.file }}</span>
            <span v-if="sample.score != null" class="lora-quality-hints__per-image-score">
              {{ sample.score }}/5
            </span>
            <span v-if="sample.reason" class="lora-quality-hints__per-image-reason">{{ sample.reason }}</span>
          </li>
        </ul>
      </details>

      <div v-if="showVlmAction" class="lora-quality-hints__actions">
        <DqButton
          size="xs"
          type="secondary"
          :loading="vlmLoading"
          @click="$emit('run-vlm')"
        >
          {{ vlmButtonLabel }}
        </DqButton>
        <span v-if="showVlmDescLine" class="lora-quality-hints__vlm-hint">{{ vlmDescText }}</span>
      </div>
    </DqAlert>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import {
  qualityAlertType,
  type LoraDatasetHealthReport,
  type LoraQualityHint,
  type LoraTrainingQualityReport,
} from '@/utils/loraQuality';

const props = defineProps<{
  report: LoraDatasetHealthReport | LoraTrainingQualityReport | null | undefined;
  mode: 'dataset' | 'training';
  visionAvailable?: boolean;
  vlmLoading?: boolean;
  datasetAuditKind?: 'concept' | 'style';
  hideVlmDesc?: boolean;
}>();

defineEmits<{
  (e: 'run-vlm'): void;
}>();

const { t, te } = useI18n();

const effectiveAuditKind = computed((): 'concept' | 'style' => {
  const fromReport = props.report?.audit_kind || props.report?.vlm?.audit_kind;
  if (fromReport === 'style' || fromReport === 'concept') return fromReport;
  return props.datasetAuditKind === 'style' ? 'style' : 'concept';
});

const alertType = computed(() => qualityAlertType(props.report?.level));

const visibleHints = computed(() => {
  const hints = props.report?.hints || [];
  if (props.mode === 'dataset') return hints;
  return hints.filter((h) => h.code !== 'dataset_healthy');
});

const titleText = computed(() => {
  const level = props.report?.level || 'fair';
  const key = `loraTrain.quality.title.${props.mode}.${level}`;
  const translated = t(key);
  return translated !== key ? translated : level;
});

const showVlmAction = computed(() => Boolean(props.visionAvailable));

const showVlmDescLine = computed(() => showVlmAction.value && !props.hideVlmDesc);

const vlmButtonLabel = computed(() => {
  const kind = effectiveAuditKind.value;
  const audited = props.report?.vlm_audited;
  if (props.mode === 'training') {
    const key = audited
      ? `loraTrain.quality.vlmReauditTraining.${kind}`
      : `loraTrain.quality.vlmAuditTraining.${kind}`;
    const translated = t(key);
    return translated !== key ? translated : t('loraTrain.quality.vlmAuditTraining.concept');
  }
  const key = audited
    ? `loraTrain.quality.vlmReaudit.${kind}`
    : `loraTrain.quality.vlmAudit.${kind}`;
  const translated = t(key);
  return translated !== key ? translated : t('loraTrain.quality.vlmAudit.concept');
});

const vlmDescText = computed(() => {
  if (props.mode === 'dataset') {
    return t('loraTrain.datasetKindDesc');
  }
  const kind = effectiveAuditKind.value;
  const key = `loraTrain.quality.vlmAuditDescTraining.${kind}`;
  const translated = t(key);
  return translated !== key ? translated : t('loraTrain.quality.vlmAuditDescTraining.concept');
});

const perImageSamples = computed(() => {
  if (!props.report?.vlm_audited) return [];
  const raw = props.report?.vlm?.samples;
  if (!Array.isArray(raw)) return [];
  return raw.map((s) => ({
    file: String((s as Record<string, unknown>).file || ''),
    score: (s as Record<string, unknown>).score as number | null | undefined,
    reason: String((s as Record<string, unknown>).reason || ''),
  }));
});

const statsLine = computed(() => {
  if (!props.report) return '';
  if (props.mode === 'dataset') {
    const stats = (props.report as LoraDatasetHealthReport).stats || {};
    const count = stats.image_count ?? 0;
    const median = stats.median_short_edge ?? 0;
    if (!count) return '';
    if (median > 0) {
      return t('loraTrain.quality.statsDataset', { count, median });
    }
    return t('loraTrain.quality.statsDatasetShort', { count });
  }
  const metrics = (props.report as LoraTrainingQualityReport).metrics || {};
  const initial = metrics.initial_loss;
  const final = metrics.final_loss;
  if (initial == null || final == null) return '';
  return t('loraTrain.quality.statsTraining', {
    initial: Number(initial).toFixed(3),
    final: Number(final).toFixed(3),
  });
});

const vlmScoreLine = computed(() => {
  const vlm = props.report?.vlm;
  const avg = vlm?.avg_score;
  if (avg == null || !props.report?.vlm_audited) return '';
  const audited = vlm?.audited_count ?? perImageSamples.value.length;
  if (props.mode === 'dataset') {
    const kind = effectiveAuditKind.value;
    const key = kind === 'style' ? 'vlmAvgStyle' : 'vlmAvgPortrait';
    return t(`loraTrain.quality.${key}`, { avg, count: audited });
  }
  const kind = effectiveAuditKind.value;
  const key = kind === 'style' ? 'vlmAvgStyleMatch' : 'vlmAvgLikeness';
  return t(`loraTrain.quality.${key}`, { avg });
});

function hintText(hint: LoraQualityHint): string {
  const key = `loraTrain.quality.hint.${hint.code}`;
  if (te(key)) {
    return t(key, hint.params || {});
  }
  return hint.code;
}
</script>

<style scoped>
.lora-quality-hints__list {
  margin: 0;
  padding-left: 1.1rem;
  font-size: var(--dq-font-size-caption);
  line-height: 1.5;
  color: var(--dq-label-secondary);
}

.lora-quality-hints__list li {
  margin-bottom: 4px;
}

.lora-quality-hints__list li.is-error {
  color: var(--dq-danger);
}

.lora-quality-hints__list li.is-warning {
  color: var(--dq-warning, var(--dq-label-primary));
}

.lora-quality-hints__tag {
  margin-right: 4px;
  vertical-align: middle;
}

.lora-quality-hints__stats {
  margin: 8px 0 0;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
}

.lora-quality-hints__per-image {
  margin-top: 10px;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
}

.lora-quality-hints__per-image summary {
  cursor: pointer;
  color: var(--dq-label-primary);
  font-weight: 500;
}

.lora-quality-hints__per-image-list {
  margin: 8px 0 0;
  padding-left: 1rem;
  list-style: disc;
}

.lora-quality-hints__per-image-list li {
  margin-bottom: 6px;
  line-height: 1.45;
}

.lora-quality-hints__per-image-list li.is-weak {
  color: var(--dq-warning, var(--dq-label-primary));
}

.lora-quality-hints__per-image-name {
  font-weight: 500;
  margin-right: 6px;
}

.lora-quality-hints__per-image-score {
  margin-right: 6px;
  color: var(--dq-label-tertiary);
}

.lora-quality-hints__per-image-reason {
  display: block;
  margin-top: 2px;
  color: var(--dq-label-tertiary);
}

.lora-quality-hints__actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin-top: 10px;
}

.lora-quality-hints__vlm-hint {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
}
</style>
