<template>
  <section v-if="provenance" class="lv-t2i-prov">
    <button type="button" class="lv-t2i-prov__toggle" @click="open = !open">
      <span>{{ $tt('video.longVideoT2iProvenanceTitle') }}</span>
      <span class="lv-t2i-prov__chev" :class="{ 'is-open': open }" aria-hidden="true">▾</span>
    </button>
    <div v-show="open" class="lv-t2i-prov__body">
      <p class="lv-t2i-prov__row">
        <span class="lv-t2i-prov__label">{{ $tt('video.longVideoT2iProvenanceNarrative') }}</span>
        <span v-if="provenance.narrative_merged" class="lv-t2i-prov__val is-on">
          {{ $tt('video.longVideoT2iProvenanceMerged') }}
        </span>
        <span v-else class="lv-t2i-prov__val">
          {{ narrativeSkipLabel }}
          <template v-if="provenance.narrative_token_coverage != null">
            · {{ $tt('video.longVideoT2iProvenanceCoverage', {
              pct: Math.round(provenance.narrative_token_coverage * 100),
            }) }}
          </template>
        </span>
      </p>
      <p class="lv-t2i-prov__row">
        <span class="lv-t2i-prov__label">{{ $tt('video.longVideoT2iProvenanceLocation') }}</span>
        <span class="lv-t2i-prov__val">{{ locationMergeLabel }}</span>
      </p>
      <p class="lv-t2i-prov__row">
        <span class="lv-t2i-prov__label">{{ $tt('video.longVideoT2iProvenanceFfr') }}</span>
        <span class="lv-t2i-prov__val">
          {{ ffrSkipLabel }}
        </span>
      </p>
      <ul v-if="provenance.scene_parts.length" class="lv-t2i-prov__parts">
        <li v-for="(part, idx) in provenance.scene_parts" :key="idx">
          <span class="lv-t2i-prov__part-src">{{ sourceLabel(part.source) }}</span>
          <span class="lv-t2i-prov__part-text">{{ part.text_preview }}</span>
        </li>
      </ul>
      <p v-if="parseRunId" class="lv-t2i-prov__meta">
        {{ $tt('video.longVideoT2iProvenanceParseRun') }}
        <code :title="parseRunId">{{ shortId(parseRunId) }}</code>
      </p>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import type { KeyframeT2iProvenance } from '@/types';
import { shortActivityId } from '@/utils/longVideoActivity';

const props = defineProps<{
  provenance?: KeyframeT2iProvenance | null;
  parseRunId?: string;
}>();

const { t: $tt } = useI18n();
const open = ref(false);

const narrativeSkipLabel = computed(() => {
  const reason = props.provenance?.narrative_skip_reason;
  if (!reason) return $tt('video.longVideoT2iProvenanceSkipped');
  const key = `video.longVideoT2iProvenanceSkip_${reason}`;
  const label = $tt(key);
  return label !== key ? label : reason;
});

const locationMergeLabel = computed(() => {
  const mode = props.provenance?.location_merge ?? 'none';
  const key = `video.longVideoT2iProvenanceLocation_${mode}`;
  const label = $tt(key);
  return label !== key ? label : mode;
});

const ffrSkipLabel = computed(() => {
  const reason = props.provenance?.ffr_skip_reason;
  if (!reason) return $tt('video.longVideoT2iProvenanceSkipped');
  const key = `video.longVideoT2iProvenanceFfrSkip_${reason}`;
  const label = $tt(key);
  return label !== key ? label : reason;
});

function sourceLabel(source: string) {
  const key = `video.longVideoT2iProvenanceSource_${source}`;
  const label = $tt(key);
  return label !== key ? label : source;
}

function shortId(id: string) {
  return shortActivityId(id);
}
</script>

<style scoped>
.lv-t2i-prov {
  margin-top: 8px;
  padding: 8px 10px;
  border-radius: 8px;
  border: 0.5px dashed var(--dq-border-subtle);
  background: color-mix(in srgb, var(--dq-fill-control) 60%, transparent);
}

.lv-t2i-prov__toggle {
  display: flex;
  width: 100%;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 0;
  border: none;
  background: none;
  font-size: var(--dq-font-size-caption);
  font-weight: 650;
  color: var(--dq-label-secondary);
  cursor: pointer;
}

.lv-t2i-prov__chev {
  transition: transform 0.15s ease;
}

.lv-t2i-prov__chev.is-open {
  transform: rotate(180deg);
}

.lv-t2i-prov__body {
  margin-top: 8px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.lv-t2i-prov__row {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  display: flex;
  flex-wrap: wrap;
  gap: 4px 8px;
}

.lv-t2i-prov__label {
  font-weight: 600;
  color: var(--dq-label-primary);
}

.lv-t2i-prov__val {
  color: var(--dq-label-secondary);
}

.lv-t2i-prov__val.is-on {
  color: var(--dq-accent);
}

.lv-t2i-prov__parts {
  margin: 4px 0 0;
  padding-left: 16px;
  list-style: decimal;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
}

.lv-t2i-prov__part-src {
  font-weight: 600;
  color: var(--dq-label-primary);
  margin-right: 4px;
}

.lv-t2i-prov__meta {
  margin: 4px 0 0;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
}
</style>
