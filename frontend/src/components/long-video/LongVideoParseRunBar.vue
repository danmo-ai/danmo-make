<template>
  <section v-if="visible" class="lv-parse-run-bar" :class="{ 'lv-parse-run-bar--blocked': parseBlocked }">
    <div class="lv-parse-run-bar__head">
      <div class="lv-parse-run-bar__title-row">
        <span v-if="parseBlocked" class="lv-parse-run-bar__blocked" aria-hidden="true">!</span>
        <span v-else class="lv-parse-run-bar__ok" aria-hidden="true">✓</span>
        <span class="lv-parse-run-bar__title">
          {{ parseBlocked ? $tt('video.longVideoParseRunBlockedTitle') : $tt('video.longVideoParseRunTitle') }}
        </span>
        <code v-if="parseRunId" class="lv-parse-run-bar__id" :title="parseRunId">{{ shortId(parseRunId) }}</code>
        <DqButton v-if="parseRunId" size="sm" type="text" @click="copyId(parseRunId)">
          {{ $tt('studio.taskIdCopy') }}
        </DqButton>
      </div>
      <div class="lv-parse-run-bar__meta">
        <span>{{ $tt('video.longVideoParseRunShots', { n: shotCount }) }}</span>
        <span v-if="lastParseAt">· {{ formatTime(lastParseAt) }}</span>
        <span v-if="qualityCount > 0" class="lv-parse-run-bar__warn">
          · {{ $tt('video.longVideoParseRunQuality', { count: qualityCount }) }}
        </span>
      </div>
    </div>
    <div class="lv-parse-run-bar__actions">
      <DqButton size="sm" type="text" @click="expanded = !expanded">
        {{ expanded ? $tt('video.longVideoParseRunCollapse') : $tt('video.longVideoParseRunExpand') }}
      </DqButton>
    </div>
    <div v-if="expanded" class="lv-parse-run-bar__body">
      <p v-if="loading" class="lv-parse-run-bar__loading">{{ $tt('common.loading') }}</p>
      <template v-else>
        <ol v-if="phaseRows.length" class="lv-parse-run-bar__phases">
          <li v-for="(row, idx) in phaseRows" :key="`${row.phase}-${idx}`" class="lv-parse-run-bar__phase">
            <span class="lv-parse-run-bar__phase-name">{{ phaseLabel(row.phase) }}</span>
            <span v-if="row.message" class="lv-parse-run-bar__phase-msg">{{ row.message }}</span>
          </li>
        </ol>
        <ul v-if="qualityIssues.length" class="lv-parse-run-bar__issues">
          <li
            v-for="(issue, idx) in qualityIssues.slice(0, 8)"
            :key="idx"
            :class="{ 'is-critical': issue.severity === 'critical' }"
          >
            {{ issue.message }}
          </li>
        </ul>
        <p v-if="parseBlockedReason" class="lv-parse-run-bar__blocked-reason">{{ parseBlockedReason }}</p>
        <div v-if="parseHistory.length > 1" class="lv-parse-run-bar__history">
          <span class="lv-parse-run-bar__history-label">{{ $tt('video.longVideoParseHistoryTitle') }}</span>
          <ul class="lv-parse-run-bar__history-list">
            <li v-for="entry in parseHistoryNewestFirst" :key="entry.parse_run_id">
              <button
                type="button"
                class="lv-parse-run-bar__history-btn"
                :class="{ 'is-active': entry.parse_run_id === selectedHistoryRunId }"
                @click="selectedHistoryRunId = entry.parse_run_id"
              >
                <code :title="entry.parse_run_id">{{ shortId(entry.parse_run_id) }}</code>
                <span>{{ formatTime(entry.at) }}</span>
                <span>· {{ $tt('video.longVideoParseRunShots', { n: entry.shot_count }) }}</span>
              </button>
            </li>
          </ul>
          <p v-if="selectedHistorySummary" class="lv-parse-run-bar__history-summary">
            {{ selectedHistorySummary }}
          </p>
        </div>
        <p v-if="projectId" class="lv-parse-run-bar__project">
          {{ $tt('video.longVideoProjectIdLabel') }}
          <code :title="projectId">{{ shortId(projectId) }}</code>
          <DqButton size="sm" type="text" @click="copyId(projectId)">{{ $tt('studio.taskIdCopy') }}</DqButton>
        </p>
      </template>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { toast } from '@/utils/feedback';
import { api } from '@/utils/api';
import { formatActivityTime, shortActivityId } from '@/utils/longVideoActivity';
import type { KeyframeT2iProvenance, LongVideoChapterAnalysis } from '@/types';

const props = defineProps<{
  projectId?: string;
  chapterAnalysis?: LongVideoChapterAnalysis | null;
  shotCount?: number;
}>();

const { t: $tt, locale } = useI18n();
const expanded = ref(false);
const loading = ref(false);
const remotePhases = ref<Array<{ phase?: string; message?: string }>>([]);
const selectedHistoryRunId = ref('');

const parseRunId = computed(() => props.chapterAnalysis?.parse_run_id?.trim() || '');
const parseHistory = computed(() => props.chapterAnalysis?.parse_history ?? []);
const parseHistoryNewestFirst = computed(() =>
  [...parseHistory.value].reverse(),
);

watch(parseRunId, (id) => {
  if (id) selectedHistoryRunId.value = id;
}, { immediate: true });

function summarizeProvenance(map: Record<string, KeyframeT2iProvenance>): string {
  const rows = Object.values(map);
  if (!rows.length) return '';
  const narrativeMerged = rows.filter((r) => r.narrative_merged).length;
  const faceSkip = rows.filter((r) => r.narrative_skip_reason === 'face_anchor').length;
  const closeSkip = rows.filter((r) => r.narrative_skip_reason === 'close_up').length;
  return $tt('video.longVideoParseHistoryProvenanceSummary', {
    narrative: narrativeMerged,
    face: faceSkip,
    close: closeSkip,
    total: rows.length,
  });
}

const selectedHistorySummary = computed(() => {
  const id = selectedHistoryRunId.value || parseRunId.value;
  if (!id) return '';
  const entry = parseHistory.value.find((h) => h.parse_run_id === id);
  if (!entry?.provenance_by_shot_id) return '';
  return summarizeProvenance(entry.provenance_by_shot_id);
});
const lastParseAt = computed(() => props.chapterAnalysis?.last_parse_at?.trim() || '');
const shotCount = computed(() => Math.max(0, props.shotCount ?? 0));
const qualityIssues = computed(() => props.chapterAnalysis?.quality_issues ?? []);
const parseBlocked = computed(() => Boolean(props.chapterAnalysis?.parse_blocked));
const parseBlockedReason = computed(() => props.chapterAnalysis?.parse_blocked_reason?.trim() || '');
const qualityCount = computed(
  () => (props.chapterAnalysis?.quality_warnings?.length ?? 0) + qualityIssues.value.length,
);
const visible = computed(() => Boolean(parseRunId.value || props.chapterAnalysis?.scene_beats?.length));

const phaseRows = computed(() => {
  if (remotePhases.value.length) return remotePhases.value;
  const cached = props.chapterAnalysis?.parse_phases ?? [];
  return cached.map((p) => ({ phase: p.phase, message: p.message || '' }));
});

function shortId(id: string) {
  return shortActivityId(id);
}

function formatTime(iso: string) {
  return formatActivityTime(iso, locale.value);
}

function phaseLabel(phase: string | undefined) {
  const p = (phase || '').trim();
  if (!p) return '';
  if (p === 'review_retry') return $tt('video.longVideoParsePhaseReviewRetry');
  const key = `video.longVideoParsePhase${p
    .split('_')
    .map((p) => p.charAt(0).toUpperCase() + p.slice(1))
    .join('')}`;
  const label = $tt(key);
  return label !== key ? label : p;
}

async function copyId(id: string) {
  try {
    await navigator.clipboard.writeText(id);
    toast.success($tt('studio.taskIdCopied'));
  } catch {
    toast.error($tt('studio.error', { msg: 'clipboard' }));
  }
}

watch(
  () => [expanded.value, props.projectId, parseRunId.value] as const,
  async ([isOpen, projectId, runId]) => {
    if (!isOpen || !projectId || !runId) return;
    loading.value = true;
    try {
      const detail = await api.longVideo.getParseRun(projectId, runId);
      remotePhases.value = detail.phases ?? [];
    } catch {
      remotePhases.value = [];
    } finally {
      loading.value = false;
    }
  },
);
</script>

<style scoped>
.lv-parse-run-bar {
  margin-top: 10px;
  padding: 10px 12px;
  border-radius: 10px;
  border: 0.5px solid color-mix(in srgb, var(--dq-accent) 28%, var(--dq-border-subtle));
  background: color-mix(in srgb, var(--dq-accent) 6%, var(--dq-fill-control));
}

.lv-parse-run-bar__head {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.lv-parse-run-bar__title-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
}

.lv-parse-run-bar--blocked {
  border-color: color-mix(in srgb, var(--dq-danger) 35%, var(--dq-border-subtle));
  background: color-mix(in srgb, var(--dq-danger) 6%, var(--dq-fill-control));
}

.lv-parse-run-bar__blocked {
  color: var(--dq-danger);
  font-weight: 700;
}

.lv-parse-run-bar__blocked-reason {
  margin: 0 0 8px;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-danger);
}

.lv-parse-run-bar__issues li.is-critical {
  color: var(--dq-danger);
  font-weight: 600;
}

.lv-parse-run-bar__ok {
  color: var(--dq-success);
  font-weight: 700;
}

.lv-parse-run-bar__title {
  font-size: var(--dq-font-size-body);
  font-weight: 650;
  color: var(--dq-label-primary);
}

.lv-parse-run-bar__id {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
}

.lv-parse-run-bar__meta {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
}

.lv-parse-run-bar__warn {
  color: var(--dq-warning);
}

.lv-parse-run-bar__actions {
  margin-top: 4px;
}

.lv-parse-run-bar__body {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 0.5px solid var(--dq-border-subtle);
}

.lv-parse-run-bar__phases {
  margin: 0 0 8px;
  padding-left: 18px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.lv-parse-run-bar__phase {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
}

.lv-parse-run-bar__phase-name {
  font-weight: 600;
  color: var(--dq-label-primary);
}

.lv-parse-run-bar__issues {
  margin: 0 0 8px;
  padding-left: 18px;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-warning);
}

.lv-parse-run-bar__project {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
}

.lv-parse-run-bar__loading {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
}

.lv-parse-run-bar__history {
  margin: 0 0 8px;
}

.lv-parse-run-bar__history-label {
  display: block;
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  color: var(--dq-label-secondary);
  margin-bottom: 4px;
}

.lv-parse-run-bar__history-list {
  margin: 0;
  padding: 0;
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.lv-parse-run-bar__history-btn {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  width: 100%;
  padding: 4px 6px;
  border-radius: 6px;
  border: 0.5px solid transparent;
  background: none;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
  cursor: pointer;
  text-align: left;
}

.lv-parse-run-bar__history-btn.is-active {
  border-color: color-mix(in srgb, var(--dq-accent) 35%, var(--dq-border-subtle));
  background: color-mix(in srgb, var(--dq-accent) 8%, transparent);
}

.lv-parse-run-bar__history-summary {
  margin: 6px 0 0;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
}
</style>
