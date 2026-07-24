<template>
  <section v-if="projectId && shotId" class="lv-related-tasks">
    <button type="button" class="lv-related-tasks__toggle" @click="expanded = !expanded">
      <span>{{ $tt('video.longVideoRelatedTasksTitle') }}</span>
      <span class="lv-related-tasks__chev" :class="{ 'is-open': expanded }" aria-hidden="true">▾</span>
    </button>
    <div v-show="expanded" class="lv-related-tasks__body">
      <p v-if="loading" class="lv-related-tasks__hint">{{ $tt('common.loading') }}</p>
      <p v-else-if="!rows.length" class="lv-related-tasks__hint">{{ $tt('video.longVideoRelatedTasksEmpty') }}</p>
      <ul v-else class="lv-related-tasks__list">
        <li v-for="row in rows" :key="row.phase" class="lv-related-tasks__row">
          <span class="lv-related-tasks__label">{{ row.label }}</span>
          <code class="lv-related-tasks__tid" :title="row.taskId">{{ shortId(row.taskId) }}</code>
          <DqTag size="sm" effect="plain" :type="statusTagType(row.status)">
            {{ statusLabel(row.status) }}
          </DqTag>
          <span v-if="row.at" class="lv-related-tasks__time">{{ formatTime(row.at) }}</span>
          <DqButton size="sm" type="text" @click="openLog(row.taskId)">
            {{ $tt('video.longVideoRelatedTasksViewLog') }}
          </DqButton>
        </li>
      </ul>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { api } from '@/utils/api';
import { openTaskLog } from '@/utils/appEvents';
import {
  activityStatusLabelKey,
  formatActivityTime,
  latestShotTasksByPhase,
  shortActivityId,
} from '@/utils/longVideoActivity';

const props = defineProps<{
  projectId?: string;
  shotId?: string;
  /** Which generation phases to show (defaults: keyframe + segment). */
  phases?: string[];
  /** Bump to reload activity (e.g. after generation finishes). */
  refreshToken?: string;
}>();

const { t: $tt, locale } = useI18n();
const expanded = ref(false);
const loading = ref(false);
const items = ref<Awaited<ReturnType<typeof api.longVideo.listProjectActivity>>['items']>([]);

const phaseList = computed(() => props.phases ?? ['keyframe', 'segment']);

const rows = computed(() => {
  const map = latestShotTasksByPhase(items.value, phaseList.value);
  return phaseList.value
    .map((phase) => {
      const row = map.get(phase);
      if (!row) return null;
      return {
        phase,
        label: phaseLabel(phase),
        taskId: row.taskId,
        status: row.status,
        at: row.at,
      };
    })
    .filter(Boolean) as Array<{ phase: string; label: string; taskId: string; status: string; at: string }>;
});

function phaseLabel(phase: string) {
  const keys: Record<string, string> = {
    keyframe: 'video.longVideoRelatedTasksKeyframe',
    segment: 'video.longVideoRelatedTasksSegment',
    cast_portrait: 'video.longVideoRelatedTasksPortrait',
    scene_ref: 'video.longVideoRelatedTasksSceneRef',
  };
  const key = keys[phase] ?? phase;
  const label = $tt(key);
  return label !== key ? label : phase;
}

function shortId(id: string) {
  return shortActivityId(id);
}

function formatTime(iso: string) {
  return formatActivityTime(iso, locale.value);
}

function statusLabel(status: string) {
  const key = activityStatusLabelKey(status);
  const label = $tt(key);
  return label !== key ? label : status;
}

function statusTagType(status: string): 'success' | 'warning' | 'danger' | 'info' {
  if (status === 'completed') return 'success';
  if (status === 'failed') return 'danger';
  if (status === 'cancelled') return 'warning';
  return 'info';
}

function openLog(taskId: string) {
  openTaskLog(taskId);
}

async function load() {
  const projectId = props.projectId?.trim();
  const shotId = props.shotId?.trim();
  if (!projectId || !shotId) {
    items.value = [];
    return;
  }
  loading.value = true;
  try {
    const res = await api.longVideo.listProjectActivity(projectId, {
      shot_id: shotId,
      limit: 80,
    });
    items.value = res.items ?? [];
  } catch {
    items.value = [];
  } finally {
    loading.value = false;
  }
}

watch(
  () => [props.projectId, props.shotId, expanded.value, props.refreshToken] as const,
  ([pid, sid, isOpen]) => {
    if (pid && sid && isOpen) void load();
  },
  { immediate: true },
);

defineExpose({ reload: load });
</script>

<style scoped>
.lv-related-tasks {
  margin-top: 12px;
  padding-top: 10px;
  border-top: 0.5px solid var(--dq-border-subtle);
}

.lv-related-tasks__toggle {
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

.lv-related-tasks__chev {
  transition: transform 0.15s ease;
}

.lv-related-tasks__chev.is-open {
  transform: rotate(180deg);
}

.lv-related-tasks__body {
  margin-top: 8px;
}

.lv-related-tasks__hint {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
}

.lv-related-tasks__list {
  margin: 0;
  padding: 0;
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.lv-related-tasks__row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  font-size: var(--dq-font-size-caption);
}

.lv-related-tasks__label {
  min-width: 4.5em;
  font-weight: 600;
  color: var(--dq-label-primary);
}

.lv-related-tasks__tid {
  color: var(--dq-label-secondary);
}

.lv-related-tasks__time {
  color: var(--dq-label-tertiary);
}
</style>
