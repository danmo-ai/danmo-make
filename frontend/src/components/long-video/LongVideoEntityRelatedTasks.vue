<template>
  <div v-if="projectId && (row || loading)" class="lv-entity-task">
    <span class="lv-entity-task__label">{{ label }}</span>
    <template v-if="loading">
      <span class="lv-entity-task__hint">{{ $tt('common.loading') }}</span>
    </template>
    <template v-else-if="row">
      <code class="lv-entity-task__tid" :title="row.taskId">{{ shortId(row.taskId) }}</code>
      <DqTag size="sm" effect="plain" :type="statusTagType(row.status)">
        {{ statusLabel(row.status) }}
      </DqTag>
      <DqButton size="sm" type="text" @click="openLog(row.taskId)">
        {{ $tt('video.longVideoRelatedTasksViewLog') }}
      </DqButton>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { api } from '@/utils/api';
import { openTaskLog } from '@/utils/appEvents';
import {
  activityStatusLabelKey,
  latestEntityTask,
  shortActivityId,
} from '@/utils/longVideoActivity';

const props = defineProps<{
  projectId?: string;
  phase: string;
  match: Record<string, string>;
  label: string;
  refreshToken?: string | null;
}>();

const { t: $tt } = useI18n();
const loading = ref(false);
const items = ref<Awaited<ReturnType<typeof api.longVideo.listProjectActivity>>['items']>([]);

const row = computed(() => {
  const match = props.match ?? {};
  if (!Object.values(match).every((v) => String(v || '').trim())) return null;
  return latestEntityTask(items.value, props.phase, match);
});

function shortId(id: string) {
  return shortActivityId(id);
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
  if (!projectId) {
    items.value = [];
    return;
  }
  loading.value = true;
  try {
    const res = await api.longVideo.listProjectActivity(projectId, {
      phase: props.phase,
      limit: 40,
    });
    items.value = res.items ?? [];
  } catch {
    items.value = [];
  } finally {
    loading.value = false;
  }
}

watch(
  () => [props.projectId, props.phase, JSON.stringify(props.match), props.refreshToken] as const,
  () => {
    void load();
  },
  { immediate: true },
);
</script>

<style scoped>
.lv-entity-task {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  margin-top: 8px;
  padding-top: 8px;
  border-top: 0.5px dashed var(--dq-border-subtle);
  font-size: var(--dq-font-size-caption);
}

.lv-entity-task__label {
  font-weight: 600;
  color: var(--dq-label-secondary);
}

.lv-entity-task__tid {
  color: var(--dq-label-secondary);
}

.lv-entity-task__hint {
  color: var(--dq-label-tertiary);
}
</style>
