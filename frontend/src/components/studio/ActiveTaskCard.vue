<template>
  <div class="active-task-card">
    <div class="active-task-card__media">
      <!-- Preview image -->
      <div v-if="previewUrl" class="active-task-card__preview">
        <img :src="previewUrl" alt="preview" />
      </div>

      <!-- Generating animation placeholder -->
      <div v-else class="active-task-card__generating">
        <div class="active-task-card__spinner">
          <DqIcon size="32" class="is-spinning"><Loading /></DqIcon>
        </div>
        <span class="active-task-card__status-text">{{ statusText }}</span>
      </div>

      <!-- Progress overlay -->
      <div class="active-task-card__progress-bar">
        <div
          class="active-task-card__progress-fill"
          :style="{ width: progressPercent + '%' }"
        />
      </div>

      <!-- Step info -->
      <div v-if="stepInfo" class="active-task-card__step-info">
        {{ stepInfo }}
      </div>
    </div>

    <div class="active-task-card__footer">
      <span class="active-task-card__prompt" :title="taskPrompt">
        {{ truncate(taskPrompt, 40) || $t('studio.generating') }}
      </span>
      <DqIconButton
        type="danger"
        size="sm"
        class="active-task-card__cancel"
        :label="$t('studio.cancelTask')"
        @click.stop="$emit('cancel', task.id)"
      >
        <DqIcon :size="12"><Close /></DqIcon>
      </DqIconButton>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { Close, Loading } from '@danqing/dq-shell';
import { api } from '@/utils/api';
import type { Task } from '@/types';

const props = defineProps<{
  task: Task;
  media: 'image' | 'video' | 'audio';
}>();

const emit = defineEmits<{
  (e: 'cancel', taskId: string): void;
}>();

const { t: $t } = useI18n();

const previewUrl = ref('');
let previewTimer: ReturnType<typeof setInterval> | null = null;
let previewBlobUrl: string | null = null;

const progressPercent = computed(() => {
  const p = props.task.progress || 0;
  return Math.round(p * 100);
});

const statusText = computed(() => {
  const s = props.task.status;
  if (s === 'submitting') return $t('studio.submitting');
  if (s === 'queued') return $t('studio.queued');
  if (s === 'running') return $t('studio.running');
  return $t('studio.generating');
});

const stepInfo = computed(() => {
  const t = props.task;
  if (t.status === 'running' && t.step != null && t.total && t.total > 0) {
    return `Step ${t.step}/${t.total}`;
  }
  if (t.status === 'queued' && t.estimated_wait_seconds != null) {
    return $t('queue.estimatedWait', { s: Math.round(t.estimated_wait_seconds) });
  }
  return '';
});

const taskPrompt = computed(() => {
  return props.task.params?.prompt || props.task.params?.title || '';
});

async function pollPreview() {
  if (!props.task.id || props.media !== 'image') return;
  const url = `${api.tasks.previewUrl(props.task.id)}?t=${Date.now()}`;
  try {
    const res = await fetch(url, { method: 'GET', cache: 'no-store' });
    if (!res.ok) return;
    const blob = await res.blob();
    if (!blob.size) return;
    if (previewBlobUrl) {
      URL.revokeObjectURL(previewBlobUrl);
    }
    previewBlobUrl = URL.createObjectURL(blob);
    previewUrl.value = previewBlobUrl;
  } catch {
    // 404 until first preview is written
  }
}

function startPolling() {
  stopPolling();
  if (props.media !== 'image') return;
  pollPreview();
  previewTimer = setInterval(pollPreview, 1500);
}

function stopPolling() {
  if (previewTimer) {
    clearInterval(previewTimer);
    previewTimer = null;
  }
  if (previewBlobUrl) {
    URL.revokeObjectURL(previewBlobUrl);
    previewBlobUrl = null;
  }
}

watch(
  () => props.task.status,
  (status) => {
    if (status === 'completed' || status === 'failed' || status === 'cancelled') {
      stopPolling();
    }
  }
);

onMounted(() => {
  if (props.task.status === 'running' || props.task.status === 'queued') {
    startPolling();
  }
});

onBeforeUnmount(() => {
  stopPolling();
});

function truncate(text: string, length: number): string {
  if (!text) return '';
  return text.length > length ? text.substring(0, length) + '…' : text;
}
</script>

<style scoped>
.active-task-card {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.active-task-card__media {
  position: relative;
  aspect-ratio: 1 / 1;
  overflow: hidden;
  border-radius: var(--dq-radius-group);
  border: 0.5px solid color-mix(in srgb, var(--dq-accent) 35%, var(--dq-glass-border));
  background: var(--dq-surface-inset);
  box-shadow: var(--dq-shadow-sm);
}

.active-task-card__preview {
  width: 100%;
  height: 100%;
}

.active-task-card__preview img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.active-task-card__generating {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  color: var(--dq-label-secondary);
}

.active-task-card__spinner {
  opacity: 0.5;
}

.is-spinning {
  animation: spin 1.5s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.active-task-card__status-text {
  font-size: var(--dq-font-size-caption);
  font-weight: 500;
}

/* Progress bar */
.active-task-card__progress-bar {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: var(--dq-overlay-light);
}

.active-task-card__progress-fill {
  height: 100%;
  background: var(--dq-accent);
  transition: width 0.3s ease;
}

/* Step info */
.active-task-card__step-info {
  position: absolute;
  bottom: 8px;
  right: 8px;
  padding: 2px 8px;
  background: var(--dq-overlay-card);
  color: var(--dq-label-on-media);
  font-size: var(--dq-font-size-caption);
  font-weight: 500;
  border-radius: 4px;
  backdrop-filter: var(--dq-glass-blur-light);
  -webkit-backdrop-filter: var(--dq-glass-blur-light);
}

/* Footer */
.active-task-card__footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 2px;
  gap: 8px;
}

.active-task-card__prompt {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1;
  line-height: 1.4;
}

.active-task-card__cancel {
  flex-shrink: 0;
  opacity: 1;
  transition: opacity 140ms ease;
}

.active-task-card:hover .active-task-card__cancel {
  opacity: 1;
}
</style>
