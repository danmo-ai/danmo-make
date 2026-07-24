<template>
  <div class="lora-train-history" :class="`lora-train-history--${variant}`">
    <div v-if="loading && !runs.length" class="lora-train-history__skeleton">
      <div v-for="i in skeletonCount" :key="i" class="lora-train-history__skeleton-row" />
    </div>

    <DqEmpty
      v-else-if="!loading && !runs.length"
      :description="$t('loraTrain.noRecentRuns')"
      class="lora-train-history__empty"
    />

    <div v-else class="lora-train-history__list">
      <button
        v-for="run in runs"
        :key="runTaskId(run)"
        type="button"
        class="lora-train-history__item"
        :class="[
          `is-${String(run.status || 'unknown')}`,
          { 'is-active': runTaskId(run) === activeId },
        ]"
        @click="emit('select', runTaskId(run))"
      >
        <span class="lora-train-history__status-bar" aria-hidden="true" />

        <span class="lora-train-history__item-content">
          <span class="lora-train-history__item-top">
            <span
              class="lora-train-history__item-name"
              :title="runOutputName(run)"
            >
              {{ runOutputName(run) }}
            </span>
            <span v-if="variant === 'sidebar'" class="lora-train-history__item-badges">
              <DqTag size="sm" :type="statusType(String(run.status || ''))" effect="plain">
                {{ statusLabel(String(run.status || '')) }}
              </DqTag>
              <DqTag
                v-if="userLoraForRun(run)"
                size="sm"
                type="success"
                effect="plain"
                class="lora-train-history__registered"
              >
                {{ $t('loraTrain.registeredBadge') }}
              </DqTag>
            </span>
            <template v-else>
              <DqTag size="sm" :type="statusType(String(run.status || ''))" effect="plain">
                {{ statusLabel(String(run.status || '')) }}
              </DqTag>
              <DqTag
                v-if="userLoraForRun(run)"
                size="sm"
                type="success"
                effect="plain"
                class="lora-train-history__registered"
              >
                {{ $t('loraTrain.registeredBadge') }}
              </DqTag>
            </template>
          </span>
          <span class="lora-train-history__item-meta">
            <span class="lora-train-history__base">{{ runBaseModel(run) }}</span>
            <span class="lora-train-history__sep" aria-hidden="true">·</span>
            <span class="lora-train-history__when">{{ formatWhen(run) }}</span>
          </span>
        </span>

        <DqIcon v-if="variant === 'page'" class="lora-train-history__chevron" aria-hidden="true">
          <arrow-right />
        </DqIcon>
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted } from 'vue';
import { useI18n } from 'vue-i18n';
import { ArrowRight } from '@danqing/dq-shell';
import {
  runBaseModel,
  runOutputName,
  runTaskId,
  useLoraTrainLibrary,
} from '@/composables/useLoraTrainLibrary';

const props = withDefaults(
  defineProps<{
    activeId?: string;
    variant?: 'sidebar' | 'page';
    limit?: number;
  }>(),
  {
    activeId: '',
    variant: 'sidebar',
    limit: 12,
  }
);

const emit = defineEmits<{
  (e: 'select', taskId: string): void;
}>();

const { t, locale } = useI18n();
const { runs, loading, refresh, userLoraForRun } = useLoraTrainLibrary();

const skeletonCount = computed(() => (props.variant === 'page' ? 5 : 3));

function statusType(status: string): string {
  if (status === 'completed') return 'success';
  if (status === 'failed') return 'danger';
  if (status === 'running') return 'primary';
  if (status === 'queued') return 'info';
  return 'info';
}

function statusLabel(status: string): string {
  const key = `loraTrain.runStatus.${status}`;
  const translated = t(key);
  return translated !== key ? translated : status;
}

function formatWhen(run: Record<string, unknown>): string {
  const raw = run.created_at || run.started_at || run.updated_at;
  if (!raw) return '';
  try {
    return new Date(String(raw)).toLocaleString(locale.value === 'zh' ? 'zh-CN' : undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return String(raw);
  }
}

function reload() {
  return refresh({ limit: props.limit });
}

onMounted(() => {
  void reload();
});

defineExpose({
  refresh: reload,
});
</script>

<style scoped>
.lora-train-history {
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.lora-train-history__title {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--dq-label-tertiary);
  white-space: nowrap;
  flex-shrink: 0;
}

.lora-train-history--page .lora-train-history__title {
  font-size: var(--dq-font-size-body);
  font-weight: 600;
  letter-spacing: normal;
  text-transform: none;
  color: var(--dq-label-primary);
}

.lora-train-history__count {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 22px;
  height: 22px;
  padding: 0 7px;
  border-radius: 999px;
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  color: var(--dq-label-secondary);
  background: var(--dq-fill-tertiary);
}

.lora-train-history__empty {
  padding: 8px 0;
}

.lora-train-history--page .lora-train-history__empty {
  padding: 24px 0;
}

.lora-train-history__skeleton {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.lora-train-history__skeleton-row {
  height: 52px;
  border-radius: var(--radius-sm);
  background: linear-gradient(
    90deg,
    var(--dq-fill-tertiary) 0%,
    var(--dq-fill-secondary) 50%,
    var(--dq-fill-tertiary) 100%
  );
  background-size: 200% 100%;
  animation: lora-train-history-shimmer 1.2s ease-in-out infinite;
}

@keyframes lora-train-history-shimmer {
  0% {
    background-position: 100% 0;
  }
  100% {
    background-position: -100% 0;
  }
}

.lora-train-history__list {
  display: flex;
  flex-direction: column;
  gap: 6px;
  flex: 1;
  min-height: 0;
  overflow-y: auto;
}

.lora-train-history--page .lora-train-history__list {
  gap: 8px;
}

.lora-train-history__item {
  position: relative;
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  padding: 10px 12px;
  border: 1px solid transparent;
  border-radius: var(--radius-sm);
  background: transparent;
  cursor: pointer;
  text-align: left;
  transition: background 0.15s ease, border-color 0.15s ease;
}

.lora-train-history--sidebar .lora-train-history__item {
  padding: 8px 8px;
  gap: 8px;
}

.lora-train-history--page .lora-train-history__item {
  padding: 12px 14px;
  border: 1px solid var(--dq-border-subtle);
  border-radius: 12px;
  background: var(--dq-fill-control);
}

.lora-train-history__item:hover {
  background: var(--dq-fill-tertiary);
  border-color: color-mix(in srgb, var(--dq-accent) 22%, var(--dq-border-subtle));
}

.lora-train-history--page .lora-train-history__item:hover {
  background: var(--dq-surface-inset-hover, var(--dq-fill-tertiary));
}

.lora-train-history__item.is-active {
  background: color-mix(in srgb, var(--dq-accent) 10%, var(--dq-fill-secondary));
  border-color: color-mix(in srgb, var(--dq-accent) 35%, transparent);
}

.lora-train-history__status-bar {
  flex-shrink: 0;
  width: 3px;
  align-self: stretch;
  border-radius: 999px;
  background: var(--dq-label-quaternary);
}

.lora-train-history__item.is-completed .lora-train-history__status-bar {
  background: var(--dq-success);
}

.lora-train-history__item.is-failed .lora-train-history__status-bar {
  background: var(--dq-danger);
}

.lora-train-history__item.is-running .lora-train-history__status-bar {
  background: var(--dq-accent);
}

.lora-train-history__item.is-queued .lora-train-history__status-bar {
  background: var(--dq-info, var(--dq-accent));
}

.lora-train-history__item-content {
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex: 1;
  min-width: 0;
}

.lora-train-history__item-top {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
}

.lora-train-history--sidebar .lora-train-history__item-top {
  flex-direction: column;
  align-items: flex-start;
  gap: 4px;
}

.lora-train-history__item-name {
  font-size: var(--dq-font-size-body);
  font-weight: 600;
  color: var(--dq-label-primary);
  word-break: break-word;
}

.lora-train-history--sidebar .lora-train-history__item-name {
  width: 100%;
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  word-break: normal;
}

.lora-train-history--sidebar .lora-train-history__item-badges {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px;
  max-width: 100%;
}

.lora-train-history--sidebar .lora-train-history__item-badges :deep(.dq-tag) {
  flex-shrink: 0;
  max-width: 100%;
}

.lora-train-history--page .lora-train-history__item-name {
  font-size: var(--dq-font-size-body);
}

.lora-train-history__registered {
  flex-shrink: 0;
}

.lora-train-history__item-meta {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
}

.lora-train-history--sidebar .lora-train-history__item-meta {
  flex-wrap: nowrap;
  min-width: 0;
  gap: 4px;
}

.lora-train-history--sidebar .lora-train-history__base {
  flex: 1;
  min-width: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.lora-train-history--sidebar .lora-train-history__when {
  flex-shrink: 0;
}

.lora-train-history__base {
  color: var(--dq-label-secondary);
}

.lora-train-history__chevron {
  flex-shrink: 0;
  font-size: var(--dq-font-size-body);
  color: var(--dq-label-quaternary);
  transition: transform 0.15s ease, color 0.15s ease;
}

.lora-train-history__item:hover .lora-train-history__chevron {
  color: var(--dq-accent);
  transform: translateX(2px);
}
</style>
