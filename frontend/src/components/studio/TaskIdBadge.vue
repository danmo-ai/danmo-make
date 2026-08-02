<template>
  <div
    v-if="taskId"
    class="task-id-badge"
    :class="{ 'task-id-badge--compact': compact }"
  >
    <span class="task-id-badge__label">{{ $t('studio.taskId') }}</span>
    <code class="task-id-badge__value" :title="taskId">{{ taskId }}</code>
    <DqIconButton
      type="text"
      size="sm"
      class="task-id-badge__copy"
      :label="$t('studio.taskIdCopy')"
      @click="onCopy"
    >
      <DqIcon :size="14"><CopyDocument /></DqIcon>
    </DqIconButton>
    <p v-if="showHint && !compact" class="task-id-badge__hint">
      {{ $t('studio.taskIdAiHint') }}
    </p>
  </div>
</template>

<script setup lang="ts">
import { CopyDocument } from '@danqing/dq-shell';
import { useI18n } from 'vue-i18n';
import { toast } from '@/utils/feedback';
import { copyTextToClipboard } from '@/utils/clipboard';

const props = defineProps<{
  taskId?: string | null;
  /** Inline row for queue cards */
  compact?: boolean;
  /** Show API hint (log panel) */
  showHint?: boolean;
}>();

const { t: $t } = useI18n();

async function onCopy() {
  const id = String(props.taskId || '').trim();
  if (!id) return;
  const ok = await copyTextToClipboard(id);
  if (ok) {
    toast.success($t('studio.taskIdCopied'));
  } else {
    toast.error($t('gallery.copyFailed'));
  }
}
</script>

<style scoped>
.task-id-badge {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px 8px;
  min-width: 0;
}

.task-id-badge--compact {
  margin-top: 4px;
}

.task-id-badge__label {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  letter-spacing: var(--dq-tracking-wide);
  text-transform: uppercase;
  color: var(--dq-label-tertiary);
  flex-shrink: 0;
}

.task-id-badge__value {
  font-family: var(--dq-font-mono);
  font-size: var(--dq-font-size-caption);
  line-height: 1.35;
  padding: 2px 6px;
  border-radius: 4px;
  background: var(--dq-fill-on-glass);
  color: var(--dq-label-secondary);
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.task-id-badge__copy {
  flex-shrink: 0;
  margin-left: -2px;
}

.task-id-badge__hint {
  flex: 1 1 100%;
  margin: 0;
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
  color: var(--dq-label-tertiary);
}
</style>
