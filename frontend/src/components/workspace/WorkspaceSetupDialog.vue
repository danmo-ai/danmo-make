<template>
  <DqDialog
    v-model:open="visibleModel"
    class="dq-workspace-setup-dialog"
    :title="$t('settings.workspaceSetupTitle')"
    width="520px"
    :closable="false"
  >
    <p class="dq-workspace-setup-intro">{{ $t('settings.workspaceSetupIntro') }}</p>
    <p class="dq-workspace-setup-hint">{{ $t('settings.workspaceSetupEmptyHint') }}</p>
    <div class="settings-workspace-input-row dq-workspace-setup-input-row">
      <DqInput
        v-model="pickedPath"
        :placeholder="$t('settings.customWorkspacePlaceholder')"
        readonly
      />
      <DqButton size="sm" class="settings-workspace-pick-btn" :loading="picking" @click="pickDirectory">
        {{ $t('settings.workspaceSetupPick') }}
      </DqButton>
    </div>
    <p v-if="effectiveRoot" class="dq-workspace-setup-from">
      <span class="dq-workspace-setup-from-label">{{ $t('settings.workspaceLayoutTitle') }}</span>
      <span class="dq-workspace-setup-from-path">{{ effectiveRoot }}</span>
    </p>
    <template #footer>
      <DqButton type="primary" :loading="applying" :disabled="!pickedPath.trim()" @click="confirm">
        {{ $t('settings.workspaceSetupConfirm') }}
      </DqButton>
    </template>
  </DqDialog>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue';
import { toast } from '@/utils/feedback';
import { api } from '@/utils/api';
import { $tt } from '@/utils/i18n';

const props = defineProps<{
  visible: boolean;
  effectiveRoot?: string;
}>();

const emit = defineEmits<{
  'update:visible': [value: boolean];
  completed: [];
}>();

const visibleModel = computed({
  get: () => props.visible,
  set: (v: boolean) => emit('update:visible', v),
});

const pickedPath = ref('');
const picking = ref(false);
const applying = ref(false);

watch(
  () => props.visible,
  (open) => {
    if (open) {
      pickedPath.value = '';
    }
  },
);

function extractApiError(e: unknown): string {
  if (typeof e === 'object' && e !== null && 'response' in e) {
    const err = e as { response?: { data?: { detail?: string } } };
    if (err.response?.data?.detail) {
      return err.response.data.detail;
    }
  }
  if (e instanceof Error) return e.message;
  return String(e);
}

async function pickDirectory() {
  picking.value = true;
  try {
    const { path } = await api.settings.pickWorkspaceDirectory();
    if (path) {
      pickedPath.value = path;
    }
  } catch (e) {
    toast.error(extractApiError(e));
  } finally {
    picking.value = false;
  }
}

async function confirm() {
  const path = pickedPath.value.trim();
  if (!path) {
    toast.warning($tt('settings.workspaceRequired'));
    return;
  }
  applying.value = true;
  try {
    const res = await api.settings.applyWorkspace(path);
    emit('completed');
    visibleModel.value = false;
    if (res.restart_required) {
      toast.warning($tt('settings.customWorkspaceRestartHint'));
    } else {
      toast.success($tt('settings.saved'));
    }
  } catch (e) {
    toast.error(extractApiError(e) || $tt('settings.workspaceApplyFailed'));
  } finally {
    applying.value = false;
  }
}
</script>

<style scoped>
.dq-workspace-setup-intro {
  margin: 0 0 8px;
  line-height: 1.5;
  color: var(--dq-label-secondary);
}
.dq-workspace-setup-hint {
  margin: 0 0 16px;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
}
.dq-workspace-setup-input-row {
  margin-bottom: 12px;
}
.dq-workspace-setup-from {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
  color: var(--dq-label-tertiary);
}
.dq-workspace-setup-from-label {
  display: block;
  margin-bottom: 4px;
}
.dq-workspace-setup-from-path {
  word-break: break-all;
  font-family: var(--dq-font-mono);
}
</style>
