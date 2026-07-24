<template>
  <div class="studio-canvas-session-controls">
    <template v-if="ready">
      <DqSelect
      :model-value="sessionId"
      size="sm"
      class="studio-canvas-session-controls__select"
      :placeholder="$t('canvas.sessionSelect')"
      @update:model-value="onSwitchSession"
    >
      <DqOption
        v-for="s in sessions"
        :key="s.id"
        :label="sessionLabel(s)"
        :value="s.id"
      />
    </DqSelect>

    <DqIconButton
      type="text"
      size="sm"
      :label="$t('canvas.newSession')"
      @click="onCreateSession"
    >
      <DqIcon :size="14"><Plus /></DqIcon>
    </DqIconButton>

    <DqIconButton
      type="text"
      size="sm"
      :label="$t('canvas.renameSession')"
      @click="startRename"
    >
      <DqIcon :size="14"><Document /></DqIcon>
    </DqIconButton>

    <DqIconButton
      v-if="sessions.length > 1"
      type="text"
      size="sm"
      :label="$t('canvas.deleteSession')"
      @click="onDeleteSession(sessionId)"
    >
      <DqIcon :size="14"><Delete /></DqIcon>
    </DqIconButton>

    <span v-if="syncing" class="studio-canvas-session-controls__sync">{{ $t('canvas.syncing') }}</span>
    </template>
    <span v-else class="studio-canvas-session-controls__loading">{{ $t('common.loading') }}</span>
  </div>

  <DqDialog
    v-model:open="renameOpen"
    :title="$t('canvas.renameSession')"
    width="360px"
  >
    <DqInput
      v-model="titleDraft"
      size="sm"
      :placeholder="$t('canvas.sessionSelect')"
      @keydown.enter="commitRename"
    />
    <template #footer>
      <DqButton size="sm" @click="renameOpen = false">{{ $t('common.cancel') }}</DqButton>
      <DqButton type="primary" @click="commitRename">{{ $t('common.confirm') }}</DqButton>
    </template>
  </DqDialog>
</template>

<script setup lang="ts">
import { ref, computed, unref } from 'vue';
import { Plus, Delete, Document } from '@danqing/dq-shell';
import { toast } from '@/utils/feedback';
import { $tt } from '@/utils/i18n';
import { useCanvasStore, type CanvasMedia } from '@/composables/useCanvasStore';
import type { CanvasSessionSummary } from '@/types';

const props = defineProps<{
  media: CanvasMedia;
}>();

const emit = defineEmits<{
  (e: 'composer-restore', snapshot: import('@/types').CanvasComposerSnapshot): void;
}>();

const store = useCanvasStore(props.media);

const ready = computed(() => unref(store.ready));
const sessionId = computed(() => unref(store.sessionId));
const sessions = computed(() => unref(store.sessions));
const syncing = computed(() => unref(store.syncing));

const renameOpen = ref(false);
const titleDraft = ref('');

function sessionLabel(s: CanvasSessionSummary): string {
  const n = s.item_count ?? 0;
  return `${s.title} (${n})`;
}

async function onSwitchSession(id: string) {
  await store.switchSession(id);
  emit('composer-restore', { ...store.composerSnapshot });
}

async function onCreateSession() {
  await store.createSession();
  emit('composer-restore', { ...store.composerSnapshot });
  toast.success($tt('canvas.sessionCreated'));
}

async function onDeleteSession(id: string) {
  const ok = await store.deleteSession(id);
  if (ok) {
    emit('composer-restore', { ...store.composerSnapshot });
    toast.success($tt('canvas.sessionDeleted'));
  }
}

function startRename() {
  titleDraft.value = unref(store.sessionTitle) || '';
  renameOpen.value = true;
}

async function commitRename() {
  renameOpen.value = false;
  const t = titleDraft.value.trim();
  if (!t || t === unref(store.sessionTitle)) return;
  await store.renameSession(t);
  toast.success($tt('canvas.sessionRenamed'));
}
</script>

<style scoped>
.studio-canvas-session-controls {
  display: flex;
  align-items: center;
  gap: 2px;
  min-width: 0;
  flex: 1;
}

.studio-canvas-session-controls__select {
  width: 148px;
  flex-shrink: 1;
  min-width: 108px;
}

.studio-canvas-session-controls__sync {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
  white-space: nowrap;
  padding-left: 4px;
}

.studio-canvas-session-controls__loading {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
  white-space: nowrap;
}
</style>
