<template>
  <div
    class="main-container dq-app-shell dq-app-shell--sidebar"
    :class="{ 'is-sidebar-collapsed': sidebarCollapsed }"
  >
    <aside class="app-sidebar dq-app-sidebar">
      <TopNav
        :active-page="activePage"
        :queue-count="globalQueueCount"
        :collapsed="sidebarCollapsed"
        @navigate="handleNavSelect"
        @open-queue="openTaskQueue"
        @toggle-collapse="toggleSidebar"
      />
    </aside>

    <main class="app-main dq-app-main">
      <router-view />
    </main>
  </div>

  <OnboardingWizard
    v-model:open="showOnboarding"
    @completed="onOnboardingCompleted"
  />

  <DqDrawer
    v-model:open="showGlobalQueueDrawer"
    class="dq-task-queue-drawer"
    :title="$tt('studio.queueDialogTitle')"
    direction="rtl"
    size="420px"
  >
    <DqInspectorEmpty
      v-if="globalTaskQueue.running.length === 0 && globalTaskQueue.queued.length === 0"
      class="dq-task-queue-empty"
    >
      {{ $tt('studio.queueEmpty') }}
    </DqInspectorEmpty>
    <div v-else>
      <div v-if="globalTaskQueue.running.length > 0" class="dq-task-queue-section">
        <div class="dq-task-queue-heading dq-task-queue-heading--primary">
          {{ $tt('studio.running') }}
        </div>
        <div
          v-for="task in globalTaskQueue.running"
          :key="task.id"
          class="queue-dialog-item running"
        >
          <div class="dq-task-queue-item-head">
            <div class="dq-task-queue-item-main">
              <div class="dq-task-queue-kind">
                {{ taskKindLabel(task.kind) }}
              </div>
              <div class="dq-task-queue-model">
                {{ task.params?.model || $tt('queue.unspecifiedModel') }}
              </div>
              <div class="dq-queue-prompt-line">{{ queueTaskLabel(task) }}</div>
              <TaskIdBadge :task-id="task.id" compact />
            </div>
            <DqIconButton
              type="danger"
              size="sm"
              class="dq-task-queue-cancel-btn dq-icon-btn--circle"
              :label="$tt('studio.cancelTask')"
              @click="cancelGlobalTask(task.id)"
            >
              <DqIcon><delete /></DqIcon>
            </DqIconButton>
          </div>
          <DqProgress
            class="dq-queue-progress"
            :percentage="Math.round((task.progress || 0) * 100)"
            :stroke-width="4"
          />
          <div
            v-if="(task.total || 0) > 0"
            class="dq-task-queue-progress-hint"
          >
            <template v-if="String(task.kind || '').startsWith('image.')">
              {{ $tt('studio.queueDenoiseProgress', { current: task.step != null ? task.step : 0, total: task.total || 0 }) }}
            </template>
            <template v-else>
              {{ $tt('studio.queueStepProgress', { current: task.step != null ? task.step : 0, total: task.total || 0 }) }}
            </template>
          </div>
          <div
            v-if="task.progressMessage === 'post' && typeof task.progress === 'number' && task.progress < 1"
            class="dq-task-queue-progress-hint dq-task-queue-progress-hint--tight"
          >
            {{ $tt('studio.queuePostProcessHint') }}
          </div>
          <DqButton
            type="text"
            size="sm"
            class="dq-task-queue-logs-btn"
            @click="openQueueTaskLogs(task.id)"
          >
            {{ $tt('studio.viewLogs') }}
          </DqButton>
        </div>
      </div>

      <div v-if="globalTaskQueue.queued.length > 0">
        <div class="dq-task-queue-heading dq-task-queue-heading--muted">
          {{ $tt('studio.queued') }} ({{ globalTaskQueue.queued.length }})
        </div>
        <div
          v-for="(task, index) in globalTaskQueue.queued"
          :key="task.id"
          class="queue-dialog-item queued"
        >
          <div class="dq-task-queue-queued-layout">
            <div class="dq-task-queue-queued-left">
              <span class="dq-task-queue-idx">#{{ index + 1 }}</span>
              <div class="dq-task-queue-queued-main">
                <div class="dq-task-queue-row-meta">
                  <span>{{ taskKindLabel(task.kind) }}</span>
                  <DqTag
                    v-if="(task.priority ?? 100) <= 50"
                    type="warning"
                    effect="plain"
                    size="sm"
                    class="dq-task-queue-priority-tag"
                  >
                    {{ $tt('studio.queuePriorityHigh') }}
                  </DqTag>
                </div>
                <div class="dq-task-queue-row-model">
                  {{ task.params?.model || $tt('queue.unspecifiedModel') }}
                </div>
                <div class="dq-queue-prompt-line">{{ queueTaskLabel(task) }}</div>
                <TaskIdBadge :task-id="task.id" compact />
                <div
                  v-if="task.estimated_wait_seconds != null"
                  class="dq-task-queue-wait"
                >
                  {{ $tt('queue.estimatedWait', { s: task.estimated_wait_seconds }) }}
                </div>
              </div>
            </div>
            <div class="dq-task-queue-side-actions">
              <DqButton
                size="sm"
                @click="setQueuedPriority(task.id, 'high')"
                :disabled="(task.priority ?? 100) <= 50"
              >
                {{ $tt('studio.queueSetHigh') }}
              </DqButton>
              <DqButton
                size="sm"
                @click="setQueuedPriority(task.id, 'normal')"
                :disabled="(task.priority ?? 100) > 50"
              >
                {{ $tt('studio.queueSetNormal') }}
              </DqButton>
              <DqIconButton
                type="danger"
                size="sm"
                class="dq-task-queue-cancel-end dq-icon-btn--circle"
                :label="$tt('studio.cancelTask')"
                @click="cancelGlobalTask(task.id)"
              >
                <DqIcon><delete /></DqIcon>
              </DqIconButton>
            </div>
          </div>
        </div>
      </div>
    </div>
  </DqDrawer>

  <GenTaskLogDialog
    v-model:open="showQueueTaskLogs"
    :task-id="queueLogTaskId"
  />
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount, watch, provide } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import { toast } from '@/utils/feedback';
import TopNav from '@/components/shell/TopNav.vue';
import GenTaskLogDialog from '@/components/studio/GenTaskLogDialog.vue';
import TaskIdBadge from '@/components/studio/TaskIdBadge.vue';
import OnboardingWizard from '@/components/onboarding/OnboardingWizard.vue';
import { useTasksStore } from '@/stores/tasks';
import { api } from '@/utils/api';
import { applyTheme, migrateThemeId } from '@/stores/theme';
import { $tt } from '@/utils/i18n';
import { getItem, DQ_STORAGE, setItem } from '@/utils/storage';
import type { PageKey, SystemInfo, Task } from '@/types';
import { appEvents, openOnboarding } from '@/utils/appEvents';

const router = useRouter();
const route = useRoute();
const tasksStore = useTasksStore();

const activePage = ref<PageKey>('image_create');
const showGlobalQueueDrawer = ref(false);
const showQueueTaskLogs = ref(false);
const queueLogTaskId = ref<string | null>(null);
const storedSidebarState = getItem(DQ_STORAGE.NAV_COLLAPSED);
const sidebarPreference = ref<boolean | null>(
  storedSidebarState == null ? null : storedSidebarState === '1',
);
const compactViewport = ref(window.innerWidth <= 1180);
const sidebarCollapsed = computed(
  () => sidebarPreference.value ?? compactViewport.value,
);

function toggleSidebar() {
  sidebarPreference.value = !sidebarCollapsed.value;
  setItem(DQ_STORAGE.NAV_COLLAPSED, sidebarPreference.value ? '1' : '0');
}

function onViewportResize() {
  compactViewport.value = window.innerWidth <= 1180;
}

function openQueueTaskLogs(taskId: string) {
  queueLogTaskId.value = taskId;
  showQueueTaskLogs.value = true;
}
const currentLang = ref('zh');
const showOnboarding = ref(false);

const systemInfo = ref<SystemInfo>({
  env_ready: false,
  platform: '',
  architecture: '',
  memory_gb: 0,
  mlx_memory_limit: 120,
});

const globalTaskQueue = computed(() => {
  const running = tasksStore.queueState.running.map((t: Task) => {
    const live = tasksStore.liveTaskProgress[t.id];
    return live ? { ...t, ...live } : t;
  });
  const queued = tasksStore.queueState.queued.map((t: Task) => {
    const live = tasksStore.liveTaskProgress[t.id];
    return live ? { ...t, ...live } : t;
  });
  return { running, queued };
});

const globalQueueCount = computed(
  () => globalTaskQueue.value.running.length + globalTaskQueue.value.queued.length
);



watch(
  () => route.name as PageKey,
  (newVal) => {
    if (newVal) {
      activePage.value = newVal;
    }
  },
  { immediate: true }
);

function handleNavSelect(index: string) {
  router.push({ name: index });
}

function openTaskQueue() {
  showGlobalQueueDrawer.value = true;
}

async function cancelGlobalTask(taskId: string) {
  try {
    const result = await api.gen.cancelMediaTask(taskId);
    if (!result || result.ok !== true) {
      throw new Error('cancel failed');
    }
    await tasksStore.pollQueueOnce();
    toast.success($tt('studio.cancelled'));
  } catch (e: unknown) {
    console.error('cancelGlobalTask', e);
    let msg = '';
    if (typeof e === 'object' && e !== null && 'response' in e) {
      const err = e as { response?: { data?: { detail?: string } } };
      msg = err.response?.data?.detail || '';
    }
    if (!msg && e instanceof Error) msg = e.message;
    if (!msg) msg = String(e);
    toast.error($tt('studio.error', { msg }));
  }
}

async function setQueuedPriority(taskId: string, priority: string) {
  try {
    await api.gen.patchMediaTaskPriority(taskId, { priority });
    await tasksStore.pollQueueOnce();
    toast.success($tt('studio.priorityUpdated'));
  } catch (e: unknown) {
    console.error('setQueuedPriority', e);
    let msg = '';
    if (typeof e === 'object' && e !== null && 'response' in e) {
      const err = e as { response?: { data?: { detail?: string } } };
      msg = err.response?.data?.detail || '';
    }
    if (!msg && e instanceof Error) msg = e.message;
    toast.error($tt('studio.error', { msg: msg || String(e) }));
  }
}

async function loadSystemInfo() {
  try {
    const info = await api.settings.getSystemInfo();
    systemInfo.value = info;
  } catch (e) {
    console.error('Failed to load system info:', e);
  }
}

function onOnboardingCompleted() {
  setItem(DQ_STORAGE.ONBOARDED, '1');
  showOnboarding.value = false;
}

function queueTaskLabel(task: { params?: { title?: string; prompt?: string } }) {
  const title = String(task.params?.title || '').trim();
  const prompt = String(task.params?.prompt || '').trim();
  return queueTruncate(title || prompt, 40);
}

function queueTruncate(text: string, length: number): string {
  if (!text) return '';
  return text.length > length ? text.substring(0, length) + '...' : text;
}

function taskKindLabel(kind?: string): string {
  if (!kind) return '';
  const key = 'taskKind.' + String(kind).replace(/\./g, '_');
  const result = $tt(key);
  return result && result !== key ? result : kind;
}

let sysInfoInterval: ReturnType<typeof setInterval> | null = null;

onMounted(async () => {
  const savedLang = getItem(DQ_STORAGE.LANG);
  if (savedLang) {
    currentLang.value = savedLang;
  }

  await loadSystemInfo();

  const savedTheme = migrateThemeId(getItem(DQ_STORAGE.THEME));
  if (savedTheme) {
    applyTheme(savedTheme);
  }

  sysInfoInterval = setInterval(loadSystemInfo, 30000);

  tasksStore.ensureQueuePoller();

  // First launch: no completion marker yet. The wizard is skipped once the user
  // finishes it (Finish), which writes ONBOARDED.
  showOnboarding.value = !getItem(DQ_STORAGE.ONBOARDED);

  appEvents.on('open-global-task-queue', onOpenGlobalTaskQueue);
  appEvents.on('open-task-log', onOpenTaskLog);
  appEvents.on('open-onboarding', onOpenOnboarding);
  window.addEventListener('resize', onViewportResize);
});

function onOpenOnboarding(_: void) {
  showOnboarding.value = true;
}

function onOpenGlobalTaskQueue(_: void) {
  showGlobalQueueDrawer.value = true;
}

function onOpenTaskLog(payload: { taskId: string }) {
  openQueueTaskLogs(payload.taskId);
}

onBeforeUnmount(() => {
  appEvents.off('open-global-task-queue', onOpenGlobalTaskQueue);
  appEvents.off('open-task-log', onOpenTaskLog);
  appEvents.off('open-onboarding', onOpenOnboarding);
  window.removeEventListener('resize', onViewportResize);
  if (sysInfoInterval) {
    clearInterval(sysInfoInterval);
  }
  tasksStore.releaseQueuePoller();
});

provide('systemInfo', systemInfo);
</script>

<style scoped>
.main-container {
  height: 100vh;
}
.app-sidebar {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  width: var(--dq-shell-sidebar-width, 60px);
  flex-shrink: 0;
  padding-top: env(safe-area-inset-top, 0px);
}
.app-main {
  flex: 1;
  padding: 16px 20px 20px;
  overflow-y: auto;
  min-width: 0;
  box-sizing: border-box;
}
.dq-task-queue-logs-btn {
  margin-top: 8px;
  align-self: flex-start;
}
</style>