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
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount, watch, provide } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import TopNav from '@/components/shell/TopNav.vue';
import OnboardingWizard from '@/components/onboarding/OnboardingWizard.vue';
import { useTasksStore } from '@/stores/tasks';
import { api } from '@/utils/api';
import { applyTheme, migrateThemeId } from '@/stores/theme';
import { getItem, DQ_STORAGE, setItem } from '@/utils/storage';
import type { PageKey, SystemInfo, Task } from '@/types';
import { appEvents } from '@/utils/appEvents';

const router = useRouter();
const route = useRoute();
const tasksStore = useTasksStore();

const activePage = ref<PageKey>('image_create');
/** Route the queue used to open as a drawer; queue state is now a real page. */
const TASK_QUEUE_PAGE: PageKey = 'task_queue';
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

/** Cross-feature entry point (LoRA run detail, gallery, …) for the task page. */
function onOpenGlobalTaskQueue(payload: { taskId?: string }) {
  const taskId = String(payload?.taskId || '').trim();
  void router.push({
    name: TASK_QUEUE_PAGE,
    query: taskId ? { id: taskId } : {},
  });
}

function onOpenTaskLog(payload: { taskId: string }) {
  onOpenGlobalTaskQueue({ taskId: payload.taskId });
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

  // First launch: no completion marker yet. Skipped after Finish /「不再提示」+ close.
  showOnboarding.value = !getItem(DQ_STORAGE.ONBOARDED);

  appEvents.on('open-global-task-queue', onOpenGlobalTaskQueue);
  appEvents.on('open-task-log', onOpenTaskLog);
  appEvents.on('open-onboarding', onOpenOnboarding);
  window.addEventListener('resize', onViewportResize);
});

function onOpenOnboarding(_: void) {
  showOnboarding.value = true;
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
</style>
