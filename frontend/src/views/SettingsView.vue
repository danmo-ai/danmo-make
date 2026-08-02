<!-- @ts-nocheck -->
<template>
  <div class="settings-page settings-page--v2 settings-page--macos">
    <!-- Left sidebar navigation -->
    <div class="settings-sidebar-wrapper">
      <DqSurfaceCard class="settings-sidebar-card">
        <div class="card-title">
          <DqIcon><Setting /></DqIcon>
          {{ $t('settings.title') }}
        </div>
        <div class="settings-sidebar-desc">
          {{ $t('settings.desc') }}
        </div>
        <nav class="dq-download-menu" role="navigation" :aria-label="$t('settings.title')">
          <button
            v-for="item in navItems"
            :key="item.id"
            type="button"
            class="dq-download-menu__item"
            :class="{ 'is-active': activeSection === item.id }"
            @click="activeSection = item.id"
          >
            <DqIcon class="dq-download-menu__icon"><component :is="item.icon" /></DqIcon>
            <span class="dq-download-menu__label">{{ $t(item.labelKey) }}</span>
          </button>
        </nav>

      </DqSurfaceCard>
    </div>

    <!-- Main content area -->
    <div class="settings-content-area">
      <DqSurfaceCard class="settings-tab-panel">
        <SystemSettingsSidebar
          v-if="activeSection === 'systeminfo'"
          :system-info="systemInfo"
          :cache-status="cacheStatus"
          :cache-loading="cacheLoading"
          :cache-error="cacheError"
          :monitor-data="monitorData"
          @refresh-cache="refreshCacheStatus"
        />
        <RuntimeSetupView v-else-if="activeSection === 'runtime'" />
        <SystemSettingsForm
          v-else
          :active-section="activeSection"
          :settings="settings"
          :workspace-paths="workspacePaths"
          :restore-config-busy="restoreConfigBusy"
          @save="saveSettings"
          @language-change="handleLanguageChange"
          @theme-change="handleThemeChange"
          @pick-workspace="pickWorkspaceDirectory"
          @restore-model-registry="confirmRestoreModelRegistry"
        />
      </DqSurfaceCard>
    </div>
  </div>
</template>

<script setup lang="ts">
// @ts-nocheck
import { ref, reactive, watch, onMounted, onUnmounted, inject, type Ref } from 'vue';
import { useI18n } from 'vue-i18n';
import { toast, confirm } from '@/utils/feedback';
import { api } from '@/utils/api';
import { $tt, applyTheme, migrateThemeId, VALID_THEME_IDS, type ThemeId } from '@/utils/i18n';
import { DQ_STORAGE, getItem, setItem } from '@/utils/storage';
import { useThemeStore } from '@/stores/theme';
import { useRegistryStore } from '@/stores/registry';
import type { SystemInfo } from '@/types';
import SystemSettingsForm from '@/components/settings/SystemSettingsForm.vue';
import SystemSettingsSidebar from '@/components/settings/SystemSettingsSidebar.vue';
import RuntimeSetupView from '@/views/RuntimeSetupView.vue';
import {
  Monitor,
  FolderChecked,
  Document,
  Tools,
  Setting,
  Box,
  Picture,
  Cpu,
} from '@danqing/dq-shell';

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

type SectionId =
  | 'general'
  | 'performance'
  | 'studio'
  | 'quicksetup'
  | 'workspace'
  | 'integrations'
  | 'runtime'
  | 'maintenance'
  | 'systeminfo';

interface NavItem {
  id: SectionId;
  labelKey: string;
  icon: any;
}

interface MonitorData {
  cpu_percent: number;
  memory: {
    total_gb: number;
    used_gb: number;
    percent: number;
  };
  gpu: null | {
    model?: string;
    memory_gb?: number;
    note?: string;
  };
}

/* ------------------------------------------------------------------ */
/*  Nav items                                                          */
/* ------------------------------------------------------------------ */

const navItems: NavItem[] = [
  { id: 'general', labelKey: 'settings.general', icon: Monitor },
  { id: 'performance', labelKey: 'settings.performance', icon: Monitor },
  { id: 'studio', labelKey: 'settings.studio', icon: Picture },
  { id: 'quicksetup', labelKey: 'settings.quickSetupTitle', icon: Setting },
  { id: 'workspace', labelKey: 'settings.workspace', icon: FolderChecked },
  { id: 'integrations', labelKey: 'settings.integrations', icon: Document },
  { id: 'runtime', labelKey: 'settings.runtime', icon: Cpu },
  { id: 'maintenance', labelKey: 'settings.maintenance', icon: Tools },
  { id: 'systeminfo', labelKey: 'settings.systeminfo', icon: Box },
];

/* ------------------------------------------------------------------ */
/*  Injected / External                                                */
/* ------------------------------------------------------------------ */

const systemInfo = inject<Ref<SystemInfo>>('systemInfo');
const registryStore = useRegistryStore();
const { locale } = useI18n();

/* ------------------------------------------------------------------ */
/*  State                                                              */
/* ------------------------------------------------------------------ */

const activeSection = ref<SectionId>(
  (getItem(DQ_STORAGE.SETTINGS_TAB) as SectionId | null) || 'quicksetup'
);

const settings = reactive<Record<string, unknown>>({
  language: 'zh',
  theme: 'mac',
  auto_save_prompts: true,
  output_format: 'png',
  default_model_llm: 'qwen3.5-4b',
  default_model_vlm: 'qwen3-vl-4b-instruct',
  default_model_llm_think: false,
  mlx_memory_limit: 120,
  model_cache_ttl_minutes: 30,
  queue_image_first: false,
  civitai_token: '',
  huggingface_token: '',
  nsfw_enabled: false,
  custom_workspace_dir: '',
});

const workspacePaths = ref<Record<string, string> | null>(null);
const initialWorkspaceDir = ref('');
const workspaceEffectiveRoot = ref('');

const monitorData = reactive<MonitorData>({
  cpu_percent: 0,
  memory: { total_gb: 0, used_gb: 0, percent: 0 },
  gpu: null,
});
let monitorInterval: ReturnType<typeof setInterval> | null = null;

const cacheStatus = reactive<{ cache: Record<string, unknown> | null; mlx: Record<string, unknown> }>({
  cache: null,
  mlx: {},
});
const cacheLoading = ref(false);
const cacheError = ref('');
const restoreConfigBusy = ref(false);

/* ------------------------------------------------------------------ */
/*  Persistence                                                        */
/* ------------------------------------------------------------------ */

watch(activeSection, (newVal, oldVal) => {
  setItem(DQ_STORAGE.SETTINGS_TAB, newVal);
  if (newVal === 'performance' || newVal === 'systeminfo') {
    refreshCacheStatus();
  }
  if (newVal === 'systeminfo') {
    startMonitor();
  } else if (oldVal === 'systeminfo') {
    stopMonitor();
  }
});

watch(
  () => settings.language as string,
  (newVal) => {
    if (locale.value !== newVal) {
      locale.value = newVal;
      setItem(DQ_STORAGE.LANG, newVal);
      document.documentElement.lang = newVal;
    }
  }
);

/* ------------------------------------------------------------------ */
/*  Settings load / save                                               */
/* ------------------------------------------------------------------ */

const loadSettings = async () => {
  try {
    const data = await api.settings.getSettings();
    Object.assign(settings, data);
    delete settings.custom_models_dir;
    delete settings.custom_loras_dir;
    delete settings.custom_outputs_dir;
    initialWorkspaceDir.value = String(data.custom_workspace_dir || '').trim();
    if (data.language) {
      locale.value = data.language;
      settings.language = data.language;
      setItem(DQ_STORAGE.LANG, data.language);
      document.documentElement.lang = data.language;
    }
    const migrated = migrateThemeId(data.theme as string | undefined);
    if (migrated && VALID_THEME_IDS.includes(migrated)) {
      settings.theme = migrated;
      applyTheme(migrated);
      setItem(DQ_STORAGE.THEME, migrated);
    } else {
      const savedTheme = migrateThemeId(getItem(DQ_STORAGE.THEME));
      if (savedTheme) {
        settings.theme = savedTheme;
        applyTheme(savedTheme);
      } else {
        settings.theme = 'mac';
        applyTheme('mac');
      }
    }
  } catch (e) {
    console.error('Failed to load settings:', e);
  }
};

const loadWorkspacePaths = async () => {
  try {
    const [paths, status] = await Promise.all([
      api.settings.getWorkspacePaths(),
      api.settings.getWorkspaceStatus(),
    ]);
    workspacePaths.value = paths;
    workspaceEffectiveRoot.value = status.effective_root || '';
  } catch (e) {
    workspacePaths.value = null;
  }
};

const pickWorkspaceDirectory = async () => {
  try {
    const { path } = await api.settings.pickWorkspaceDirectory();
    if (path) {
      settings.custom_workspace_dir = path;
    }
  } catch (e) {
    toast.error((e as Error).message || String(e));
  }
};

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

async function confirmWorkspaceRelocation(fromPath: string, toPath: string): Promise<boolean> {
  try {
    await confirm(
      $tt('settings.workspaceChangeConfirm', { from: fromPath, to: toPath }),
      $tt('settings.workspaceChangeTitle'),
      {
        type: 'warning',
        confirmButtonText: $tt('settings.workspaceChangeContinue'),
        cancelButtonText: $tt('settings.deleteCancel'),
      },
    );
    await confirm(
      $tt('settings.workspaceChangeConfirmFinal'),
      $tt('settings.workspaceChangeTitle'),
      {
        type: 'warning',
        confirmButtonText: $tt('settings.workspaceChangeContinue'),
        cancelButtonText: $tt('settings.deleteCancel'),
      },
    );
    return true;
  } catch {
    return false;
  }
}

const saveSettings = async () => {
  const newWs = String(settings.custom_workspace_dir || '').trim();
  const oldWs = initialWorkspaceDir.value;

  if (newWs !== oldWs) {
    if (!newWs) {
      toast.warning($tt('settings.workspaceRequired'));
      settings.custom_workspace_dir = oldWs;
      return;
    }
    const fromLabel = oldWs || workspaceEffectiveRoot.value || $tt('settings.workspaceLayoutTitle');
    const ok = await confirmWorkspaceRelocation(fromLabel, newWs);
    if (!ok) {
      settings.custom_workspace_dir = oldWs;
      return;
    }
    try {
      const wsRes = await api.settings.applyWorkspace(newWs);
      initialWorkspaceDir.value = String(wsRes.workspace || newWs).trim();
      settings.custom_workspace_dir = initialWorkspaceDir.value;
      if (wsRes.restart_required) {
        toast.warning($tt('settings.customWorkspaceRestartHint'));
      }
      await loadWorkspacePaths();
    } catch (e) {
      settings.custom_workspace_dir = oldWs;
      toast.error(extractApiError(e) || $tt('settings.workspaceApplyFailed'));
      return;
    }
  }

  try {
    await api.settings.updateSettings({
      language: settings.language,
      theme: settings.theme,
      output_format: settings.output_format,
      default_model_llm: settings.default_model_llm,
      default_model_vlm: settings.default_model_vlm,
      default_model_llm_think: settings.default_model_llm_think,
      mlx_memory_limit: settings.mlx_memory_limit,
      model_cache_ttl_minutes: settings.model_cache_ttl_minutes,
      queue_image_first: settings.queue_image_first,
      auto_save_prompts: settings.auto_save_prompts,
      civitai_token: settings.civitai_token || '',
      huggingface_token: settings.huggingface_token || '',
      nsfw_enabled: settings.nsfw_enabled,
    });
    toast.success($tt('settings.saved'));
  } catch (e) {
    toast.error(extractApiError(e) || $tt('settings.saveFailed'));
  }
};

const handleLanguageChange = (lang: string) => {
  settings.language = lang;
};

const handleThemeChange = (theme: string) => {
  const id = migrateThemeId(theme) ?? 'mac';
  settings.theme = id;
  useThemeStore().setTheme(id);
};

/* ------------------------------------------------------------------ */
/*  Maintenance                                                        */
/* ------------------------------------------------------------------ */

async function runRestoreConfigDefaults(files: string[]) {
  if (restoreConfigBusy.value) return;
  restoreConfigBusy.value = true;
  try {
    const res = await api.settings.restoreConfigDefaults(files);
    const restored = res.restored || [];
    if (!restored.length) {
      toast.warning($tt('settings.restoreConfigNothing'));
      return;
    }
    if (restored.includes('models_registry.json')) {
      await registryStore.load(true);
    }
    toast.success($tt('settings.restoreConfigSuccess'));
    if (res.restart_required) {
      toast.warning($tt('settings.restoreRegistryRestartHint'));
    }
  } catch (e) {
    toast.error(extractApiError(e) || $tt('settings.restoreConfigFailed'));
  } finally {
    restoreConfigBusy.value = false;
  }
}

async function confirmRestoreModelRegistry() {
  try {
    await confirm(
      $tt('settings.restoreModelRegistryConfirm'),
      $tt('settings.restoreModelRegistryTitle'),
      {
        type: 'warning',
        confirmButtonText: $tt('settings.restoreConfirm'),
        cancelButtonText: $tt('settings.deleteCancel'),
      },
    );
    await runRestoreConfigDefaults(['models_registry.json']);
  } catch {
    /* cancelled */
  }
}

/* ------------------------------------------------------------------ */
/*  Monitor / Cache                                                    */
/* ------------------------------------------------------------------ */

const loadMonitorData = async () => {
  try {
    const data = await api.settings.getSystemMonitor();
    Object.assign(monitorData, data);
  } catch (e) {
    console.error('Failed to load monitor data:', e);
  }
};

const startMonitor = () => {
  loadMonitorData();
  monitorInterval = setInterval(loadMonitorData, 3000);
};

const stopMonitor = () => {
  if (monitorInterval) {
    clearInterval(monitorInterval);
    monitorInterval = null;
  }
};

const refreshCacheStatus = async () => {
  cacheLoading.value = true;
  cacheError.value = '';
  try {
    const data = await api.system.getCacheStatus();
    cacheStatus.cache = (data as { cache?: Record<string, unknown> }).cache || null;
    cacheStatus.mlx = (data as { mlx?: Record<string, unknown> }).mlx || {};
  } catch (e: unknown) {
    cacheError.value = e instanceof Error ? e.message : String(e);
  } finally {
    cacheLoading.value = false;
  }
};

/* ------------------------------------------------------------------ */
/*  Lifecycle                                                          */
/* ------------------------------------------------------------------ */

onMounted(() => {
  loadSettings();
  loadWorkspacePaths();
  refreshCacheStatus();
  if (activeSection.value === 'systeminfo') {
    startMonitor();
  }
});

onUnmounted(() => {
  stopMonitor();
});
</script>

<style scoped>
.settings-page--v2 {
  display: flex;
  gap: 20px;
  width: 100%;
  height: 100%;
  overflow: hidden;
}

/* ── Left Sidebar Wrapper ── */
.settings-sidebar-wrapper {
  width: 220px;
  flex-shrink: 0;
}

.settings-sidebar-card {
  height: 100%;
}

.settings-sidebar-card.dq-surface-card {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.settings-sidebar-card.dq-surface-card > .dq-surface-card__body {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.settings-sidebar-desc {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
  margin: 6px 0 10px 0;
  line-height: 1.4;
}

/* ── Content Area ── */
.settings-content-area {
  flex: 1;
  min-width: 0;
  overflow-y: auto;
  padding: 4px 8px 0 0;
}

.settings-pref-pane-form--mt {
  margin-top: 16px;
}
</style>
