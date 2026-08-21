<script setup lang="ts">
/**
 * In-app MLX (mlx[cuda]) runtime setup / repair surface (Tauri thin builds).
 * First-run also uses desktop/loader/runtime-setup.html before the API is up.
 */
import { onMounted, onUnmounted, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import { confirm } from '@/utils/feedback';
import { isTauriRuntime } from '@/utils/desktop';

const { t } = useI18n();

const available = ref(false);
const ready = ref(false);
const mirror = ref('official');
const mirrors = ref<string[]>(['official', 'tuna', 'aliyun']);
const statusMessage = ref('');
const logText = ref('');
const errorText = ref('');
const busy = ref(false);
const progressPct = ref(0);
const detail = ref<Record<string, unknown> | null>(null);

let unlisten: (() => void) | null = null;

async function invoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  const { invoke: inv } = await import('@tauri-apps/api/core');
  return inv<T>(cmd, args);
}

async function refresh() {
  if (!isTauriRuntime()) {
    available.value = false;
    statusMessage.value = t('runtimeSetup.notDesktop');
    return;
  }
  try {
    const st = await invoke<{
      thin: boolean;
      ready: boolean;
      mirror: string;
      mirrors: string[];
      message: string;
      detail: Record<string, unknown> | null;
      log_path?: string;
    }>('runtime_status');
    available.value = Boolean(st.thin);
    ready.value = Boolean(st.ready);
    if (st.mirror) mirror.value = st.mirror;
    if (st.mirrors?.length) mirrors.value = st.mirrors;
    statusMessage.value = st.message || (st.ready ? t('runtimeSetup.ready') : t('runtimeSetup.notReady'));
    detail.value = st.detail;
  } catch (e) {
    available.value = false;
    errorText.value = String(e);
  }
}

async function start(mode: 'bootstrap' | 'repair' | 'reinstall') {
  try {
    if (mode === 'reinstall') {
      await confirm(t('runtimeSetup.reinstallHint'), t('runtimeSetup.reinstall'), { type: 'warning' });
    } else if (mode === 'repair') {
      await confirm(t('runtimeSetup.desc'), t('runtimeSetup.repair'));
    }
  } catch {
    return;
  }
  errorText.value = '';
  logText.value = '';
  progressPct.value = 5;
  busy.value = true;
  try {
    await invoke('runtime_set_mirror', { mirror: mirror.value });
    await invoke('runtime_install_start', { mode, mirror: mirror.value });
  } catch (e) {
    busy.value = false;
    errorText.value = String(e);
  }
}

async function cancel() {
  try {
    await invoke('runtime_install_cancel');
  } catch {
    /* ignore */
  }
}

onMounted(async () => {
  await refresh();
  if (!isTauriRuntime()) return;
  try {
    const { listen } = await import('@tauri-apps/api/event');
    unlisten = await listen<{ phase?: string; message?: string; ok?: boolean }>('runtime-setup://progress', (ev) => {
      const p = ev.payload || {};
      if (p.message) {
        logText.value += `${p.message}\n`;
        progressPct.value = Math.min(95, progressPct.value + 2);
      }
      if (p.phase === 'done' || p.ok === true) {
        busy.value = false;
        progressPct.value = 100;
        void (async () => {
          try {
            await invoke('runtime_start_api');
            await refresh();
          } catch (e) {
            errorText.value = String(e);
          }
        })();
      } else if (p.phase === 'error' || p.ok === false) {
        busy.value = false;
        errorText.value = p.message || t('runtimeSetup.failed');
      }
    });
  } catch (e) {
    errorText.value = String(e);
  }
});

onUnmounted(() => {
  unlisten?.();
});

defineExpose({ refresh, start, available, ready });
</script>

<template>
  <section v-if="available" class="runtime-setup">
    <h2 class="settings-section-title">{{ $t('runtimeSetup.title') }}</h2>
    <p class="settings-section-desc">{{ $t('runtimeSetup.desc') }}</p>

    <DqPrefPane class="settings-grouped-form settings-pref-pane-form">
      <DqPrefRow :label="$t('runtimeSetup.status')">
        <span>{{ statusMessage }}</span>
      </DqPrefRow>
      <DqPrefRow :label="$t('runtimeSetup.mirror')">
        <DqSelect v-model="mirror" class="settings-mac-value-control" :disabled="busy">
          <DqOption
            v-for="m in mirrors"
            :key="m"
            :value="m"
            :label="$t(`runtimeSetup.mirrors.${m}`)"
          />
        </DqSelect>
      </DqPrefRow>
      <DqPrefRow :label="$t('runtimeSetup.actions')" stacked>
        <div class="runtime-setup__actions">
          <DqButton type="primary" :disabled="busy" @click="start(ready ? 'repair' : 'bootstrap')">
            {{ ready ? $t('runtimeSetup.repair') : $t('runtimeSetup.install') }}
          </DqButton>
          <DqButton type="danger" plain :disabled="busy" @click="start('reinstall')">
            {{ $t('runtimeSetup.reinstall') }}
          </DqButton>
          <DqButton :disabled="!busy" @click="cancel">{{ $t('runtimeSetup.cancel') }}</DqButton>
        </div>
        <p class="settings-form-hint">{{ $t('runtimeSetup.reinstallHint') }}</p>
      </DqPrefRow>
    </DqPrefPane>

    <div v-if="busy || logText" class="runtime-setup__progress">
      <div class="runtime-setup__bar"><i :style="{ width: progressPct + '%' }" /></div>
      <pre v-if="logText" class="runtime-setup__log">{{ logText }}</pre>
    </div>
    <p v-if="errorText" class="runtime-setup__err">{{ errorText }}</p>
  </section>
  <section v-else class="runtime-setup runtime-setup--na">
    <h2 class="settings-section-title">{{ $t('runtimeSetup.title') }}</h2>
    <p class="settings-section-desc">{{ statusMessage || $t('runtimeSetup.notApplicable') }}</p>
  </section>
</template>

<style scoped>
.runtime-setup__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}
.runtime-setup__progress {
  margin-top: 1rem;
}
.runtime-setup__bar {
  height: 8px;
  border-radius: 999px;
  background: var(--dq-fill-secondary, #222);
  overflow: hidden;
}
.runtime-setup__bar > i {
  display: block;
  height: 100%;
  background: var(--dq-accent, #5b8def);
}
.runtime-setup__log {
  margin-top: 0.75rem;
  max-height: 220px;
  overflow: auto;
  font-size: 0.75rem;
  padding: 0.75rem;
  border-radius: 8px;
  background: var(--dq-fill-tertiary, #111);
  white-space: pre-wrap;
}
.runtime-setup__err {
  color: var(--dq-danger, #e57373);
  margin-top: 0.75rem;
  white-space: pre-wrap;
}
</style>
