<script setup lang="ts">
import type { SystemInfo } from '@/types';
import { $tt } from '@/utils/i18n';

type CacheModel = {
  key: string;
  idle_minutes: number;
  size_gb: number;
};

type CacheSnapshot = {
  cached_models?: number;
  total_gb?: number;
  limit_gb?: number;
  models?: CacheModel[];
};

type MonitorData = {
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
};

defineProps<{
  systemInfo?: SystemInfo | null;
  cacheStatus: { cache: CacheSnapshot | null };
  cacheLoading: boolean;
  cacheError: string;
  monitorData: MonitorData;
}>();

const emit = defineEmits<{
  refreshCache: [];
}>();

function progressColor(percent: number) {
  if (percent < 50) return 'var(--dq-success)';
  if (percent < 80) return 'var(--dq-warning)';
  return 'var(--dq-danger)';
}
</script>

<template>
  <div class="settings-system-form">
    <section class="settings-group-block">
      <h2 class="settings-section-title">{{ $t('settings.systeminfo') }}</h2>
      <p class="settings-section-desc">{{ $t('settings.systeminfoDesc') }}</p>

      <DqPrefPane
        v-if="systemInfo"
        class="settings-grouped-form settings-pref-pane-form settings-pref-pane-form--system"
      >
        <DqPrefRow :label="$t('settings.platform')">
          <span>{{ systemInfo.platform }} {{ systemInfo.architecture }}</span>
        </DqPrefRow>
        <DqPrefRow :label="$t('settings.memory')">
          <span>{{ systemInfo.memory_gb?.toFixed(1) }} GB</span>
        </DqPrefRow>
        <DqPrefRow :label="$t('settings.pythonVersion')">
          <span>{{ systemInfo.python_version }}</span>
        </DqPrefRow>
        <DqPrefRow v-if="systemInfo.dependencies" :label="$t('settings.dependencies')" stacked>
          <div class="settings-stacked-control">
            <div class="settings-dep-tags">
              <DqTag
                v-for="(version, name) in systemInfo.dependencies"
                :key="name"
                size="sm"
                type="info"
                effect="plain"
              >
                {{ name }} {{ version }}
              </DqTag>
            </div>
          </div>
        </DqPrefRow>
      </DqPrefPane>

      <h3 class="settings-group-block-title settings-system-subsection-title">
        {{ $t('settings.modelCacheTitle') }}
      </h3>
      <div class="settings-header-actions">
        <p v-if="cacheError" class="settings-section-desc settings-section-desc--compact">
          {{ cacheError }}
        </p>
        <p v-else-if="cacheStatus.cache" class="settings-section-desc settings-section-desc--compact">
          {{
            $tt('settings.modelCacheTotal', {
              count: cacheStatus.cache.cached_models ?? 0,
              total: cacheStatus.cache.total_gb ?? 0,
              limit: cacheStatus.cache.limit_gb ?? 0,
            })
          }}
        </p>
        <DqButton
          size="sm"
          class="settings-workspace-pick-btn"
          :loading="cacheLoading"
          @click="emit('refreshCache')"
        >
          {{ $t('gallery.refresh') }}
        </DqButton>
      </div>
      <DqPrefPane class="settings-grouped-form settings-pref-pane-form settings-pref-pane-form--system">
        <DqPrefRow
          v-for="m in cacheStatus.cache?.models || []"
          :key="m.key"
          :label="m.key"
        >
          <span>
            {{ m.size_gb }} GB · {{ $tt('settings.modelCacheIdle', { minutes: m.idle_minutes }) }}
          </span>
        </DqPrefRow>

        <DqPrefRow v-if="!cacheError && !cacheStatus.cache?.models?.length" no-label>
          <span class="settings-form-hint">{{ $t('settings.modelCacheEmpty') }}</span>
        </DqPrefRow>
      </DqPrefPane>

      <h3 class="settings-group-block-title settings-system-subsection-title">
        {{ $t('settings.resourceMonitor') }}
      </h3>
      <p class="settings-section-desc settings-section-desc--compact">{{ $t('settings.realtime') }}</p>
      <DqPrefPane class="settings-grouped-form settings-pref-pane-form settings-pref-pane-form--system">
        <DqPrefRow :label="$t('settings.cpu')">
          <div class="param-control-row settings-pref-slider-row">
            <div class="param-slider">
              <DqProgress
                :percentage="monitorData.cpu_percent"
                :color="progressColor(monitorData.cpu_percent)"
                :show-text="false"
                :stroke-width="6"
              />
            </div>
            <span class="settings-slider-suffix">{{ monitorData.cpu_percent }}%</span>
          </div>
        </DqPrefRow>

        <DqPrefRow :label="$t('settings.memoryLabel')">
          <div class="param-control-row settings-pref-slider-row">
            <div class="param-slider">
              <DqProgress
                :percentage="monitorData.memory.percent"
                :color="progressColor(monitorData.memory.percent)"
                :show-text="false"
                :stroke-width="6"
              />
            </div>
            <span class="settings-slider-suffix">
              {{ monitorData.memory.used_gb }} / {{ monitorData.memory.total_gb }} GB
            </span>
          </div>
        </DqPrefRow>

        <DqPrefRow v-if="monitorData.gpu" :label="$t('settings.gpu')" stacked>
          <div class="settings-stacked-control">
            <span v-if="monitorData.gpu.model">{{ monitorData.gpu.model }}</span>
            <p class="settings-form-hint settings-form-hint--below-control">
              <template v-if="monitorData.gpu.memory_gb">
                {{ monitorData.gpu.memory_gb }} {{ $t('settings.unifiedMemory') }}
              </template>
              <template v-if="monitorData.gpu.memory_gb && monitorData.gpu.note"> · </template>
              <template v-if="monitorData.gpu.note">{{ monitorData.gpu.note }}</template>
            </p>
          </div>
        </DqPrefRow>
      </DqPrefPane>
      <p class="settings-group-footnote">{{ $t('settings.refreshInterval') }}</p>
    </section>
  </div>
</template>
