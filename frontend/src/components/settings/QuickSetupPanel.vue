<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { toast } from '@/utils/feedback';
import { api } from '@/utils/api';
import { $tt, $mn, $mvn } from '@/utils/i18n';
import { useModelInstall } from '@/composables/useModelInstall';

type SetupSlotKey = 'image' | 'video' | 'audio' | 'llm' | 'vlm';

interface SlotRecommendation {
  slot: SetupSlotKey;
  status: string;
  model_id?: string | null;
  version_key?: string | null;
  estimated_gb?: number | null;
  warning?: string | null;
  reason?: string | null;
  installed?: boolean;
  name?: Record<string, string>;
  version_name?: Record<string, string>;
  size_human?: string;
}

interface RecommendationsPayload {
  reference_memory_gb: number;
  memory_tier: string;
  available_backends: string[];
  slots: SlotRecommendation[];
}

const emit = defineEmits<{
  patchSettings: [patch: Record<string, unknown>];
}>();

const loading = ref(false);
const applyingDefaults = ref(false);
const recommendations = ref<RecommendationsPayload | null>(null);

const { downloading, progressByKey, installModel, uiKey } = useModelInstall({
  onCompleted: () => {
    void loadRecommendations();
  },
});

function extractApiError(e: unknown): string {
  if (typeof e === 'object' && e !== null && 'response' in e) {
    const err = e as { response?: { data?: { detail?: string } } };
    if (err.response?.data?.detail) return err.response.data.detail;
  }
  if (e instanceof Error) return e.message;
  return String(e);
}

function slotLabel(slot: SetupSlotKey): string {
  const map: Record<SetupSlotKey, string> = {
    image: 'settings.quickSetupSlotImage',
    video: 'settings.quickSetupSlotVideo',
    audio: 'settings.quickSetupSlotAudio',
    llm: 'settings.quickSetupSlotLlm',
    vlm: 'settings.quickSetupSlotVlm',
  };
  return $tt(map[slot]);
}

function reasonText(reason: string): string {
  if (reason === 'mlx_required') return $tt('settings.quickSetupReasonMlxRequired');
  if (reason === 'no_compatible_models') return $tt('settings.quickSetupReasonNoModels');
  return reason;
}

function displayName(slot: SlotRecommendation): string {
  if (slot.name && slot.model_id) {
    return $mn({ name: slot.name }, slot.model_id);
  }
  return slot.model_id || '—';
}

function displayVersionName(slot: SlotRecommendation): string {
  if (slot.version_name && slot.model_id) {
    return $mvn(slot.model_id, { name: slot.name || {} }, { name: slot.version_name });
  }
  return slot.version_key || '';
}

function canDownload(slot: SlotRecommendation): boolean {
  return (
    slot.status !== 'unavailable' &&
    Boolean(slot.model_id) &&
    Boolean(slot.version_key) &&
    !slot.installed
  );
}

async function loadRecommendations() {
  loading.value = true;
  try {
    recommendations.value = (await api.setup.getRecommendations()) as RecommendationsPayload;
  } catch (e) {
    toast.error(extractApiError(e));
  } finally {
    loading.value = false;
  }
}

async function downloadSlot(slot: SlotRecommendation) {
  if (!slot.model_id || !slot.version_key) return;
  const label = `${displayName(slot)} ${displayVersionName(slot)}`.trim();
  await installModel(slot.model_id, slot.version_key, label);
}

function buildDefaultsPatch(): Record<string, string> {
  const patch: Record<string, string> = {};
  for (const slot of recommendations.value?.slots || []) {
    if (!slot.model_id || slot.status === 'unavailable') continue;
    patch[slot.slot] = slot.model_id;
  }
  return patch;
}

async function applyRecommendedDefaults() {
  const raw = buildDefaultsPatch();
  if (!Object.keys(raw).length) return;

  const payload: Record<string, string> = {};
  if (raw.image) {
    payload.default_model_image = raw.image;
    payload.default_model = raw.image;
  }
  if (raw.video) payload.default_model_video = raw.video;
  if (raw.audio) payload.default_model_audio = raw.audio;
  if (raw.llm) payload.default_model_llm = raw.llm;
  if (raw.vlm) payload.default_model_vlm = raw.vlm;

  applyingDefaults.value = true;
  try {
    await api.settings.updateSettings(payload);
    emit('patchSettings', payload);
    toast.success($tt('settings.saved'));
  } catch (e) {
    toast.error(extractApiError(e) || $tt('settings.saveFailed'));
  } finally {
    applyingDefaults.value = false;
  }
}

onMounted(() => {
  void loadRecommendations();
});

defineExpose({
  applyingDefaults,
  applyRecommendedDefaults,
});
</script>

<template>
  <section class="settings-group-block">
    <h2 class="settings-section-title">{{ $t('settings.quickSetupTitle') }}</h2>
    <p class="settings-section-desc">{{ $t('settings.quickSetupDesc') }}</p>
    <p class="settings-section-desc settings-section-desc--muted">
      {{ $t('settings.quickSetupWorkspaceOptional') }}
    </p>

    <div v-if="loading" class="quick-setup-panel__loading">{{ $t('common.loading') }}</div>

    <template v-else-if="recommendations">
      <DqPrefPane class="settings-grouped-form settings-pref-pane-form settings-pref-pane-form--system">
        <DqPrefRow :label="$t('settings.quickSetupSystemSummary')" stacked>
          <div class="quick-setup-panel__summary settings-form-hint settings-form-hint--block">
            <div>
              {{
                $tt('settings.quickSetupMemoryRef', {
                  gb: recommendations.reference_memory_gb,
                  tier: recommendations.memory_tier,
                })
              }}
            </div>
            <div>
              {{
                $tt('settings.quickSetupBackends', {
                  backends: (recommendations.available_backends || []).join(', ') || '—',
                })
              }}
            </div>
          </div>
        </DqPrefRow>

        <DqPrefRow
          v-for="slot in recommendations.slots"
          :key="slot.slot"
          :label="slotLabel(slot.slot)"
          stacked
        >
          <div class="quick-setup-panel__row">
            <div class="quick-setup-panel__meta">
              <span class="quick-setup-panel__model">{{ displayName(slot) }}</span>
              <span v-if="slot.version_key" class="quick-setup-panel__muted">
                · {{ displayVersionName(slot) }}
              </span>
              <span v-if="slot.size_human" class="quick-setup-panel__muted">
                ({{ slot.size_human }})
              </span>
              <DqTag v-if="slot.installed" type="success" size="sm" effect="plain">
                {{ $t('settings.quickSetupInstalled') }}
              </DqTag>
              <DqTag
                v-else-if="slot.status === 'unavailable'"
                type="info"
                size="sm"
                effect="plain"
              >
                {{ $t('settings.quickSetupUnavailable') }}
              </DqTag>
              <DqTag
                v-else-if="slot.warning === 'insufficient_memory'"
                type="warning"
                size="sm"
                effect="plain"
              >
                {{ $t('settings.quickSetupWarningMemory') }}
              </DqTag>
            </div>
            <p v-if="slot.reason && !slot.model_id" class="quick-setup-panel__reason settings-form-hint">
              {{ reasonText(slot.reason) }}
            </p>
            <div v-if="canDownload(slot)" class="quick-setup-panel__actions">
              <DqButton
                size="sm"
                class="settings-workspace-pick-btn"
                :loading="downloading[uiKey(slot.model_id!, slot.version_key!)]"
                @click="downloadSlot(slot)"
              >
                {{ $t('download.downloadVersion') }}
              </DqButton>
              <DqProgress
                v-if="progressByKey[uiKey(slot.model_id!, slot.version_key!)] != null"
                class="quick-setup-panel__progress"
                :percentage="progressByKey[uiKey(slot.model_id!, slot.version_key!)]"
                :stroke-width="4"
              />
            </div>
          </div>
        </DqPrefRow>
      </DqPrefPane>
    </template>
  </section>
</template>

<style scoped>
.quick-setup-panel__loading {
  padding: 16px 0;
  color: var(--dq-label-tertiary);
}
.quick-setup-panel__row {
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: 100%;
}
.quick-setup-panel__meta {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
}
.quick-setup-panel__model {
  font-weight: 500;
}
.quick-setup-panel__muted {
  color: var(--dq-label-tertiary);
}
.quick-setup-panel__reason {
  margin: 0;
}
.quick-setup-panel__actions {
  display: flex;
  flex-direction: column;
  gap: 6px;
  max-width: 280px;
}
.quick-setup-panel__progress {
  width: 100%;
}
.settings-section-desc--muted {
  margin-top: -4px;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
}
</style>
