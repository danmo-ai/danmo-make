<script setup lang="ts">
/**
 * First-run onboarding wizard.
 *
 * Four steps: workspace (optional) → recommendations → install → done.
 * Everything it shows comes from `GET /api/setup/recommendations`; the install
 * step reuses the shared `useModelInstall` SSE path so progress and failures are
 * the same ones the Models page reports.
 */
import { computed, ref, watch } from 'vue';
import { toast } from '@/utils/feedback';
import { api } from '@/utils/api';
import { $tt, $mn, $vn } from '@/utils/i18n';
import { useModelInstall } from '@/composables/useModelInstall';

type StepKey = 'workspace' | 'recommend' | 'install' | 'done';

interface SlotRecommendation {
  slot: string;
  status: 'recommended' | 'warning' | 'unavailable';
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
  memory_detected?: boolean;
  available_backends: string[];
  primary_backend?: string;
  slots: SlotRecommendation[];
}

const props = defineProps<{
  open: boolean;
}>();

const emit = defineEmits<{
  'update:open': [value: boolean];
  completed: [];
}>();

const STEPS: StepKey[] = ['workspace', 'recommend', 'install', 'done'];
const STEP_LABEL: Record<StepKey, string> = {
  workspace: 'settings.quickSetupStepWorkspace',
  recommend: 'settings.quickSetupStepRecommend',
  install: 'settings.quickSetupStepInstall',
  done: 'settings.quickSetupStepDone',
};

const step = ref<StepKey>('workspace');
const stepIndex = computed(() => STEPS.indexOf(step.value));

const loading = ref(false);
const loadError = ref('');
const recommendations = ref<RecommendationsPayload | null>(null);

const workspaceRoot = ref('');
const pickedPath = ref('');
const applyingWorkspace = ref(false);
const workspaceRestartNeeded = ref(false);

const selected = ref<Record<string, boolean>>({});
const installing = ref(false);
const finished = ref<Record<string, boolean>>({});
const applyingDefaults = ref(false);

const { downloading, progressByKey, installModel, uiKey } = useModelInstall();

function extractApiError(e: unknown): string {
  if (typeof e === 'object' && e !== null && 'response' in e) {
    const err = e as { response?: { data?: { detail?: string } } };
    if (err.response?.data?.detail) return err.response.data.detail;
  }
  if (e instanceof Error) return e.message;
  return String(e);
}

function slotLabel(slot: string): string {
  const map: Record<string, string> = {
    image: 'settings.quickSetupSlotImage',
    video: 'settings.quickSetupSlotVideo',
    audio: 'settings.quickSetupSlotAudio',
    llm: 'settings.quickSetupSlotLlm',
    vlm: 'settings.quickSetupSlotVlm',
  };
  return $tt(map[slot] || slot);
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
  if (slot.version_name) {
    return $vn({ name: slot.version_name }, slot.version_key || '');
  }
  return slot.version_key || '';
}

function isInstalled(slot: SlotRecommendation): boolean {
  return Boolean(slot.installed) || Boolean(finished.value[slot.slot]);
}

function selectable(slot: SlotRecommendation): boolean {
  return (
    slot.status !== 'unavailable' &&
    Boolean(slot.model_id) &&
    Boolean(slot.version_key) &&
    !isInstalled(slot)
  );
}

const selectableSlots = computed(
  () => (recommendations.value?.slots || []).filter(selectable),
);

const selectedSlots = computed(() =>
  selectableSlots.value.filter((slot) => selected.value[slot.slot]),
);

const installTotal = computed(() => selectedSlots.value.length);

const installDone = computed(
  () => selectedSlots.value.filter((slot) => finished.value[slot.slot]).length,
);

const allDone = computed(
  () => installTotal.value > 0 && installDone.value >= installTotal.value,
);

/** Reset local selection whenever a new payload arrives. */
function primeSelection(payload: RecommendationsPayload) {
  const next: Record<string, boolean> = {};
  for (const slot of payload.slots) {
    if (slot.status !== 'unavailable' && slot.model_id && slot.version_key && !slot.installed) {
      next[slot.slot] = true;
    }
  }
  selected.value = next;
}

async function loadRecommendations() {
  loading.value = true;
  loadError.value = '';
  try {
    const [payload, status] = await Promise.all([
      api.setup.getRecommendations() as Promise<RecommendationsPayload>,
      api.settings.getWorkspaceStatus().catch(() => null),
    ]);
    recommendations.value = payload;
    primeSelection(payload);
    workspaceRoot.value = status?.effective_root || '';
  } catch (e) {
    loadError.value = extractApiError(e);
    recommendations.value = null;
  } finally {
    loading.value = false;
  }
}

async function pickDirectory() {
  try {
    const { path } = await api.settings.pickWorkspaceDirectory();
    if (path) pickedPath.value = path;
  } catch (e) {
    // The server refuses the picker when it cannot open a native dialog; that is
    // an explicit error, so surface it instead of silently keeping the old path.
    toast.error(extractApiError(e));
  }
}

async function applyWorkspace() {
  const path = pickedPath.value.trim();
  if (!path) return;
  applyingWorkspace.value = true;
  try {
    const res = await api.settings.applyWorkspace(path);
    workspaceRoot.value = String(res.workspace || path).trim();
    workspaceRestartNeeded.value = Boolean(res.restart_required);
    pickedPath.value = '';
    if (res.restart_required) {
      toast.warning($tt('settings.customWorkspaceRestartHint'));
      return;
    }
    toast.success($tt('settings.saved'));
  } catch (e) {
    toast.error(extractApiError(e) || $tt('settings.workspaceApplyFailed'));
  } finally {
    applyingWorkspace.value = false;
  }
}

async function startInstall() {
  const targets = selectedSlots.value;
  if (!targets.length) {
    await applyRecommendedDefaults();
    step.value = 'done';
    return;
  }
  step.value = 'install';
  installing.value = true;
  const waits = targets.map(
    (slot) =>
      new Promise<void>((resolve) => {
        void installModel(
          slot.model_id as string,
          slot.version_key as string,
          `${displayName(slot)} ${displayVersionName(slot)}`.trim(),
          () => {
            finished.value = { ...finished.value, [slot.slot]: true };
            resolve();
          },
        );
      }),
  );
  await Promise.all(waits);
  installing.value = false;
  await applyRecommendedDefaults();
  step.value = 'done';
}

async function applyRecommendedDefaults() {
  const slots = recommendations.value?.slots || [];
  const payload: Record<string, string> = {};
  const bySlot: Record<string, string> = {};
  for (const slot of slots) {
    if (!slot.model_id || slot.status === 'unavailable') continue;
    bySlot[slot.slot] = slot.model_id;
  }
  if (bySlot.image) {
    payload.default_model_image = bySlot.image;
    payload.default_model = bySlot.image;
  }
  if (bySlot.video) payload.default_model_video = bySlot.video;
  if (bySlot.audio) payload.default_model_audio = bySlot.audio;
  if (bySlot.llm) payload.default_model_llm = bySlot.llm;
  if (bySlot.vlm) payload.default_model_vlm = bySlot.vlm;
  if (!Object.keys(payload).length) return;
  applyingDefaults.value = true;
  try {
    await api.settings.updateSettings(payload);
  } catch (e) {
    toast.error(extractApiError(e) || $tt('settings.saveFailed'));
  } finally {
    applyingDefaults.value = false;
  }
}

function close() {
  // Any dismissal (Finish / Go to create / close button) marks onboarding as seen,
  // otherwise the wizard would reappear on every launch.
  emit('completed');
  emit('update:open', false);
}

function next() {
  const current = step.value;
  if (current === 'workspace') {
    step.value = 'recommend';
    return;
  }
  if (current === 'recommend') {
    if (!installTotal.value) {
      void applyRecommendedDefaults().then(() => {
        step.value = 'done';
      });
      return;
    }
    void startInstall();
    return;
  }
  close();
}

function back() {
  const current = step.value;
  if (current === 'recommend') {
    step.value = 'workspace';
  } else if (current === 'install') {
    step.value = 'recommend';
  }
}

function skipInstall() {
  void applyRecommendedDefaults().then(() => {
    step.value = 'done';
  });
}

const primaryLabel = computed(() => {
  if (step.value === 'recommend') {
    return installTotal.value
      ? $tt('settings.quickSetupInstallSelected')
      : $tt('settings.quickSetupNext');
  }
  return $tt('settings.quickSetupNext');
});

const primaryBusy = computed(
  () => installing.value || applyingWorkspace.value || applyingDefaults.value,
);

const primaryDisabled = computed(() => {
  if (installing.value) return true;
  if (step.value === 'recommend') return loading.value || Boolean(loadError.value);
  return false;
});

watch(
  () => props.open,
  (open) => {
    if (!open) return;
    step.value = 'workspace';
    installing.value = false;
    finished.value = {};
    pickedPath.value = '';
    void loadRecommendations();
  },
  { immediate: true },
);
</script>

<template>
  <DqDialog
    :open="props.open"
    :title="$tt('settings.quickSetupTitle')"
    width="640px"
    :closable="!installing"
    @update:open="emit('update:open', $event)"
  >
    <div class="onboard">
      <ol class="onboard__steps">
        <li
          v-for="(key, index) in STEPS"
          :key="key"
          class="onboard__step"
          :class="{
            'onboard__step--active': index === stepIndex,
            'onboard__step--done': index < stepIndex,
          }"
        >
          <span class="onboard__step-index">{{ index + 1 }}</span>
          <span class="onboard__step-label">{{ $tt(STEP_LABEL[key]) }}</span>
        </li>
      </ol>

      <!-- Step 1: workspace (optional) -->
      <section v-if="step === 'workspace'" class="onboard__pane">
        <p class="onboard__desc">{{ $tt('settings.quickSetupWorkspaceOptional') }}</p>
        <p class="onboard__path">
          <span class="onboard__path-label">{{ $tt('settings.workspaceLayoutTitle') }}</span>
          <code>{{ workspaceRoot || '—' }}</code>
        </p>
        <div class="onboard__row">
          <DqInput
            v-model="pickedPath"
            :placeholder="$tt('settings.customWorkspacePlaceholder')"
            readonly
          />
          <DqButton size="sm" :loading="applyingWorkspace" @click="pickDirectory">
            {{ $tt('settings.workspaceSetupPick') }}
          </DqButton>
          <DqButton
            size="sm"
            type="primary"
            :loading="applyingWorkspace"
            :disabled="!pickedPath.trim()"
            @click="applyWorkspace"
          >
            {{ $tt('settings.workspaceSetupConfirm') }}
          </DqButton>
        </div>
        <p class="onboard__hint">{{ $tt('settings.workspaceSetupEmptyHint') }}</p>
        <p v-if="workspaceRestartNeeded" class="onboard__hint onboard__hint--warn">
          {{ $tt('settings.customWorkspaceRestartHint') }}
        </p>
      </section>

      <!-- Step 2: recommendations -->
      <section v-else-if="step === 'recommend'" class="onboard__pane">
        <p v-if="loading" class="onboard__desc">{{ $tt('common.loading') }}</p>
        <DqAlert v-else-if="loadError" type="error" :title="loadError" />
        <template v-else-if="recommendations">
          <p class="onboard__desc">{{ $tt('settings.quickSetupDesc') }}</p>
          <p class="onboard__hint">
            {{
              $tt('settings.quickSetupMemoryRef', {
                gb: recommendations.reference_memory_gb,
                tier: recommendations.memory_tier,
              })
            }}
            ·
            {{
              $tt('settings.quickSetupBackends', {
                backends: (recommendations.available_backends || []).join(', ') || '—',
              })
            }}
          </p>
          <ul class="onboard__slots">
            <li v-for="slot in recommendations.slots" :key="slot.slot" class="onboard__slot">
              <DqCheckbox
                v-if="selectable(slot)"
                :model-value="!!selected[slot.slot]"
                @update:model-value="selected = { ...selected, [slot.slot]: Boolean($event) }"
              />
              <span v-else class="onboard__slot-spacer" />
              <div class="onboard__slot-main">
                <div class="onboard__slot-head">
                  <span class="onboard__slot-name">{{ slotLabel(slot.slot) }}</span>
                  <span class="onboard__slot-model">{{ displayName(slot) }}</span>
                  <span v-if="slot.version_key" class="onboard__slot-version">
                    · {{ displayVersionName(slot) }}
                  </span>
                  <span v-if="slot.size_human" class="onboard__slot-version">
                    ({{ slot.size_human }})
                  </span>
                </div>
                <div class="onboard__slot-tags">
                  <DqTag v-if="isInstalled(slot)" type="success" size="sm" effect="plain">
                    {{ $tt('settings.quickSetupInstalled') }}
                  </DqTag>
                  <DqTag
                    v-else-if="slot.status === 'unavailable'"
                    type="info"
                    size="sm"
                    effect="plain"
                  >
                    {{ $tt('settings.quickSetupUnavailable') }}
                  </DqTag>
                  <DqTag
                    v-else-if="slot.warning === 'insufficient_memory'"
                    type="warning"
                    size="sm"
                    effect="plain"
                  >
                    {{ $tt('settings.quickSetupWarningMemory') }}
                  </DqTag>
                  <span v-if="slot.reason && !slot.model_id" class="onboard__hint">
                    {{ reasonText(slot.reason) }}
                  </span>
                </div>
              </div>
            </li>
          </ul>
        </template>
      </section>

      <!-- Step 3: install -->
      <section v-else-if="step === 'install'" class="onboard__pane">
        <p class="onboard__desc">
          {{
            $tt('settings.quickSetupInstallProgress', {
              done: installDone,
              total: installTotal,
            })
          }}
        </p>
        <ul class="onboard__slots">
          <li v-for="slot in selectedSlots" :key="slot.slot" class="onboard__slot">
            <div class="onboard__slot-main">
              <div class="onboard__slot-head">
                <span class="onboard__slot-name">{{ slotLabel(slot.slot) }}</span>
                <span class="onboard__slot-model">{{ displayName(slot) }}</span>
                <DqTag v-if="finished[slot.slot]" type="success" size="sm" effect="plain">
                  {{ $tt('settings.quickSetupInstalled') }}
                </DqTag>
              </div>
              <DqProgress
                class="onboard__progress"
                :percentage="finished[slot.slot] ? 100 : progressByKey[uiKey(slot.model_id!, slot.version_key!)] || 0"
                :stroke-width="4"
              />
            </div>
          </li>
        </ul>
      </section>

      <!-- Step 4: done -->
      <section v-else class="onboard__pane">
        <h3 class="onboard__done-title">{{ $tt('settings.quickSetupCompleteTitle') }}</h3>
        <p class="onboard__desc">{{ $tt('settings.quickSetupCompleteDesc') }}</p>
        <p v-if="allDone" class="onboard__hint">
          {{
            $tt('settings.quickSetupInstallProgress', {
              done: installDone,
              total: installTotal,
            })
          }}
        </p>
      </section>
    </div>

    <template #footer>
      <DqButton v-if="stepIndex > 0 && step !== 'install' && step !== 'done'" size="sm" @click="back">
        {{ $tt('settings.quickSetupBack') }}
      </DqButton>
      <span class="onboard__spacer" />
      <DqButton
        v-if="step === 'recommend' && installTotal > 0"
        size="sm"
        :disabled="installing"
        @click="skipInstall"
      >
        {{ $tt('settings.quickSetupSkipInstall') }}
      </DqButton>
      <DqButton
        v-if="step !== 'done'"
        type="primary"
        size="sm"
        :loading="primaryBusy"
        :disabled="primaryDisabled"
        @click="next"
      >
        {{ primaryLabel }}
      </DqButton>
      <DqButton v-else type="primary" size="sm" @click="close">
        {{ $tt('settings.quickSetupGoCreate') }}
      </DqButton>
    </template>
  </DqDialog>
</template>

<style scoped>
.onboard {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.onboard__steps {
  display: flex;
  gap: 8px;
  margin: 0;
  padding: 0;
  list-style: none;
}
.onboard__step {
  display: flex;
  align-items: center;
  gap: 6px;
  flex: 1;
  min-width: 0;
  padding: 6px 8px;
  border-radius: var(--dq-radius-control-sm);
  background: var(--dq-fill-tertiary);
  color: var(--dq-label-tertiary);
  font-size: var(--dq-font-size-caption);
}
.onboard__step--active {
  background: var(--dq-fill-secondary);
  color: var(--dq-label-primary);
  font-weight: 600;
}
.onboard__step--done {
  color: var(--dq-label-secondary);
}
.onboard__step-index {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 18px;
  height: 18px;
  flex-shrink: 0;
  border-radius: 50%;
  background: var(--dq-fill-secondary);
  font-size: 11px;
}
.onboard__step--active .onboard__step-index {
  background: var(--dq-accent);
  color: #fff;
}
.onboard__step-label {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.onboard__pane {
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-height: 132px;
}
.onboard__desc {
  margin: 0;
  line-height: 1.5;
  color: var(--dq-label-secondary);
}
.onboard__hint {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
  line-height: 1.5;
}
.onboard__hint--warn {
  color: var(--dq-warning);
}
.onboard__path {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin: 0;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
}
.onboard__path code {
  word-break: break-all;
  font-family: var(--dq-font-mono);
  color: var(--dq-label-secondary);
}
.onboard__row {
  display: flex;
  gap: 8px;
  align-items: center;
}
.onboard__row > :first-child {
  flex: 1;
  min-width: 0;
}
.onboard__slots {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin: 0;
  padding: 0;
  list-style: none;
}
.onboard__slot {
  display: flex;
  gap: 10px;
  align-items: flex-start;
  padding: 8px 10px;
  border-radius: var(--dq-radius-control-sm);
  background: var(--dq-fill-tertiary);
}
.onboard__slot-spacer {
  width: 16px;
  flex-shrink: 0;
}
.onboard__slot-main {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
  flex: 1;
}
.onboard__slot-head {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: baseline;
}
.onboard__slot-name {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
}
.onboard__slot-model {
  font-weight: 500;
}
.onboard__slot-version {
  color: var(--dq-label-tertiary);
  font-size: var(--dq-font-size-caption);
}
.onboard__slot-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
}
.onboard__progress {
  width: 100%;
  max-width: 280px;
}
.onboard__done-title {
  margin: 0;
  font-size: 15px;
}
.onboard__spacer {
  flex: 1;
}
</style>
