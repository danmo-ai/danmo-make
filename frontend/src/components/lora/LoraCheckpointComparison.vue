<template>
  <div class="lora-ckpt-compare">
    <div class="lora-ckpt-compare__head">
      <h4 class="lora-ckpt-compare__title">{{ $t('loraTrain.candidateTitle') }}</h4>
      <p class="lora-ckpt-compare__hint">{{ $t('loraTrain.candidateHint') }}</p>
    </div>

    <DqAlert
      v-if="status === 'pending_evaluation' || status === 'evaluating'"
      type="info"
      :closable="false"
      :title="$t('loraTrain.candidateEvaluating')"
    >
      <p>{{ $t('loraTrain.candidateEvaluatingBody') }}</p>
      <DqButton size="sm" type="secondary" :loading="retrying" @click="retryEval">
        {{ $t('loraTrain.candidateRetry') }}
      </DqButton>
    </DqAlert>

    <DqAlert
      v-else-if="status === 'failed'"
      type="danger"
      :closable="false"
      :title="$t('loraTrain.candidateFailed')"
    >
      <p>{{ evaluation?.error || $t('loraTrain.candidateFailedBody') }}</p>
      <DqButton size="sm" type="secondary" :loading="retrying" @click="retryEval">
        {{ $t('loraTrain.candidateRetry') }}
      </DqButton>
    </DqAlert>

    <div
      v-else-if="status === 'ready'"
      class="lora-ckpt-compare__ready-actions"
    >
      <DqButton size="sm" type="secondary" :loading="retrying" @click="retryEval">
        {{ $t('loraTrain.candidateRetry') }}
      </DqButton>
    </div>

    <DqAlert
      v-else-if="status === 'registered'"
      type="success"
      :closable="false"
      :title="$t('loraTrain.candidateRegistered', { checkpoint: evaluation?.selected_checkpoint || '' })"
    />

    <div v-if="matrixRows.length" class="lora-ckpt-compare__matrix-wrap">
      <table class="lora-ckpt-compare__matrix">
        <thead>
          <tr>
            <th class="lora-ckpt-compare__corner">{{ $t('loraTrain.candidateCheckpointCol') }}</th>
            <th v-for="prompt in prompts" :key="prompt.id">
              <span class="lora-ckpt-compare__prompt-id">{{ promptLabel(prompt.id) }}</span>
              <span class="lora-ckpt-compare__prompt-text" :title="prompt.prompt">{{ prompt.prompt }}</span>
            </th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="row in matrixRows"
            :key="row.checkpoint"
            class="lora-ckpt-compare__row"
            :class="{ 'is-selected': selected === row.checkpoint }"
          >
            <th>
              <label class="lora-ckpt-compare__pick">
                <input
                  type="radio"
                  name="lora-ckpt-pick"
                  :value="row.checkpoint"
                  :disabled="!canSelect"
                  :checked="selected === row.checkpoint"
                  @change="selected = row.checkpoint"
                />
                <span>
                  <strong>{{ row.label }}</strong>
                  <em v-if="row.kind === 'best'">{{ $t('loraTrain.candidateValLossOnly') }}</em>
                </span>
              </label>
            </th>
            <td v-for="prompt in prompts" :key="`${row.checkpoint}-${prompt.id}`">
              <div class="lora-ckpt-compare__cell">
                <button
                  v-for="seed in seeds"
                  :key="`${row.checkpoint}-${prompt.id}-${seed}`"
                  type="button"
                  class="lora-ckpt-compare__thumb"
                  :disabled="!cellFor(row.checkpoint, prompt.id, seed)?.filename"
                  @click="openPreview(cellFor(row.checkpoint, prompt.id, seed))"
                >
                  <img
                    v-if="cellFor(row.checkpoint, prompt.id, seed)?.status === 'ok'"
                    :src="artifactUrl(cellFor(row.checkpoint, prompt.id, seed)!.filename!)"
                    :alt="`${row.label} / ${prompt.id} / ${seed}`"
                    loading="lazy"
                  />
                  <span v-else class="lora-ckpt-compare__missing">
                    {{ cellFor(row.checkpoint, prompt.id, seed)?.status || '—' }}
                  </span>
                  <span class="lora-ckpt-compare__seed">s{{ seed }}</span>
                </button>
              </div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <div v-if="canSelect" class="lora-ckpt-compare__register">
      <DqInput v-model="registerName" :placeholder="$t('loraTrain.registerName')" size="sm" />
      <DqButton
        type="primary"
        size="sm"
        :disabled="!selected"
        :loading="registering"
        @click="confirmRegister"
      >
        {{ $t('loraTrain.candidateConfirmRegister') }}
      </DqButton>
    </div>

    <div v-if="previewUrl" class="lora-ckpt-compare__lightbox" @click="previewUrl = ''">
      <img :src="previewUrl" alt="" @click.stop />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { api } from '@/utils/api';
import { toast } from '@/utils/feedback';

type EvalPrompt = { id: string; prompt: string };
type EvalCell = {
  checkpoint: string;
  prompt_id: string;
  seed: number;
  filename?: string;
  status?: string;
  error?: string;
};
type EvalCheckpoint = {
  checkpoint: string;
  kind?: string;
  step?: number | null;
  label?: string;
};
type CandidateEvaluation = {
  status?: string;
  error?: string;
  output_name?: string;
  selected_checkpoint?: string | null;
  user_lora_id?: string;
  checkpoints?: EvalCheckpoint[];
  prompts?: EvalPrompt[];
  seeds?: number[];
  cells?: EvalCell[];
  inference?: Record<string, unknown>;
};

const props = defineProps<{
  taskId: string;
  evaluation: CandidateEvaluation | null;
}>();

const emit = defineEmits<{
  (e: 'registered', payload: { user_lora_id: string; checkpoint: string }): void;
  (e: 'retry'): void;
}>();

const { t, te } = useI18n();
const selected = ref('');
const registerName = ref('');
const registering = ref(false);
const retrying = ref(false);
const previewUrl = ref('');

const status = computed(() => String(props.evaluation?.status || ''));
const prompts = computed(() => props.evaluation?.prompts || []);
const seeds = computed(() => (props.evaluation?.seeds?.length ? props.evaluation.seeds : [42, 3407]));
const canSelect = computed(() => status.value === 'ready');

const matrixRows = computed(() => {
  const rows = props.evaluation?.checkpoints || [];
  return rows.map((row) => ({
    checkpoint: row.checkpoint,
    kind: row.kind || '',
    label: row.label || row.checkpoint,
  }));
});

const cellIndex = computed(() => {
  const map = new Map<string, EvalCell>();
  for (const cell of props.evaluation?.cells || []) {
    map.set(`${cell.checkpoint}::${cell.prompt_id}::${cell.seed}`, cell);
  }
  return map;
});

watch(
  () => props.evaluation,
  (ev) => {
    if (!registerName.value && ev?.output_name) registerName.value = String(ev.output_name);
    if (ev?.selected_checkpoint) selected.value = String(ev.selected_checkpoint);
  },
  { immediate: true }
);

function cellFor(checkpoint: string, promptId: string, seed: number): EvalCell | undefined {
  return cellIndex.value.get(`${checkpoint}::${promptId}::${seed}`);
}

function promptLabel(id: string): string {
  const key = `loraTrain.candidatePrompt.${id}`;
  return te(key) ? t(key) : id;
}

function artifactUrl(filename: string): string {
  return api.loras.artifactFileUrl(props.taskId, `candidates/${filename}`);
}

function openPreview(cell: EvalCell | undefined) {
  if (!cell?.filename || cell.status !== 'ok') return;
  previewUrl.value = artifactUrl(cell.filename);
}

async function retryEval() {
  if (retrying.value) return;
  retrying.value = true;
  try {
    await api.loras.submitCheckpointEvaluation(props.taskId, { force: true });
    toast.success(t('loraTrain.candidateRetryQueued'));
    emit('retry');
  } catch (e: unknown) {
    const err = e as { response?: { data?: { detail?: { message?: string } | string } }; message?: string };
    const detail = err?.response?.data?.detail;
    toast.error(
      typeof detail === 'object' && detail?.message
        ? detail.message
        : typeof detail === 'string'
          ? detail
          : err?.message || String(e)
    );
  } finally {
    retrying.value = false;
  }
}

async function confirmRegister() {
  if (!selected.value || registering.value) return;
  registering.value = true;
  try {
    const res = (await api.loras.registerCheckpoint(props.taskId, {
      checkpoint: selected.value,
      name: registerName.value.trim(),
    })) as { user_lora?: { id?: string } };
    const id = String(res.user_lora?.id || '');
    toast.success(t('loraTrain.registerSuccess'));
    emit('registered', { user_lora_id: id, checkpoint: selected.value });
  } catch (e: unknown) {
    const err = e as { response?: { data?: { detail?: { message?: string } | string } }; message?: string };
    const detail = err?.response?.data?.detail;
    toast.error(
      typeof detail === 'object' && detail?.message
        ? detail.message
        : typeof detail === 'string'
          ? detail
          : err?.message || String(e)
    );
  } finally {
    registering.value = false;
  }
}
</script>

<style scoped>
.lora-ckpt-compare {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.lora-ckpt-compare__title {
  margin: 0 0 6px;
  font-size: var(--dq-font-size-body);
  font-weight: 600;
}

.lora-ckpt-compare__hint {
  margin: 0;
  color: var(--dq-label-secondary);
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
}

.lora-ckpt-compare__matrix-wrap {
  overflow: auto;
  border: 0.5px solid var(--dq-border-subtle);
  border-radius: var(--dq-radius-group);
  background: var(--dq-fill-secondary);
}

.lora-ckpt-compare__matrix {
  border-collapse: collapse;
  min-width: 100%;
  font-size: var(--dq-font-size-caption);
}

.lora-ckpt-compare__matrix th,
.lora-ckpt-compare__matrix td {
  border-bottom: 0.5px solid var(--dq-border-subtle);
  padding: 10px;
  vertical-align: top;
  text-align: left;
}

.lora-ckpt-compare__matrix thead th {
  position: sticky;
  top: 0;
  background: color-mix(in srgb, var(--dq-bg-base) 88%, var(--dq-fill-secondary));
  z-index: 1;
  min-width: 160px;
}

.lora-ckpt-compare__corner {
  min-width: 140px;
}

.lora-ckpt-compare__prompt-id {
  display: block;
  font-weight: 600;
  color: var(--dq-label-primary);
}

.lora-ckpt-compare__prompt-text {
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
  color: var(--dq-label-tertiary);
  line-height: 1.35;
}

.lora-ckpt-compare__row.is-selected {
  background: color-mix(in srgb, var(--dq-accent) 8%, transparent);
}

.lora-ckpt-compare__pick {
  display: flex;
  gap: 8px;
  align-items: flex-start;
  cursor: pointer;
}

.lora-ckpt-compare__pick strong {
  display: block;
}

.lora-ckpt-compare__pick em {
  display: block;
  font-style: normal;
  color: var(--dq-label-tertiary);
}

.lora-ckpt-compare__cell {
  display: flex;
  gap: 6px;
}

.lora-ckpt-compare__thumb {
  position: relative;
  width: 96px;
  height: 96px;
  padding: 0;
  border: 0.5px solid var(--dq-border);
  border-radius: var(--dq-radius-group);
  overflow: hidden;
  background: var(--dq-bg-base);
  cursor: zoom-in;
}

.lora-ckpt-compare__thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.lora-ckpt-compare__seed {
  position: absolute;
  left: 4px;
  bottom: 4px;
  padding: 1px 4px;
  border-radius: 4px;
  background: color-mix(in srgb, var(--dq-bg-base) 75%, transparent);
  font-size: var(--dq-font-size-caption);
}

.lora-ckpt-compare__missing {
  display: grid;
  place-items: center;
  width: 100%;
  height: 100%;
  color: var(--dq-label-tertiary);
}

.lora-ckpt-compare__ready-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 8px;
}

.lora-ckpt-compare__register {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}

.lora-ckpt-compare__lightbox {
  position: fixed;
  inset: 0;
  z-index: 80;
  display: grid;
  place-items: center;
  background: color-mix(in srgb, #000 55%, transparent);
  padding: 24px;
}

.lora-ckpt-compare__lightbox img {
  max-width: min(920px, 96vw);
  max-height: 90vh;
  border-radius: var(--dq-radius-group);
}
</style>
