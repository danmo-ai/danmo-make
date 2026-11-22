<template>
  <DqDialog
    v-model:open="open"
    :title="$t('audio.lmAssistant')"
    width="min(760px, 94vw)"
    variant="glass"
    destroy-on-close
  >
    <div class="audio-lm-panel">
      <DqSegmented
        v-model="mode"
        block
        size="sm"
        :options="[
          { label: 'Inspire', value: 'inspire' },
          { label: 'Format', value: 'format' },
          { label: 'Generate', value: 'generate' },
        ]"
      />
      <DqInput v-model="prompt" type="textarea" :rows="3" :placeholder="$t('audio.promptPlaceholder')" />
      <DqInput v-model="lyrics" type="textarea" :rows="5" :placeholder="$t('audio.lyricsPlaceholder')" />

      <div class="audio-lm-panel__grid">
        <DqInput v-model="duration" size="sm" inputmode="numeric" :placeholder="$t('audio.duration')" />
        <DqInput v-model="bpm" size="sm" inputmode="numeric" placeholder="BPM" />
        <DqInput v-model="keyscale" size="sm" :placeholder="$t('audio.keyScale')" />
        <DqInput v-model="timesignature" size="sm" :placeholder="$t('audio.timeSignature')" />
        <DqInput v-model="vocalLanguage" size="sm" :placeholder="$t('audio.vocalLanguage')" />
        <DqInput v-model="seed" size="sm" inputmode="numeric" :placeholder="$t('audio.seed')" />
      </div>

      <details class="audio-lm-panel__advanced">
        <summary>{{ $t('studio.advancedParams') }}</summary>
        <div class="audio-lm-panel__grid">
          <label>{{ $t('audio.maxNewTokens') }}<DqInput v-model="maxNewTokens" size="sm" inputmode="numeric" /></label>
          <label>{{ $t('create.temperature') }}<DqInput v-model="temperature" size="sm" inputmode="decimal" /></label>
          <label>{{ $t('audio.topP') }}<DqInput v-model="topP" size="sm" inputmode="decimal" /></label>
          <label>{{ $t('audio.topK') }}<DqInput v-model="topK" size="sm" inputmode="numeric" /></label>
          <label>{{ $t('audio.cfgScale') }}<DqInput v-model="cfgScale" size="sm" inputmode="decimal" /></label>
          <label class="audio-lm-panel__switch">{{ $t('audio.useFsm') }}<DqSwitch v-model="useFsm" size="sm" /></label>
        </div>
        <DqInput v-model="negativePrompt" type="textarea" :rows="2" :placeholder="$t('audio.negativePrompt')" />
      </details>

      <DqProgress v-if="submitting" :percentage="progress" :stroke-width="6" />
      <DqAlert v-if="error" type="error" :closable="false" :title="error" />

      <div v-if="result" class="audio-lm-panel__result">
        <strong>{{ $t('audio.lmResult') }}</strong>
        <DqInput v-model="resultCaption" type="textarea" :rows="3" />
        <DqInput v-model="resultLyrics" type="textarea" :rows="6" />
        <span v-if="resultCodes.length" class="audio-lm-panel__codes">
          {{ $t('audio.lmCodes', { count: resultCodes.length }) }}
        </span>
      </div>
      <DqEmpty v-else :description="$t('audio.lmNoResult')" />
    </div>

    <template #footer>
      <DqButton size="sm" @click="open = false">{{ $t('common.cancel') }}</DqButton>
      <DqButton v-if="result" size="sm" @click="apply(false)">{{ $t('audio.lmApply') }}</DqButton>
      <DqButton v-if="result && resultCodes.length" size="sm" @click="apply(true)">
        {{ $t('audio.lmApplyGenerate') }}
      </DqButton>
      <DqButton type="primary" size="sm" :loading="submitting" :disabled="!prompt.trim() && !lyrics.trim()" @click="run">
        {{ $t('audio.lmRun') }}
      </DqButton>
    </template>
  </DqDialog>
</template>

<script setup lang="ts">
import { computed, reactive, ref, watch } from 'vue';

export interface AudioLmResult {
  caption: string;
  lyrics: string;
  bpm: number | null;
  duration: number | null;
  keyscale: string;
  timesignature: string;
  language: string;
  audio_code_indices: number[];
}

const open = defineModel<boolean>('open', { default: false });
const props = defineProps<{
  initial: Record<string, any>;
  result: AudioLmResult | null;
  submitting: boolean;
  progress: number;
  error?: string;
}>();
const emit = defineEmits<{
  run: [payload: Record<string, unknown>];
  apply: [result: AudioLmResult, generate: boolean];
}>();

const mode = ref('inspire');
const prompt = ref('');
const lyrics = ref('');
const duration = ref('30');
const bpm = ref('');
const keyscale = ref('');
const timesignature = ref('');
const vocalLanguage = ref('');
const seed = ref('');
const maxNewTokens = ref('512');
const temperature = ref('0.85');
const topP = ref('0.9');
const topK = ref('0');
const cfgScale = ref('2');
const useFsm = ref(true);
const negativePrompt = ref('');
const edited = reactive({ caption: '', lyrics: '' });

const resultCaption = computed({
  get: () => edited.caption || props.result?.caption || '',
  set: (value) => { edited.caption = value; },
});
const resultLyrics = computed({
  get: () => edited.lyrics || props.result?.lyrics || '',
  set: (value) => { edited.lyrics = value; },
});
const resultCodes = computed(() => props.result?.audio_code_indices || []);

function numberValue(value: string): number | null {
  if (!value.trim()) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

watch(open, (visible) => {
  if (!visible) return;
  prompt.value = String(props.initial.prompt || '');
  lyrics.value = String(props.initial.lyrics || '');
  duration.value = String(props.initial.duration || 30);
  bpm.value = props.initial.bpm == null ? '' : String(props.initial.bpm);
  keyscale.value = String(props.initial.key_scale || '');
  timesignature.value = String(props.initial.time_signature || '');
  vocalLanguage.value = String(props.initial.vocal_language || '');
  seed.value = props.initial.seed == null ? '' : String(props.initial.seed);
  edited.caption = '';
  edited.lyrics = '';
});

watch(() => props.result, () => {
  edited.caption = '';
  edited.lyrics = '';
});

function run() {
  emit('run', {
    mode: mode.value,
    prompt: prompt.value.trim(),
    lyrics: lyrics.value.trim(),
    duration: numberValue(duration.value),
    bpm: numberValue(bpm.value),
    keyscale: keyscale.value.trim(),
    timesignature: timesignature.value.trim(),
    vocal_language: vocalLanguage.value.trim(),
    max_new_tokens: numberValue(maxNewTokens.value),
    temperature: numberValue(temperature.value),
    top_p: numberValue(topP.value),
    top_k: numberValue(topK.value),
    seed: numberValue(seed.value),
    use_fsm: useFsm.value,
    cfg_scale: numberValue(cfgScale.value),
    negative_prompt: negativePrompt.value,
  });
}

function apply(generate: boolean) {
  if (!props.result) return;
  emit('apply', {
    ...props.result,
    caption: resultCaption.value,
    lyrics: resultLyrics.value,
    bpm: numberValue(bpm.value) ?? props.result.bpm,
    duration: numberValue(duration.value) ?? props.result.duration,
    keyscale: keyscale.value || props.result.keyscale,
    timesignature: timesignature.value || props.result.timesignature,
    language: vocalLanguage.value || props.result.language,
  }, generate);
  open.value = false;
}
</script>

<style scoped>
.audio-lm-panel {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.audio-lm-panel__grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
}

.audio-lm-panel__advanced {
  color: var(--dq-label-secondary);
  font-size: var(--dq-font-size-caption);
}

.audio-lm-panel__advanced summary {
  margin-bottom: 10px;
  cursor: pointer;
}

.audio-lm-panel__advanced .audio-lm-panel__grid {
  margin-bottom: 8px;
}

.audio-lm-panel__switch {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.audio-lm-panel__result {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.audio-lm-panel__codes {
  color: var(--dq-label-secondary);
  font-size: var(--dq-font-size-caption);
}

@media (max-width: 600px) {
  .audio-lm-panel__grid {
    grid-template-columns: 1fr;
  }
}
</style>
