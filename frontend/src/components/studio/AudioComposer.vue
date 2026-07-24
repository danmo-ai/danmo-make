<template>
  <div class="audio-composer studio-composer-shell dq-glass--panel">
    <!-- Title -->
    <DqInput
      v-model="localTitle"
      size="sm"
      :placeholder="$tt('studio.workTitlePlaceholder')"
      class="audio-composer__title"
    />

    <!-- Prompt -->
    <div class="audio-composer__prompt-block">
    <div class="audio-composer__prompt-wrap">
      <DqInput
        v-model="localPrompt"
        type="textarea"
        :rows="5"
        :placeholder="promptPlaceholder"
        resize="none"
        class="audio-composer__prompt"
        @keydown="onPromptKeydown"
      />
      <!-- Reference / cover source inside textarea -->
      <div class="audio-composer__ref-area">
        <div v-if="referenceMedia" class="audio-composer__ref-pill">
          <ComposerIconTip
            :content="audioPlaying ? $t('create.composerTip.pauseRef') : $t('create.composerTip.playRef')"
          >
            <DqIconButton
              type="text"
              size="xs"
              class="audio-composer__ref-play"
              :aria-label="audioPlaying ? $tt('audio.pause') : $tt('audio.play')"
              @click.stop="toggleAudioPlayback"
            >
              <DqIcon :size="10"><Play v-if="!audioPlaying" /><Pause v-else /></DqIcon>
            </DqIconButton>
          </ComposerIconTip>
          <span class="audio-composer__ref-label">{{ referenceMedia.label }}</span>
          <ComposerIconTip :content="$t('create.composerTip.removeRef')">
            <DqIconButton
              type="text"
              size="xs"
              :aria-label="$tt('common.delete')"
              @click="$emit('remove-reference')"
            >
              <DqIcon :size="10"><Close /></DqIcon>
            </DqIconButton>
          </ComposerIconTip>
        </div>
        <ComposerIconTip v-else-if="workMode === 'cover'" :content="$t('create.composerTip.pickCoverSource')">
          <DqIconButton
            type="text"
            size="xs"
            class="audio-composer__ref-add"
            :aria-label="$tt('audio.pickCoverSource')"
            @click="$emit('pick-reference')"
          >
            <DqIcon :size="14"><Picture /></DqIcon>
          </DqIconButton>
        </ComposerIconTip>
      </div>

      <!-- Enhance brief + presets (prompt corner) -->
      <div class="audio-composer__preset-area">
        <ComposerIconTip
          :content="localPrompt.trim() ? $t('create.composerTip.enhanceMusicBrief') : $t('create.composerTip.enhanceEmpty')"
        >
          <DqIconButton
            type="text"
            size="xs"
            :disabled="briefEnhancing || !localPrompt.trim()"
            :aria-label="$t('create.enhanceMusicBrief')"
            @click="onEnhanceBriefClick"
          >
            <DqIcon :size="12"><MagicStick /></DqIcon>
          </DqIconButton>
        </ComposerIconTip>
        <DqDropdown
          v-if="styles && Object.keys(styles).length > 0"
          trigger="click"
          size="sm"
          @command="onStyleCommand"
        >
          <DqIconButton
            type="text"
            size="xs"
            class="audio-composer__preset-btn"
            :label="$t('create.composerTip.preset')"
          >
            <DqIcon :size="14"><DocumentCopy /></DqIcon>
          </DqIconButton>
          <template #dropdown>
            <DqDropdownMenu>
              <DqDropdownItem
                v-for="(preset, name) in styles"
                :key="name"
                :command="name"
              >
                {{ presetLabel(name, preset) }}
              </DqDropdownItem>
            </DqDropdownMenu>
          </template>
        </DqDropdown>
      </div>
    </div>
    <ComposerPromptApplyStrip
      v-if="promptApplyPreview"
      :preview="promptApplyPreview"
      @replace="$emit('prompt-apply-replace')"
      @append="$emit('prompt-apply-append')"
      @dismiss="$emit('prompt-apply-dismiss')"
    />
    </div>

    <!-- Lyrics: primary creative input (create mode or cover with support) -->
    <div v-if="showLyrics && (workMode === 'create' || supportsCoverLyrics)" class="audio-composer__lyrics-wrap">
      <div class="audio-composer__lyrics-head">
        <span class="audio-composer__lyrics-label">
          {{ workMode === 'cover' ? $tt('audio.coverLyricsLabel') : $tt('audio.lyrics') }}
          <span v-if="lyricsRequired" class="audio-composer__lyrics-required" aria-hidden="true">*</span>
        </span>
        <ComposerIconTip
          v-if="!localInstrumental"
          :content="localPrompt.trim() ? $t('create.composerTip.generateLyrics') : $t('create.composerTip.lyricsEmpty')"
        >
          <DqIconButton
            type="text"
            size="xs"
            :disabled="lyricsLoading || !localPrompt.trim()"
            :aria-label="$t('audio.generateLyrics')"
            @click="$emit('generate-lyrics')"
          >
            <DqIcon :size="12"><MagicStick /></DqIcon>
          </DqIconButton>
        </ComposerIconTip>
        <div class="audio-composer__inline-switch">
          <span>{{ $tt('audio.instrumental') }}</span>
          <DqSwitch
            v-model="localInstrumental"
            size="sm"
            :disabled="!supportsInstrumental"
          />
        </div>
      </div>
      <DqInput
        v-if="!localInstrumental"
        v-model="localLyrics"
        type="textarea"
        :rows="4"
        :placeholder="workMode === 'cover' ? $tt('audio.lyricsPlaceholder') : $tt('audio.lyricsPlaceholder')"
        resize="none"
        class="audio-composer__lyrics"
      />
      <ComposerPromptApplyStrip
        v-if="lyricsApplyPreview"
        :preview="lyricsApplyPreview"
        @replace="$emit('lyrics-apply-replace')"
        @append="$emit('lyrics-apply-append')"
        @dismiss="$emit('lyrics-apply-dismiss')"
      />
      <p class="audio-composer__lyrics-hint">
        {{
          workMode === 'cover'
            ? coverLyricsHintText
            : lyricsHintText
        }}
      </p>
    </div>

    <!-- Toolbar -->
    <ComposerToolbar>
      <template #mode>
        <DqSegmented
          :model-value="localWorkMode"
          size="sm"
          :options="workModeOptions"
          @update:model-value="localWorkMode = $event"
        />
      </template>
      <template #controls>
        <DqSelect
          v-model="localModel"
          size="sm"
          variant="ghost"
          
          class="studio-composer-toolbar__select studio-composer-toolbar__select--model"
          
          @change="(val: string) => emit('model-change', val)"
        >
          <DqOption
            v-for="item in modelOptions"
            :key="item.value"
            :label="item.label"
            :value="item.value"
            :disabled="item.disabled"
          >
            <DqTag
              v-if="item.commercialUseAllowed"
              size="sm"
              type="success"
              class="audio-composer__model-badge"
            >
              {{ $tt('download.commercialUseBadge') }}
            </DqTag>
          </DqOption>
        </DqSelect>

        <DqSelect
          v-model="localDuration"
          size="sm"
          variant="ghost"
          
          class="studio-composer-toolbar__select studio-composer-toolbar__select--duration"
        
        >
          <DqOption
            v-for="opt in durationOptions"
            :key="opt.value"
            :label="opt.label"
            :value="opt.value"
          />
        </DqSelect>
      </template>
      <template #actions>
        <DqSelect
          v-model="localBatchCount"
          size="sm"
          variant="ghost"
          
          class="studio-composer-toolbar__select studio-composer-toolbar__select--batch"
        
        >
          <DqOption
            v-for="n in 4"
            :key="n"
            :label="`x${n}`"
            :value="n"
          />
        </DqSelect>

        <DqButton
          type="primary"
          size="sm"
          class="studio-composer-toolbar__generate"
          :disabled="!canGenerate || submitting"
          @click="$emit('generate')"
        >
          <DqIcon :size="16"><MagicStick /></DqIcon>
          <span>{{ primaryActionLabel }}</span>
        </DqButton>
      </template>
    </ComposerToolbar>

    <ComposerAdvancedCollapsible
      v-model:open="advancedOpen"
      :has-custom-params="hasCustomParams"
      @reset-defaults="$emit('reset-defaults')"
    >
      <div class="audio-composer__advanced-inner">
              <!-- Steps & Guidance (create mode only — not used for cover) -->
              <div v-if="workMode !== 'cover' && (stepsDef || guidanceDef)" class="audio-composer__advanced-row">
                <div v-if="stepsDef" class="audio-composer__field">
                  <label>{{ $tt('create.steps') }}</label>
                  <DqSlider
                    v-model="localParams.steps"
                    :min="stepsDef.min"
                    :max="stepsDef.max"
                    :step="stepsDef.step"
                  />
                  <span class="audio-composer__field-val">{{ localParams.steps }}</span>
                </div>
                <div v-if="guidanceDef" class="audio-composer__field">
                  <label>{{ $tt('create.guidance') }}</label>
                  <DqSlider
                    v-model="localParams.guidance"
                    :min="guidanceDef.min"
                    :max="guidanceDef.max"
                    :step="guidanceDef.step"
                  />
                  <span class="audio-composer__field-val">{{ localParams.guidance }}</span>
                </div>
              </div>

              <!-- Temperature & Top-K (create mode only) -->
              <div v-if="workMode !== 'cover' && (temperatureDef || topKDef)" class="audio-composer__advanced-row">
                <div v-if="temperatureDef" class="audio-composer__field">
                  <label>{{ $tt('create.temperature') }}</label>
                  <DqSlider
                    v-model="localParams.temperature"
                    :min="temperatureDef.min"
                    :max="temperatureDef.max"
                    :step="temperatureDef.step"
                  />
                  <span class="audio-composer__field-val">{{ localParams.temperature }}</span>
                </div>
                <div v-if="topKDef" class="audio-composer__field">
                  <label>{{ $tt('audio.topK') }}</label>
                  <DqSlider
                    v-model="localParams.top_k"
                    :min="topKDef.min"
                    :max="topKDef.max"
                    :step="1"
                  />
                  <span class="audio-composer__field-val">{{ localParams.top_k }}</span>
                </div>
              </div>

              <!-- Seed -->
              <div class="audio-composer__advanced-row">
                <div class="audio-composer__field">
                  <label>{{ $tt('create.seed') }}</label>
                  <div class="audio-composer__seed-wrap">
                    <DqInput
                      v-model="seedInput"
                      size="sm"
                      :placeholder="$tt('studio.seedPlaceholder')"
                      style="width: 100px"
                    />
                    <DqIconButton
                      type="text"
                      size="xs"
                      :label="$tt('create.randomSeed')"
                      @click="randomizeSeed"
                    >
                      <DqIcon :size="12"><Refresh /></DqIcon>
                    </DqIconButton>
                  </div>
                </div>
              </div>

              <!-- Cover source fidelity -->
              <div v-if="workMode === 'cover' && showCoverFidelity" class="audio-composer__advanced-row">
                <div class="audio-composer__field audio-composer__field--full">
                  <label>{{ $tt('audio.coverFidelity') }}</label>
                  <div class="audio-composer__fidelity-slider-wrap">
                    <span class="audio-composer__fidelity-label audio-composer__fidelity-label--left">{{ $tt('audio.coverFidelityCreative') }}</span>
                    <DqSlider
                      v-model="localParams.source_fidelity"
                      :min="0"
                      :max="1"
                      :step="0.05"
                      class="audio-composer__fidelity-slider"
                    />
                    <span class="audio-composer__fidelity-label audio-composer__fidelity-label--right">{{ $tt('audio.coverFidelityStrict') }}</span>
                  </div>
                  <p class="audio-composer__fidelity-hint">{{
                    localParams.source_fidelity <= 0.3 ? $tt('audio.coverFidelityHint') + ' (' + $tt('audio.coverFidelityCreative') + ')'
                    : localParams.source_fidelity <= 0.7 ? $tt('audio.coverFidelityHint') + ' (' + $tt('audio.coverFidelityMid') + ')'
                    : $tt('audio.coverFidelityHint')
                  }}</p>
                </div>
              </div>

              <!-- LoRA -->
              <div
                v-if="workMode !== 'cover' && currentModelConfig?.parameters?.lora_support && compatibleLoras?.length"
                class="audio-composer__advanced-row"
              >
                <div class="audio-composer__field audio-composer__field--full" style="flex-direction: column; align-items: flex-start; gap: 8px;">
                  <div style="display: flex; align-items: center; gap: 10px; width: 100%;">
                    <label>{{ $tt('studio.loraLabel') }}</label>
                    <DqSelect
                      v-model="localParams.lora"
                      size="sm"
                      clearable
                      :placeholder="$tt('studio.noLora')"
                      style="flex: 1; max-width: 300px;"
                    >
                      <DqOption
                        v-for="l in compatibleLoras"
                        :key="String(l.id)"
                        :label="loraOptionLabel(l)"
                        :value="l.id"
                      />
                    </DqSelect>
                  </div>
                  <div v-if="localParams.lora" style="display: flex; align-items: center; gap: 10px; width: 100%;">
                    <label style="min-width: 60px;">{{ $tt('create.loraScale') }}</label>
                    <DqSlider v-model="localParams.lora_scale" :min="0" :max="2" :step="0.05" style="flex: 1;" />
                    <span class="audio-composer__field-val">{{ localParams.lora_scale }}</span>
                  </div>
                </div>
              </div>

              <!-- Negative prompt -->
              <div v-if="showNegativePrompt" class="audio-composer__advanced-row">
                <div class="audio-composer__field audio-composer__field--full">
                  <label>{{ $tt('create.negativePrompt') }}</label>
                  <DqInput
                    v-model="localParams.negative_prompt"
                    type="textarea"
                    :rows="2"
                    :placeholder="$tt('create.negativePlaceholder')"
                    resize="none"
                  />
                </div>
              </div>

              <!-- Music params: BPM, Key, Time Signature -->
              <div v-if="(workMode === 'create' || supportsCoverMusicParams) && (supportsBpm || supportsKeyScale || supportsTimeSignature)" class="audio-composer__advanced-row">
                <div v-if="supportsBpm" class="audio-composer__field">
                  <label>{{ $tt('audio.bpm') }}</label>
                  <DqInput
                    v-model="bpmInput"
                    size="sm"
                    inputmode="numeric"
                    :placeholder="$tt('audio.bpmAuto')"
                    class="audio-composer__optional-input"
                  />
                </div>
                <div v-if="supportsKeyScale" class="audio-composer__field">
                  <label>{{ $tt('audio.keyScale') }}</label>
                  <DqSelect v-model="localParams.key_scale" size="sm" clearable :placeholder="$tt('audio.keyScaleAuto')">
                    <DqOption v-for="k in musicalKeys" :key="k" :label="k" :value="k" />
                  </DqSelect>
                </div>
                <div v-if="supportsTimeSignature" class="audio-composer__field">
                  <label>{{ $tt('audio.timeSignature') }}</label>
                  <DqSelect v-model="localParams.time_signature" size="sm" clearable :placeholder="$tt('audio.timeSignatureAuto')">
                    <DqOption
                      v-for="ts in timeSignatures"
                      :key="ts.value"
                      :label="ts.label"
                      :value="ts.value"
                    />
                  </DqSelect>
                </div>
              </div>

              <!-- Vocal type / language -->
              <div v-if="(workMode === 'create' || supportsCoverVocals) && (supportsVocalType || supportsVocalLanguage)" class="audio-composer__advanced-row">
                <div v-if="supportsVocalLanguage" class="audio-composer__field">
                  <label>{{ $tt('audio.vocalLanguage') }}</label>
                  <DqSelect v-model="localParams.vocal_language" size="sm" clearable :placeholder="$tt('audio.vocalLanguageAuto')">
                    <DqOption v-for="l in vocalLanguages" :key="l.value" :label="l.label" :value="l.value" />
                  </DqSelect>
                </div>
                <div v-if="supportsVocalType" class="audio-composer__field">
                  <label>{{ $tt('audio.vocalType') }}</label>
                  <DqSelect v-model="localParams.vocal_type" size="sm" clearable :placeholder="$tt('audio.vocalTypeAuto')">
                    <DqOption v-for="vt in vocalTypes" :key="vt.value" :label="vt.label" :value="vt.value" />
                  </DqSelect>
                </div>
              </div>
      </div>
    </ComposerAdvancedCollapsible>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue';
import ComposerAdvancedCollapsible from './ComposerAdvancedCollapsible.vue';
import ComposerToolbar from './ComposerToolbar.vue';
import ComposerPromptApplyStrip from './ComposerPromptApplyStrip.vue';
import ComposerIconTip from './ComposerIconTip.vue';
import { useI18n } from 'vue-i18n';
import {
  Close,
  DocumentCopy,
  MagicStick,
  Picture,
  Pause,
  Play,
  Refresh,
} from '@danqing/dq-shell';
import { $tt } from '@/utils/i18n';
import { isAudioLyricsRequired, audioLyricsRequiredHintKey } from '@/utils/audioLyrics';
import { useVocalLanguageOptions } from '@/composables/useVocalLanguageOptions';
import { appendStyleBoost } from '@/utils/styleBoost';

const props = defineProps<{
  modelValue: string;
  title?: string;
  workMode: string;
  model: string;
  duration: number;
  batchCount?: number;
  composerBusy?: boolean;
  submitting?: boolean;
  generating: boolean;
  canGenerate: boolean;
  generateLabel: string;
  modelOptions: Array<{ label: string; value: string; disabled?: boolean; commercialUseAllowed?: boolean }>;
  durationOptions: Array<{ label: string; value: number }>;
  styles: Record<string, { applies_to?: string[]; positive?: string; negative?: string; trigger_words?: string; media_scope?: string }>;
  params: Record<string, any>;
  hasCustomParams: boolean;
  showNegativePrompt: boolean;
  showLyrics?: boolean;
  showCodecControls?: boolean;
  showCoverFidelity?: boolean;
  referenceMedia?: { type: string; previewUrl: string; label: string } | null;
  currentModelConfig?: Record<string, any> | null;
  compatibleLoras?: Array<Record<string, unknown>>;
  lyricsLoading?: boolean;
  briefEnhancing?: boolean;
  collapsed?: boolean;
  promptApplyPreview?: string | null;
  lyricsApplyPreview?: string | null;
}>();

const emit = defineEmits<{
  (e: 'update:modelValue', value: string): void;
  (e: 'update:title', value: string): void;
  (e: 'update:workMode', value: string): void;
  (e: 'update:model', value: string): void;
  (e: 'update:duration', value: number): void;
  (e: 'update:batchCount', value: number): void;
  (e: 'update:params', value: Record<string, any>): void;
  (e: 'generate'): void;
  (e: 'pick-reference'): void;
  (e: 'remove-reference'): void;
  (e: 'model-change', value: string): void;
  (e: 'reset-defaults'): void;
  (e: 'generate-lyrics'): void;
  (e: 'enhance-brief', ctx?: { stylePositive?: string }): void;
  (e: 'prompt-apply-replace'): void;
  (e: 'prompt-apply-append'): void;
  (e: 'prompt-apply-dismiss'): void;
  (e: 'lyrics-apply-replace'): void;
  (e: 'lyrics-apply-append'): void;
  (e: 'lyrics-apply-dismiss'): void;
}>();

const { t: $t } = useI18n();

const primaryActionLabel = computed(() => {
  if (props.composerBusy) return $t('create.addToBatch');
  if (props.generating) return $t('audio.generating');
  return props.generateLabel;
});

const vocalLanguages = useVocalLanguageOptions();

const advancedOpen = ref(false);
const lastStylePositive = ref('');
const seedInput = ref('');
const bpmInput = ref('');
const audioPlaying = ref(false);
const audioEl = ref<HTMLAudioElement | null>(null);

function toggleAudioPlayback() {
  if (!props.referenceMedia?.previewUrl) return;
  if (!audioEl.value) {
    audioEl.value = new Audio(props.referenceMedia.previewUrl);
    audioEl.value.addEventListener('ended', () => { audioPlaying.value = false; });
    audioEl.value.addEventListener('pause', () => { audioPlaying.value = false; });
    audioEl.value.addEventListener('play', () => { audioPlaying.value = true; });
  }
  if (audioPlaying.value) {
    audioEl.value.pause();
  } else {
    audioEl.value.play().catch(() => {
      audioPlaying.value = false;
    });
  }
}

function patchParams(patch: Record<string, unknown>) {
  emit('update:params', { ...props.params, ...patch });
}

const localTitle = computed({
  get: () => props.title || '',
  set: (v) => emit('update:title', v),
});

const localPrompt = computed({
  get: () => props.modelValue,
  set: (v) => emit('update:modelValue', v),
});

const localWorkMode = computed({
  get: () => props.workMode,
  set: (v) => emit('update:workMode', v),
});

const workModeOptions = computed(() => [
  { label: $tt('action.audio.create'), value: 'create' },
  { label: $tt('action.audio.cover'), value: 'cover' },
]);

const localModel = computed({
  get: () => props.model,
  set: (v) => emit('update:model', v),
});

const localDuration = computed({
  get: () => props.duration,
  set: (v) => emit('update:duration', v),
});

const localBatchCount = computed({
  get: () => props.batchCount ?? 1,
  set: (v) => emit('update:batchCount', v),
});

const localParams = computed({
  get: () => props.params,
  set: (v) => emit('update:params', v),
});

const localLyrics = computed({
  get: () => String(props.params.lyrics || ''),
  set: (v) => patchParams({ lyrics: v }),
});

const localInstrumental = computed({
  get: () => !!props.params.instrumental,
  set: (v) => patchParams({ instrumental: v }),
});

const supportsInstrumental = computed(() => {
  const flag = props.currentModelConfig?.parameters?.supports_instrumental;
  return flag !== false;
});

const lyricsRequired = computed(() =>
  isAudioLyricsRequired(props.currentModelConfig, localInstrumental.value),
);

const lyricsHintText = computed(() => {
  if (localInstrumental.value) return $tt('audio.lyricsPlaceholderInstrumental');
  if (lyricsRequired.value) return $tt(audioLyricsRequiredHintKey(props.currentModelConfig));
  return $tt('audio.lyricsHint');
});

const coverLyricsHintText = computed(() => {
  if (localInstrumental.value) return $tt('audio.lyricsPlaceholderInstrumental');
  if (lyricsRequired.value) return $tt(audioLyricsRequiredHintKey(props.currentModelConfig));
  return $tt('audio.coverLyricsHint');
});

const promptPlaceholder = computed(() => {
  if (props.workMode === 'cover') {
    return $tt('audio.coverPromptPlaceholder');
  }
  return $tt('audio.promptPlaceholder');
});

// Seed sync
watch(() => props.params.seed, (v) => {
  seedInput.value = v != null ? String(v) : '';
}, { immediate: true });

watch(seedInput, (v) => {
  const n = parseInt(v, 10);
  patchParams({ seed: !v || isNaN(n) ? null : n });
});

watch(() => props.params.bpm, (v) => {
  bpmInput.value = v != null ? String(v) : '';
}, { immediate: true });

watch(bpmInput, (v) => {
  const trimmed = v.trim();
  if (!trimmed) {
    if (props.params.bpm != null) patchParams({ bpm: null });
    return;
  }
  const n = parseInt(trimmed, 10);
  if (isNaN(n)) return;
  const clamped = Math.min(300, Math.max(30, n));
  if (props.params.bpm !== clamped) patchParams({ bpm: clamped });
  if (String(clamped) !== trimmed) bpmInput.value = String(clamped);
});

function randomizeSeed() {
  patchParams({ seed: Math.floor(Math.random() * 2147483647) });
}

// Style / Preset
function loraOptionLabel(l: Record<string, unknown>): string {
  const base = String(l.name || l.id || '');
  if (l.source === 'user_trained') {
    return `${base} (${$t('studio.myLoraTag')})`;
  }
  return base;
}

function presetLabel(name: string, preset: Record<string, unknown>): string {
  const a = (preset.applies_to as string[]) || [];
  const hasC = a.includes('create');
  const hasCover = a.includes('cover');
  let tag = '';
  if (hasC && !hasCover) tag = $tt('audio.presetTagCreate');
  else if (hasCover && !hasC) tag = $tt('audio.presetTagCover');
  const display = (preset as any).name?.en || (preset as any).name?.zh || name;
  return tag ? `${tag} ${display}` : display;
}

function onEnhanceBriefClick() {
  const style = lastStylePositive.value.trim();
  emit('enhance-brief', style ? { stylePositive: style } : undefined);
}

function onStyleCommand(command: string) {
  const preset = props.styles[command];
  if (!preset) return;
  if (preset.positive) {
    lastStylePositive.value = String(preset.positive);
    localPrompt.value = appendStyleBoost(localPrompt.value || '', String(preset.positive));
  }
  if (preset.trigger_words) {
    const tw = String(preset.trigger_words).trim();
    if (tw) {
      localPrompt.value = localPrompt.value
        ? localPrompt.value + '\n' + tw
        : tw;
    }
  }
  if (preset.negative && props.showNegativePrompt) {
    patchParams({
      negative_prompt: localParams.value.negative_prompt
        ? localParams.value.negative_prompt + '\n' + preset.negative
        : preset.negative,
    });
  }
}

// Capability flags from registry
const paramDefs = computed(() => props.currentModelConfig?.parameters || {});

const stepsDef = computed(() => {
  const p = paramDefs.value.steps;
  if (!p) return null;
  return { min: p.min ?? 1, max: p.max ?? 100, step: p.step ?? 1, default: p.default ?? 8 };
});

const guidanceDef = computed(() => {
  const p = paramDefs.value.guidance;
  if (!p) return null;
  return { min: p.min ?? 1, max: p.max ?? 20, step: p.step ?? 0.1, default: p.default ?? 3.0 };
});

const temperatureDef = computed(() => {
  const p = paramDefs.value.temperature;
  if (!p) return null;
  return { min: p.min ?? 0.1, max: p.max ?? 2, step: p.step ?? 0.1, default: p.default ?? 1.0 };
});

const topKDef = computed(() => {
  const p = paramDefs.value.top_k;
  if (!p) return null;
  return { min: p.min ?? 10, max: p.max ?? 100, default: p.default ?? 50 };
});

const supportsBpm = computed(() => paramDefs.value.supports_bpm === true);
const supportsKeyScale = computed(() => props.currentModelConfig?.parameters?.supports_key_scale === true);
const supportsTimeSignature = computed(() => props.currentModelConfig?.parameters?.supports_time_signature === true);
const supportsVocalType = computed(() => props.currentModelConfig?.parameters?.supports_vocal_type === true);
const supportsVocalLanguage = computed(() => props.currentModelConfig?.parameters?.supports_vocal_language === true);

const supportsCoverLyrics = computed(() => props.currentModelConfig?.parameters?.cover_lyrics_support === true);
const supportsCoverMusicParams = computed(() => props.currentModelConfig?.parameters?.cover_music_params_support === true);
const supportsCoverVocals = computed(() => props.currentModelConfig?.parameters?.cover_lyrics_support === true);

const musicalKeys = [
  'C Major', 'C# Major', 'D Major', 'D# Major', 'E Major', 'F Major', 'F# Major',
  'G Major', 'G# Major', 'A Major', 'A# Major', 'B Major',
  'C Minor', 'C# Minor', 'D Minor', 'D# Minor', 'E Minor', 'F Minor', 'F# Minor',
  'G Minor', 'G# Minor', 'A Minor', 'A# Minor', 'B Minor',
];

const timeSignatures = [
  { label: '2/4', value: '2' }, { label: '3/4', value: '3' },
  { label: '4/4', value: '4' }, { label: '6/8', value: '6' },
];

const vocalTypes = computed(() => [
  { label: $tt('audio.vocalTypeMale'), value: 'male' },
  { label: $tt('audio.vocalTypeFemale'), value: 'female' },
  { label: $tt('audio.vocalTypeChorus'), value: 'chorus' },
  { label: $tt('audio.vocalTypeDuet'), value: 'duet' },
]);

function onPromptKeydown(e: KeyboardEvent) {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
    e.preventDefault();
    if (!props.submitting && props.canGenerate) {
      emit('generate');
    }
  }
}
</script>

<style scoped>
.audio-composer {
  border-radius: var(--dq-radius-group);
  padding: 14px 16px 12px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  width: 100%;
  min-width: 0;
  container-type: inline-size;
  box-sizing: border-box;
}

.audio-composer__title {
  margin: 0;
}

.audio-composer__title :deep(.dq-input) {
  border-radius: var(--dq-radius-input);
}

.audio-composer__prompt-block {
  display: flex;
  flex-direction: column;
}

.audio-composer__prompt-wrap {
  position: relative;
  margin: 0;
}

.audio-composer__prompt :deep(.dq-input--textarea) {
  border-radius: var(--dq-radius-group);
  font-size: var(--dq-font-size-body);
  line-height: 1.5;
  padding: 10px 12px 32px;
  min-height: 7.5rem;
  max-height: 14rem;
  resize: none;
  overflow-y: auto;
  transition: border-color 0.15s ease, background 0.15s ease;
}

/* Reference area inside textarea */
.audio-composer__ref-area {
  position: absolute;
  bottom: 6px;
  left: 8px;
  display: flex;
  align-items: center;
}

.audio-composer__ref-pill {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 2px 6px 2px 6px;
  background: var(--dq-fill-secondary);
  border-radius: 6px;
  border: 1px solid var(--dq-border-subtle);
}

.audio-composer__ref-label {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
}

.audio-composer__ref-add {
  opacity: 0.6;
  transition: opacity 0.2s;
}

.audio-composer__ref-add:hover {
  opacity: 1;
}

.audio-composer__preset-area {
  position: absolute;
  bottom: 6px;
  right: 8px;
  z-index: 1;
  display: flex;
  align-items: center;
}

.audio-composer__model-badge {
  font-size: var(--dq-font-size-caption);
  padding: 0 4px;
  height: 16px;
  line-height: 16px;
}

.audio-composer__preset-btn {
  opacity: 0.6;
  transition: opacity 0.2s;
}

.audio-composer__preset-btn:hover {
  opacity: 1;
}

/* Lyrics block (main composer, below style prompt) */
.audio-composer__lyrics-wrap {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.audio-composer__lyrics-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.audio-composer__lyrics-label {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  color: var(--dq-label-secondary);
}

.audio-composer__lyrics-required {
  color: var(--dq-danger);
  margin-left: 2px;
}

.audio-composer__lyrics :deep(.dq-input--textarea) {
  border-radius: var(--dq-radius-group);
  font-size: var(--dq-font-size-body);
  line-height: 1.5;
  padding: 10px 12px;
  min-height: 5.5rem;
  max-height: 9rem;
  resize: none;
  overflow-y: auto;
  transition: border-color 0.15s ease, background 0.15s ease;
}

.audio-composer__lyrics-hint {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
  color: var(--dq-label-tertiary);
}

/* Advanced panel */
.audio-composer__advanced-inner {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.audio-composer__advanced-row {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}

.audio-composer__field {
  display: flex;
  align-items: center;
  gap: 10px;
  flex: 1;
  min-width: 260px;
  overflow: hidden;
}

.audio-composer__field--full {
  flex: 1 1 100%;
  flex-direction: column;
  align-items: flex-start;
  gap: 6px;
}

.audio-composer__field label {
  font-size: var(--dq-font-size-caption);
  font-weight: 500;
  color: var(--dq-label-secondary);
  white-space: nowrap;
  width: 90px;
  flex-shrink: 0;
  overflow: hidden;
  text-overflow: ellipsis;
}

.audio-composer__field :deep(.dq-slider) {
  flex: 1;
}

.audio-composer__field-val {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
  min-width: 32px;
  text-align: right;
  font-variant-numeric: tabular-nums;
}

.audio-composer__seed-wrap {
  display: flex;
  align-items: center;
  gap: 4px;
}

.audio-composer__optional-input {
  flex: 1;
  min-width: 0;
}

.audio-composer__optional-input :deep(.dq-input) {
  width: 100%;
}

.audio-composer__field :deep(.dq-input),
.audio-composer__field :deep(.dq-select) {
  height: 28px;
}

.audio-composer__inline-switch {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
}

/* Cover fidelity slider */
.audio-composer__fidelity-slider-wrap {
  display: flex;
  align-items: center;
  gap: 12px;
  width: 100%;
}

.audio-composer__fidelity-label {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
  white-space: nowrap;
  flex-shrink: 0;
}

.audio-composer__fidelity-slider {
  flex: 1;
  min-width: 0;
}

.audio-composer__fidelity-hint {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
  line-height: 1.4;
}

</style>