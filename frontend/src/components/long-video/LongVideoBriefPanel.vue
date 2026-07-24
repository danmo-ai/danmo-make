<template>
  <section class="lv-script-studio lv-section">
    <div class="lv-script-studio__body">
        <div class="lv-script-studio__composer">
          <div class="lv-script-studio__composer-bar">
            <span class="lv-script-studio__composer-label">{{ $tt('video.longVideoScriptInputLabel') }}</span>
            <div class="lv-script-studio__composer-bar-actions">
              <input
                class="lv-script-studio__chapter-title dq-input"
                type="text"
                :value="chapterTitle"
                :placeholder="$tt('video.longVideoChapterTitlePh')"
                @input="onChapterTitleInput"
              />
              <DqButton size="sm" type="text" @click="triggerFileUpload">
                {{ $tt('video.longVideoScriptUploadTxt') }}
              </DqButton>
            </div>
            <input
              ref="fileInputRef"
              type="file"
              accept=".txt,text/plain"
              class="lv-script-studio__file-input"
              @change="onFileSelected"
            />
          </div>
          <textarea
            class="lv-script-studio__textarea dq-input dq-input--textarea lv-script-studio__textarea--long"
            :value="scriptText"
            :placeholder="$tt('video.longVideoScriptInputPlaceholder')"
            @input="onScriptInput"
          />
          <div class="lv-script-studio__parse-params">
            <span class="lv-script-studio__parse-params-label">{{ $tt('video.longVideoScriptParseParams') }}</span>
            <label class="lv-script-studio__parse-duration">
              <span class="lv-script-studio__parse-duration-label">{{ $tt('video.longVideoScriptParseLlm') }}</span>
              <DqSelect
                :model-value="scriptParseLlmModel"
                size="sm"
                class="lv-script-studio__parse-duration-select lv-script-studio__parse-llm-select"
                :title="$tt('video.longVideoScriptParseLlmHint')"
                @update:model-value="onScriptParseLlmChange"
              >
                <DqOption
                  v-for="opt in llmModelOptions"
                  :key="opt.value"
                  :label="opt.label"
                  :value="opt.value"
                />
              </DqSelect>
            </label>
            <label class="lv-script-studio__parse-duration">
              <span class="lv-script-studio__parse-duration-label">{{ $tt('video.longVideoTargetDuration') }}</span>
              <DqSelect
                :model-value="targetDurationSec"
                size="sm"
                class="lv-script-studio__parse-duration-select"
                :title="$tt('video.longVideoTargetDurationHint')"
                @update:model-value="onDurationChange"
              >
                <DqOption
                  v-for="sec in durationChoices"
                  :key="sec"
                  :label="$tt('video.longVideoTargetDurationSec', { sec })"
                  :value="sec"
                />
              </DqSelect>
            </label>
            <span class="lv-script-studio__parse-chip">
              {{ $tt('video.longVideoStoryboardShotEstimate', { n: estimatedShots }) }}
            </span>
            <label class="lv-script-studio__parse-duration">
              <span class="lv-script-studio__parse-duration-label">{{ $tt('video.longVideoSegmentDuration') }}</span>
              <DqSelect
                :model-value="segmentDurationSec"
                size="sm"
                class="lv-script-studio__parse-duration-select"
                :title="$tt('video.longVideoSegmentDurationHint')"
                @update:model-value="onSegmentDurationChange"
              >
                <DqOption
                  v-for="sec in segmentDurationChoices"
                  :key="sec"
                  :label="$tt('video.longVideoSegmentDurationSec', { sec })"
                  :value="sec"
                />
              </DqSelect>
            </label>
            <span
              v-if="scriptParsed && sceneBeats.length"
              class="lv-script-studio__parse-chip"
            >
              {{ $tt('video.longVideoChapterDurationEstimate', {
                shots: sceneBeats.length,
                sec: estimatedTotalDuration,
              }) }}
            </span>
          </div>
          <div
            v-if="parseLoading && parseProgressPhase"
            class="lv-script-studio__parse-progress"
            role="status"
            aria-live="polite"
          >
            <span class="lv-script-studio__parse-progress-label">
              {{ $tt('video.longVideoParseProgressTitle') }}
            </span>
            <ol class="lv-script-studio__parse-steps">
              <li
                v-for="phase in parsePhaseOrder"
                :key="phase"
                class="lv-script-studio__parse-step"
                :class="parseStepClass(phase)"
              >
                <span class="lv-script-studio__parse-step-dot" aria-hidden="true" />
                <span class="lv-script-studio__parse-step-label">
                  {{ $tt(`video.longVideoParsePhase${phaseKey(phase)}`) }}
                </span>
              </li>
            </ol>
          </div>
          <LongVideoParseRunBar
            v-if="scriptParsed"
            :project-id="projectId"
            :chapter-analysis="chapterAnalysis"
            :shot-count="parsedShotCount"
          />
          <div class="lv-script-studio__composer-foot">
            <div class="lv-script-studio__composer-foot-meta">
              <span v-if="editorFootStat" class="lv-script-studio__composer-stat">{{ editorFootStat }}</span>
              <span v-if="expandHint" class="lv-script-studio__composer-expand-hint">{{ expandHint }}</span>
            </div>
            <div class="lv-script-studio__composer-foot-actions">
              <DqButton
                size="sm"
                :loading="expandLoading"
                :disabled="expandLoading || parseLoading || !scriptText.trim()"
                @click="emit('expand')"
              >
                {{ expandLoading ? $tt('video.longVideoScriptExpanding') : $tt('video.longVideoScriptExpand') }}
              </DqButton>
              <DqButton
                size="sm"
                :loading="Boolean(parsing)"
                :disabled="parseLoading || !scriptText.trim()"
                @click="emit('parse')"
              >
                {{ parsing ? parseProgressButtonLabel : parseButtonLabel }}
              </DqButton>
              <DqButton
                v-if="scriptParsed"
                type="primary"
                size="sm"
                :loading="Boolean(generatingStoryboard)"
                :disabled="parseLoading || !scriptParsed"
                @click="emit('generate-storyboard')"
              >
                {{ generatingStoryboard ? $tt('video.longVideoScriptGeneratingStoryboard') : $tt('video.longVideoScriptGenerateStoryboard') }}
              </DqButton>
            </div>
          </div>
        </div>

      <aside class="lv-script-studio__insights">
        <DqAlert
          v-if="parseBlocked"
          type="error"
          :closable="false"
          class="lv-script-studio__parse-blocked"
          :title="parseBlockedReason || $tt('video.longVideoStoryboardTabBlocked')"
        >
          <div class="lv-script-studio__parse-blocked-actions">
            <DqButton size="sm" type="primary" :loading="parseLoading" @click="emit('generate-storyboard')">
              {{ $tt('video.longVideoParseBlockedRetry') }}
            </DqButton>
            <DqButton size="sm" type="text" @click="emit('clear-parse-blocked')">
              {{ $tt('video.longVideoParseBlockedDismiss') }}
            </DqButton>
          </div>
        </DqAlert>
        <DqAlert
          v-if="parseError"
          type="error"
          :closable="false"
          class="lv-script-studio__parse-error"
          :title="parseError"
        />
        <DqAlert
          v-if="qualityNotices.length"
          type="warning"
          :closable="false"
          class="lv-script-studio__parse-quality"
        >
          <template #title>{{ $tt('video.longVideoParseQualityPanelTitle', { count: qualityNotices.length }) }}</template>
          <ul class="lv-script-studio__parse-quality-list">
            <li v-for="(item, idx) in qualityNotices" :key="idx">{{ item }}</li>
          </ul>
        </DqAlert>
        <template v-if="!scriptParsed">
          <div class="lv-script-studio__empty">
            <div class="lv-script-studio__empty-icon" aria-hidden="true" />
            <h3 class="lv-script-studio__empty-title">{{ $tt('video.longVideoScriptInsightsEmpty') }}</h3>
            <p class="lv-script-studio__empty-text">{{ $tt('video.longVideoScriptInsightsEmptyHint') }}</p>
            <ul class="lv-script-studio__empty-list">
              <li>{{ $tt('video.longVideoScriptInsightItemSynopsis') }}</li>
              <li>{{ $tt('video.longVideoScriptInsightItemStyle') }}</li>
              <li>{{ $tt('video.longVideoScriptInsightItemScenes') }}</li>
              <li>{{ $tt('video.longVideoScriptInsightItemCast') }}</li>
            </ul>
          </div>
        </template>

        <template v-else>
          <div class="lv-script-studio__insights-parsed">
            <div class="lv-script-studio__insight-tabs">
              <DqSegmented
                v-model="insightTab"
                block
                size="sm"
                class="lv-script-studio__insight-segmented"
                :options="insightTabOptions"
              />
            </div>

            <div class="lv-script-studio__insight-panel">
              <div v-show="insightTab === 'synopsis'" class="lv-script-studio__card">
                <template v-if="synopsisParts.logline">
                  <h4 class="lv-script-studio__card-title">{{ $tt('video.longVideoScriptSynopsisTitle') }}</h4>
                  <p class="lv-script-studio__synopsis">{{ synopsisParts.logline }}</p>
                </template>
                <div
                  v-if="synopsisParts.mood"
                  class="lv-script-studio__insight-field"
                  :class="{ 'lv-script-studio__insight-field--bordered': synopsisParts.logline }"
                >
                  <span class="lv-script-studio__insight-field-label">
                    {{ $tt('video.longVideoScriptSynopsisMoodTitle') }}
                  </span>
                  <p class="lv-script-studio__synopsis lv-script-studio__synopsis--mood">{{ synopsisParts.mood }}</p>
                </div>
                <div
                  class="lv-script-studio__insight-field"
                  :class="{ 'lv-script-studio__insight-field--bordered': synopsisParts.logline || synopsisParts.mood }"
                >
                  <span class="lv-script-studio__insight-field-label">{{ $tt('video.longVideoScriptStyleTitle') }}</span>
                  <DqInput
                    :model-value="styleAnchor"
                    size="sm"
                    class="lv-script-studio__style-input"
                    :placeholder="$tt('video.longVideoProjectStylePh')"
                    :title="$tt('video.longVideoProjectStyleHint')"
                    @update:model-value="emit('update:styleAnchor', $event)"
                  />
                </div>
              </div>

              <div v-show="insightTab === 'scenes'" class="lv-script-studio__card lv-script-studio__card--scenes">
                <div class="lv-script-studio__card-head">
                  <h4 class="lv-script-studio__card-title">
                    {{ $tt('video.longVideoScriptScenesTitle', { n: sceneBeats.length }) }}
                  </h4>
                  <span class="lv-script-studio__card-meta">
                    {{ $tt('video.longVideoChapterDurationEstimate', {
                      shots: sceneBeats.length,
                      sec: estimatedTotalDuration,
                    }) }}
                  </span>
                </div>
                <ol class="lv-script-studio__scene-list">
                  <li
                    v-for="(scene, idx) in sceneBeats"
                    :key="scene.order ?? idx"
                    class="lv-script-studio__scene"
                  >
                    <div class="lv-script-studio__scene-index">{{ idx + 1 }}</div>
                    <div class="lv-script-studio__scene-body">
                      <div class="lv-script-studio__scene-title-row">
                        <input
                          class="lv-script-studio__scene-title dq-input"
                          type="text"
                          :value="scene.title || ''"
                          :placeholder="$tt('video.longVideoScriptSceneTitlePh')"
                          @input="onSceneTitleInput(idx, $event)"
                        />
                        <DqButton
                          v-if="(parsedShotCount ?? 0) > 0"
                          size="sm"
                          type="text"
                          class="lv-script-studio__scene-regen"
                          :loading="beatRegeneratingIndex === sceneBeatIndex(scene, idx)"
                          :disabled="parseLoading || beatRegeneratingIndex != null"
                          @click="emit('regenerate-beat', sceneBeatIndex(scene, idx))"
                        >
                          {{ $tt('video.longVideoBeatRegenerate') }}
                        </DqButton>
                      </div>
                      <div class="lv-script-studio__scene-fields">
                        <label class="lv-script-studio__scene-field">
                          <span class="lv-script-studio__scene-field-label">
                            {{ $tt('video.longVideoScriptSceneShotSize') }}
                          </span>
                          <DqSelect
                            :model-value="parsedSceneBeat(scene.beat).shotSize"
                            size="sm"
                            class="lv-script-studio__scene-shot-select"
                            @update:model-value="onSceneFieldInput(idx, 'shotSize', $event)"
                          >
                            <DqOption
                              v-for="opt in shotSizeOptionsForBeat(scene.beat)"
                              :key="opt"
                              :label="opt"
                              :value="opt"
                            />
                          </DqSelect>
                        </label>
                        <label class="lv-script-studio__scene-field lv-script-studio__scene-field--wide">
                          <span class="lv-script-studio__scene-field-label">
                            {{ $tt('video.longVideoScriptSceneLocation') }}
                          </span>
                          <DqInput
                            :model-value="parsedSceneBeat(scene.beat).location"
                            size="sm"
                            @update:model-value="onSceneFieldInput(idx, 'location', $event)"
                          />
                        </label>
                        <label class="lv-script-studio__scene-field lv-script-studio__scene-field--full">
                          <span class="lv-script-studio__scene-field-label">
                            {{ $tt('video.longVideoScriptSceneVisual') }}
                          </span>
                          <textarea
                            class="lv-script-studio__scene-visual dq-input dq-input--textarea"
                            :value="parsedSceneBeat(scene.beat).visual"
                            rows="2"
                            @input="onSceneVisualInput(idx, $event)"
                          />
                        </label>
                      </div>
                    </div>
                  </li>
                </ol>
              </div>

              <div v-show="insightTab === 'cast'" class="lv-script-studio__card">
                <div class="lv-script-studio__card-head">
                  <h4 class="lv-script-studio__card-title">
                    {{ $tt('video.longVideoScriptCharactersTitle', { n: characters.length }) }}
                  </h4>
                  <span class="lv-script-studio__card-meta">
                    {{ $tt('video.longVideoCastLookCount', { n: totalLookCount }) }}
                  </span>
                </div>
                <ul class="lv-script-studio__cast-grid">
                  <li v-for="ch in characters" :key="ch.id" class="lv-script-studio__cast-card">
                    <div class="lv-script-studio__cast-avatar">{{ characterInitial(ch.name) }}</div>
                    <div class="lv-script-studio__cast-info">
                      <span class="lv-script-studio__cast-name">
                        {{ ch.name.trim() || $tt('video.longVideoCastNewCharacterName') }}
                      </span>
                      <ul v-if="ch.looks.length" class="lv-script-studio__look-tags">
                        <li v-for="look in ch.looks" :key="look.id" class="lv-script-studio__look-tag">
                          {{ look.label }}
                        </li>
                      </ul>
                      <template v-if="parsedCharacterLook(ch).role">
                        <span class="lv-script-studio__cast-role">
                          {{ $tt('video.longVideoScriptCastRoleLabel', { role: parsedCharacterLook(ch).role }) }}
                        </span>
                      </template>
                      <p v-if="parsedCharacterLook(ch).appearance" class="lv-script-studio__cast-body">
                        {{ parsedCharacterLook(ch).appearance }}
                      </p>
                    </div>
                  </li>
                </ul>
              </div>
            </div>

            <div class="lv-script-studio__next">
              <div class="lv-script-studio__next-copy">
                <span class="lv-script-studio__next-kicker">{{ $tt('video.longVideoScriptNextTitle') }}</span>
                <p class="lv-script-studio__next-text">{{ $tt('video.longVideoScriptNextCast') }}</p>
              </div>
              <div class="lv-script-studio__next-actions">
                <DqButton type="primary" @click="emit('go-to-cast')">
                  {{ $tt('video.longVideoScriptNextCastBtn') }}
                </DqButton>
              </div>
            </div>
          </div>
        </template>
      </aside>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import { $mn } from '@/utils/i18n';
import { toast } from '@/utils/feedback';
import { useRegistryStore } from '@/stores/registry';
import type { LongVideoChapterAnalysis, LongVideoChapterScene, LongVideoCharacter } from '@/types';
import {
  composeSceneBeat,
  parseCharacterLookBody,
  parseSceneBeat,
  shotSizeOptions,
  splitSynopsisMood,
  type ParsedSceneBeat,
} from '@/utils/longVideoSceneBeat';
import { allocateShotDurations } from '@/utils/longVideoProject';
import { suggestScriptExpand } from '@/utils/llmMessages';
import LongVideoParseRunBar from './LongVideoParseRunBar.vue';

type ScriptInsightTab = 'synopsis' | 'scenes' | 'cast';

const MAX_TXT_BYTES = 512 * 1024;

const props = defineProps<{
  scriptText: string;
  chapterTitle: string;
  chapterAnalysis?: LongVideoChapterAnalysis | null;
  characters?: LongVideoCharacter[];
  styleAnchor?: string;
  scriptParsed?: boolean;
  targetDurationSec: number;
  segmentDurationSec: number;
  scriptParseLlmModel?: string;
  parsing?: boolean;
  parseProgressPhase?: string;
  parseProgressMessage?: string;
  beatRegeneratingIndex?: number | null;
  expanding?: boolean;
  generatingStoryboard?: boolean;
  parseError?: string;
  parseBlocked?: boolean;
  parseBlockedReason?: string;
  projectId?: string;
  parsedShotCount?: number;
}>();

const emit = defineEmits<{
  (e: 'update:scriptText', value: string): void;
  (e: 'update:chapterTitle', value: string): void;
  (e: 'update:chapterAnalysis', value: LongVideoChapterAnalysis | undefined): void;
  (e: 'update:targetDurationSec', value: number): void;
  (e: 'update:segmentDurationSec', value: number): void;
  (e: 'update:scriptParseLlmModel', value: string): void;
  (e: 'update:styleAnchor', value: string): void;
  (e: 'expand'): void;
  (e: 'parse'): void;
  (e: 'generate-storyboard'): void;
  (e: 'regenerate-beat', beatIndex: number): void;
  (e: 'clear-parse-blocked'): void;
  (e: 'go-to-cast'): void;
}>();

const { t: $tt, locale } = useI18n();
const registryStore = useRegistryStore();
const fileInputRef = ref<HTMLInputElement | null>(null);
const insightTab = ref<ScriptInsightTab>('scenes');

onMounted(() => {
  void registryStore.load();
});

function hasLlmChatAction(actions: unknown): boolean {
  if (!actions || typeof actions !== 'object') return false;
  const row = actions as Record<string, unknown>;
  return row.chat != null || row.enhance != null;
}

const llmModelOptions = computed(() => {
  const models = registryStore.registry?.models || {};
  return Object.entries(models)
    .filter(([, cfg]) => cfg.media === 'llm' && cfg.category === 'llm_models' && hasLlmChatAction(cfg.actions))
    .map(([id, cfg]) => ({
      value: id,
      label: $mn(cfg, id),
    }))
    .sort((a, b) => a.label.localeCompare(b.label));
});

const durationChoices = [30, 45, 60, 90, 120] as const;
const segmentDurationChoices = [3, 5, 8] as const;

const parseLoading = computed(() => Boolean(props.parsing) || Boolean(props.generatingStoryboard));
const parseProgressPhase = computed(() => props.parseProgressPhase?.trim() || '');
const parseProgressMessage = computed(() => props.parseProgressMessage?.trim() || '');

const PARSE_PHASE_ORDER = [
  'decompose',
  'beat_plan',
  'shot_spec',
  'finalize',
  'review_retry',
  'done',
] as const;

const parsePhaseOrder = computed(() =>
  PARSE_PHASE_ORDER.filter((phase) => phase !== 'review_retry' || isReviewRetry.value),
);

const isReviewRetry = computed(() => parseProgressPhase.value === 'review_retry');

function effectiveParsePhase(): string {
  const phase = parseProgressPhase.value;
  if (phase === 'review_retry') {
    const pass = parseProgressMessage.value;
    if (pass && (PARSE_PHASE_ORDER as readonly string[]).includes(pass)) return pass;
  }
  return phase;
}

function phaseKey(phase: string): string {
  return phase
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join('');
}

function parseStepClass(phase: string): Record<string, boolean> {
  const current = effectiveParsePhase();
  if (!current) return {};
  if (phase === 'review_retry') {
    return {
      'lv-script-studio__parse-step--active': isReviewRetry.value,
    };
  }
  const curIdx = PARSE_PHASE_ORDER.indexOf(current as (typeof PARSE_PHASE_ORDER)[number]);
  const idx = PARSE_PHASE_ORDER.indexOf(phase as (typeof PARSE_PHASE_ORDER)[number]);
  if (curIdx < 0 || idx < 0) return {};
  const reviewActive = isReviewRetry.value;
  return {
    'lv-script-studio__parse-step--done': idx < curIdx && !reviewActive,
    'lv-script-studio__parse-step--active': idx === curIdx && !reviewActive,
    'lv-script-studio__parse-step--retry-parent': reviewActive && idx === curIdx,
  };
}

const parseProgressButtonLabel = computed(() => {
  const phase = parseProgressPhase.value;
  if (!phase) return $tt('video.longVideoScriptParsing');
  if (phase === 'review_retry') {
    const pass = parseProgressMessage.value;
    const passLabel = pass
      ? $tt(`video.longVideoParsePhase${phaseKey(pass)}`)
      : $tt('video.longVideoParsePhaseReviewRetry');
    return $tt('video.longVideoScriptParsingReviewRetry', { pass: passLabel });
  }
  const key = `video.longVideoParsePhase${phaseKey(phase)}`;
  const label = $tt(key);
  return $tt('video.longVideoScriptParsingPhase', { phase: label });
});
const beatRegeneratingIndex = computed(() =>
  typeof props.beatRegeneratingIndex === 'number' ? props.beatRegeneratingIndex : null,
);

const expandLoading = computed(() => Boolean(props.expanding));
const generatingStoryboard = computed(() => Boolean(props.generatingStoryboard));
const parsing = computed(() => Boolean(props.parsing));
const parseError = computed(() => props.parseError?.trim() || '');
const parseBlocked = computed(() => Boolean(props.parseBlocked));
const parseBlockedReason = computed(() => props.parseBlockedReason?.trim() || '');

const qualityNotices = computed(() => {
  const issues = props.chapterAnalysis?.quality_issues ?? [];
  if (issues.length) {
    return issues.map((i) => i.message);
  }
  return props.chapterAnalysis?.quality_warnings ?? [];
});

const parseButtonLabel = computed(() => $tt('video.longVideoScriptParseUnified'));

const expandHint = computed(() => {
  const trimmed = props.scriptText.trim();
  if (!trimmed || !suggestScriptExpand(trimmed)) return '';
  return $tt('video.longVideoScriptExpandHint');
});

const characters = computed(() => props.characters ?? []);
const styleAnchor = computed(() => props.styleAnchor ?? '');

const totalLookCount = computed(() =>
  characters.value.reduce((sum, ch) => sum + ch.looks.length, 0),
);

const estimatedShots = computed(() => {
  const seg = Math.max(1, props.segmentDurationSec || 5);
  const target = Math.max(seg, props.targetDurationSec || 60);
  return Math.max(2, Math.round(target / seg));
});

const sceneBeats = computed((): LongVideoChapterScene[] =>
  props.chapterAnalysis?.scene_beats ?? [],
);

const chapterSceneCount = computed(() => sceneBeats.value.length);

const estimatedTotalDuration = computed(() => {
  const seg = Math.max(1, props.segmentDurationSec || 5);
  const target = Math.max(seg, props.targetDurationSec || 60);
  const count = chapterSceneCount.value || estimatedShots.value;
  const beatTexts = sceneBeats.value.map((s) => s.beat);
  return allocateShotDurations({
    sceneCount: count,
    targetDurationSec: target,
    defaultSegmentSec: seg,
    beatTexts: beatTexts.length ? beatTexts : undefined,
  }).reduce((sum, sec) => sum + sec, 0);
});

const editorFootStat = computed(() => {
  const trimmed = props.scriptText.trim();
  if (!trimmed) return '';
  const cjkCount = trimmed.match(/[\u4e00-\u9fff]/g)?.length ?? 0;
  if (cjkCount / trimmed.length > 0.25) {
    return $tt('video.longVideoScriptCharCount', { n: trimmed.length });
  }
  const words = trimmed.split(/\s+/).filter(Boolean).length;
  return $tt('video.longVideoScriptWordCount', { n: words });
});

const displaySynopsis = computed(() => {
  const fromAnalysis = props.chapterAnalysis?.synopsis?.trim();
  if (fromAnalysis) return fromAnalysis;
  return '';
});

const synopsisParts = computed(() => {
  const moodFromField = props.chapterAnalysis?.mood?.trim();
  if (moodFromField) {
    return { logline: displaySynopsis.value, mood: moodFromField };
  }
  return splitSynopsisMood(displaySynopsis.value);
});

const shotSizeChoices = computed(() => shotSizeOptions(locale.value));

const insightTabOptions = computed(() => [
  { label: $tt('video.longVideoScriptInsightTabSynopsis'), value: 'synopsis' as ScriptInsightTab },
  {
    label: $tt('video.longVideoScriptInsightTabScenes', { n: sceneBeats.value.length }),
    value: 'scenes' as ScriptInsightTab,
    disabled: sceneBeats.value.length === 0,
  },
  {
    label: $tt('video.longVideoScriptInsightTabCast', { n: characters.value.length }),
    value: 'cast' as ScriptInsightTab,
    disabled: characters.value.length === 0,
  },
]);

watch(
  () => props.scriptParsed,
  (parsed) => {
    if (!parsed) return;
    insightTab.value = sceneBeats.value.length > 0 ? 'scenes' : 'synopsis';
  },
);

function characterInitial(name: string): string {
  const trimmed = name.trim();
  if (!trimmed) return '?';
  return trimmed.slice(0, 1).toUpperCase();
}

function primaryLookBody(ch: LongVideoCharacter): string {
  const look = ch.looks.find((l) => l.id === ch.default_look_id) ?? ch.looks[0];
  return look?.body?.trim() ?? '';
}

function parsedCharacterLook(ch: LongVideoCharacter) {
  return parseCharacterLookBody(primaryLookBody(ch));
}

function shotSizeOptionsForBeat(beat: string): readonly string[] {
  const base = shotSizeChoices.value;
  const current = parseSceneBeat(beat).shotSize;
  if (current && !base.includes(current)) {
    return [current, ...base];
  }
  return base;
}

function sceneBeatIndex(scene: LongVideoChapterScene, idx: number): number {
  return typeof scene.order === 'number' && scene.order >= 0 ? scene.order : idx;
}

function parsedSceneBeat(beat: string) {
  return parseSceneBeat(beat);
}

function updateSceneBeat(index: number, parsed: ParsedSceneBeat) {
  const analysis = props.chapterAnalysis;
  if (!analysis?.scene_beats?.length) return;
  const beat = composeSceneBeat(parsed.shotSize, parsed.location, parsed.visual);
  const scene_beats = analysis.scene_beats.map((s, i) =>
    i === index ? { ...s, beat } : s,
  );
  emit('update:chapterAnalysis', { ...analysis, scene_beats });
}

function onSceneFieldInput(index: number, field: keyof ParsedSceneBeat, value: string | number) {
  const analysis = props.chapterAnalysis;
  if (!analysis?.scene_beats?.length) return;
  const parsed = parseSceneBeat(analysis.scene_beats[index]?.beat ?? '');
  parsed[field] = String(value ?? '');
  updateSceneBeat(index, parsed);
}

function onSceneVisualInput(index: number, event: Event) {
  onSceneFieldInput(index, 'visual', (event.target as HTMLTextAreaElement).value);
}

function onScriptInput(event: Event) {
  emit('update:scriptText', (event.target as HTMLTextAreaElement).value);
}

function onChapterTitleInput(event: Event) {
  emit('update:chapterTitle', (event.target as HTMLInputElement).value);
}

function onDurationChange(value: number | string) {
  const sec = typeof value === 'number' ? value : Number.parseInt(String(value), 10);
  if (Number.isFinite(sec) && sec > 0) {
    emit('update:targetDurationSec', sec);
  }
}

function onSegmentDurationChange(value: number | string) {
  const sec = typeof value === 'number' ? value : Number.parseInt(String(value), 10);
  if (Number.isFinite(sec) && sec > 0) {
    emit('update:segmentDurationSec', sec);
  }
}

function onScriptParseLlmChange(value: number | string) {
  emit('update:scriptParseLlmModel', String(value));
}

const scriptParseLlmModel = computed(() => {
  const saved = props.scriptParseLlmModel?.trim();
  if (saved) return saved;
  const preferred = 'qwen3.6-27b';
  if (llmModelOptions.value.some((o) => o.value === preferred)) return preferred;
  return llmModelOptions.value[0]?.value ?? '';
});

function onSceneTitleInput(index: number, event: Event) {
  const title = (event.target as HTMLInputElement).value;
  const analysis = props.chapterAnalysis;
  if (!analysis?.scene_beats?.length) return;
  const scene_beats = analysis.scene_beats.map((s, i) =>
    i === index ? { ...s, title } : s,
  );
  emit('update:chapterAnalysis', { ...analysis, scene_beats });
}

function triggerFileUpload() {
  fileInputRef.value?.click();
}

async function onFileSelected(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  input.value = '';
  if (!file) return;
  if (file.size > MAX_TXT_BYTES) {
    toast.warning($tt('video.longVideoChapterFileTooLarge'));
    return;
  }
  try {
    const text = await file.text();
    emit('update:scriptText', text);
    if (!props.chapterTitle.trim() && file.name) {
      emit('update:chapterTitle', file.name.replace(/\.txt$/i, ''));
    }
  } catch {
    toast.error($tt('video.longVideoChapterFileReadFailed'));
  }
}
</script>

<style scoped>
.lv-script-studio {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 0;
  padding: 0;
  overflow: hidden;
  width: 100%;
}

.lv-script-studio__body {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  overflow: hidden;
}

.lv-script-studio__body > .lv-script-studio__composer {
  display: flex;
  flex-direction: column;
  min-height: 0;
  padding: 12px 16px 16px;
  border-right: 0.5px solid var(--dq-border-subtle);
}

.lv-script-studio__composer {
  flex: 1;
  min-height: 220px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.lv-script-studio__mode-bar {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px 12px;
  flex-shrink: 0;
  padding: 8px 10px;
  border-radius: 10px;
  background: color-mix(in srgb, var(--dq-fill-control) 40%, transparent);
  border: 0.5px solid var(--dq-border-subtle);
}

.lv-script-studio__mode-label {
  font-size: var(--dq-font-size-caption);
  font-weight: 650;
  color: var(--dq-label-tertiary);
  white-space: nowrap;
}

.lv-script-studio__mode-segmented {
  flex-shrink: 0;
}

.lv-script-studio__mode-hint {
  flex: 1;
  min-width: 160px;
  font-size: var(--dq-font-size-caption);
  line-height: 1.4;
  color: var(--dq-label-tertiary);
}

.lv-script-studio__composer-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-shrink: 0;
}

.lv-script-studio__composer-bar-actions {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
  flex: 1;
  justify-content: flex-end;
}

.lv-script-studio__chapter-title {
  flex: 1;
  max-width: 280px;
  min-width: 120px;
  font-size: var(--dq-font-size-caption);
  padding: 5px 10px;
  border-radius: 8px;
}

.lv-script-studio__composer-label {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  color: var(--dq-label-secondary);
}

.lv-script-studio__textarea {
  flex: 1;
  width: 100%;
  min-height: 220px;
  resize: none;
  font-size: var(--dq-font-size-body);
  line-height: 1.65;
  padding: 14px 16px;
  border-radius: 12px;
  box-sizing: border-box;
  font-family: inherit;
}

.lv-script-studio__textarea--long {
  min-height: 280px;
}

.lv-script-studio__parse-params {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px 14px;
  flex-shrink: 0;
  padding: 10px 12px;
  border-radius: 10px;
  background: color-mix(in srgb, var(--dq-fill-control) 45%, transparent);
  border: 0.5px solid var(--dq-border-subtle);
}

.lv-script-studio__parse-params-label {
  font-size: var(--dq-font-size-caption);
  font-weight: 650;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--dq-label-tertiary);
  white-space: nowrap;
}

.lv-script-studio__parse-duration {
  display: inline-flex;
  align-items: center;
  gap: 8px;
}

.lv-script-studio__parse-duration-label {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
  white-space: nowrap;
}

.lv-script-studio__parse-duration-select {
  min-width: 96px;
}

.lv-script-studio__parse-chip {
  font-size: var(--dq-font-size-caption);
  padding: 3px 9px;
  border-radius: 999px;
  color: var(--dq-label-secondary);
  background: color-mix(in srgb, var(--dq-surface-elevated) 80%, transparent);
  border: 0.5px solid var(--dq-border-subtle);
}

.lv-script-studio__parse-progress {
  display: flex;
  flex-direction: column;
  gap: 8px;
  flex-shrink: 0;
  padding: 10px 12px;
  border-radius: 10px;
  background: color-mix(in srgb, var(--dq-accent) 8%, transparent);
  border: 0.5px solid color-mix(in srgb, var(--dq-accent) 22%, var(--dq-border-subtle));
}

.lv-script-studio__parse-progress-label {
  font-size: var(--dq-font-size-caption);
  font-weight: 650;
  color: var(--dq-label-secondary);
}

.lv-script-studio__parse-step--retry-parent {
  color: var(--dq-color-text-secondary);
}

.lv-script-studio__parse-step--retry-parent .lv-script-studio__parse-step-dot {
  box-shadow: 0 0 0 2px var(--dq-color-warning, #e6a23c);
}

.lv-script-studio__scene-title-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.lv-script-studio__scene-title-row .lv-script-studio__scene-title {
  flex: 1;
  min-width: 0;
}

.lv-script-studio__scene-regen {
  flex-shrink: 0;
}

.lv-script-studio__parse-steps {
  margin: 0;
  padding: 0;
  list-style: none;
  display: flex;
  flex-wrap: wrap;
  gap: 6px 10px;
}

.lv-script-studio__parse-step {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
}

.lv-script-studio__parse-step-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--dq-border-subtle);
  flex-shrink: 0;
}

.lv-script-studio__parse-step--active {
  color: var(--dq-accent);
  font-weight: 650;
}

.lv-script-studio__parse-step--active .lv-script-studio__parse-step-dot {
  background: var(--dq-accent);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--dq-accent) 24%, transparent);
}

.lv-script-studio__parse-step--done {
  color: var(--dq-label-secondary);
}

.lv-script-studio__parse-step--done .lv-script-studio__parse-step-dot {
  background: color-mix(in srgb, var(--dq-accent) 55%, var(--dq-border-subtle));
}

.lv-script-studio__parse-params-note {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
}

.lv-script-studio__composer-foot {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-shrink: 0;
  padding-top: 4px;
}

.lv-script-studio__composer-foot-meta {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  min-width: 0;
}

.lv-script-studio__composer-stat {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  color: var(--dq-label-secondary);
}

.lv-script-studio__composer-expand-hint {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-accent);
}

.lv-script-studio__composer-foot-actions {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
}

.lv-script-studio__next-actions {
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 8px;
  flex-shrink: 0;
}

.lv-script-studio__file-input {
  display: none;
}

.lv-script-studio__insights {
  min-height: 0;
  overflow: hidden;
  padding: 16px 18px 20px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  background: color-mix(in srgb, var(--dq-surface-elevated) 35%, transparent);
}

.lv-script-studio__insights-parsed {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.lv-script-studio__insight-tabs {
  flex-shrink: 0;
}

.lv-script-studio__insight-panel {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.lv-script-studio__empty {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 32px 20px;
  min-height: 280px;
}

.lv-script-studio__empty-icon {
  width: 56px;
  height: 56px;
  margin-bottom: 16px;
  border-radius: 14px;
  background: color-mix(in srgb, var(--dq-accent) 10%, transparent);
  border: 1px solid color-mix(in srgb, var(--dq-accent) 22%, var(--dq-border-subtle));
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%230a84ff' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z'/%3E%3Cpolyline points='14 2 14 8 20 8'/%3E%3Cline x1='16' y1='13' x2='8' y2='13'/%3E%3Cline x1='16' y1='17' x2='8' y2='17'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: center;
  background-size: 26px 26px;
}

.lv-script-studio__empty-title {
  margin: 0 0 8px;
  font-size: var(--dq-font-size-title);
  font-weight: 650;
  color: var(--dq-label-primary);
}

.lv-script-studio__empty-text {
  margin: 0 0 16px;
  font-size: var(--dq-font-size-caption);
  line-height: 1.55;
  color: var(--dq-label-tertiary);
  max-width: 22rem;
}

.lv-script-studio__empty-list {
  margin: 0;
  padding: 0;
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 8px;
  text-align: left;
}

.lv-script-studio__empty-list li {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
  padding-left: 18px;
  position: relative;
}

.lv-script-studio__empty-list li::before {
  content: '';
  position: absolute;
  left: 0;
  top: 0.55em;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--dq-accent);
  opacity: 0.7;
}

.lv-script-studio__card {
  padding: 12px 14px;
  border-radius: 12px;
  background: color-mix(in srgb, var(--dq-surface-base, var(--dq-bg)) 70%, transparent);
  border: 0.5px solid var(--dq-border-subtle);
}

.lv-script-studio__insight-field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.lv-script-studio__insight-field--bordered {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 0.5px solid var(--dq-border-subtle);
}

.lv-script-studio__insight-field-label {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  color: var(--dq-label-tertiary);
}

.lv-script-studio__style-input {
  width: 100%;
}

.lv-script-studio__card--scenes {
  padding-bottom: 10px;
}

.lv-script-studio__card-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
  margin-bottom: 10px;
}

.lv-script-studio__card-title {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  font-weight: 650;
  letter-spacing: 0.02em;
  text-transform: uppercase;
  color: var(--dq-label-secondary);
}

.lv-script-studio__card-meta {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
  white-space: nowrap;
}

.lv-script-studio__parse-blocked-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 8px;
}

.lv-script-studio__parse-blocked {
  margin-bottom: var(--dq-space-sm);
}

.lv-script-studio__parse-error,
.lv-script-studio__parse-quality {
  margin-bottom: var(--dq-space-sm);
}

.lv-script-studio__parse-quality-list {
  margin: 0;
  padding-left: 1.1rem;
  font-size: var(--dq-font-size-sm);
  line-height: 1.45;
}

.lv-script-studio__parse-quality-list li + li {
  margin-top: 0.35rem;
}

.lv-script-studio__synopsis {
  margin: 0;
  font-size: var(--dq-font-size-body);
  line-height: 1.6;
  color: var(--dq-label-primary);
}

.lv-script-studio__synopsis--mood {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
}

.lv-script-studio__scene-list {
  margin: 0;
  padding: 0;
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.lv-script-studio__scene-fields {
  display: grid;
  grid-template-columns: minmax(88px, 0.45fr) minmax(0, 1fr);
  gap: 6px 8px;
}

.lv-script-studio__scene-field {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}

.lv-script-studio__scene-field--wide {
  grid-column: 1 / -1;
}

.lv-script-studio__scene-field--full {
  grid-column: 1 / -1;
}

.lv-script-studio__scene-field-label {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  color: var(--dq-label-tertiary);
}

.lv-script-studio__scene-shot-select {
  width: 100%;
}

.lv-script-studio__scene-visual {
  width: 100%;
  min-height: 44px;
  resize: vertical;
  font-size: var(--dq-font-size-caption);
  line-height: 1.5;
  padding: 8px 10px;
  border-radius: 8px;
}

.lv-script-studio__scene {
  display: flex;
  gap: 10px;
  align-items: flex-start;
}

.lv-script-studio__scene-index {
  flex-shrink: 0;
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 7px;
  font-size: var(--dq-font-size-caption);
  font-weight: 700;
  color: var(--dq-accent);
  background: color-mix(in srgb, var(--dq-accent) 12%, transparent);
  border: 0.5px solid color-mix(in srgb, var(--dq-accent) 28%, transparent);
}

.lv-script-studio__scene-body {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.lv-script-studio__scene-title {
  width: 100%;
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  padding: 4px 8px;
  border-radius: 6px;
}

.lv-script-studio__cast-role {
  display: inline-block;
  margin-top: 6px;
  font-size: var(--dq-font-size-caption);
  font-weight: 650;
  padding: 2px 8px;
  border-radius: 999px;
  color: var(--dq-accent);
  background: color-mix(in srgb, var(--dq-accent) 12%, transparent);
  border: 0.5px solid color-mix(in srgb, var(--dq-accent) 28%, transparent);
}

.lv-script-studio__cast-grid {
  margin: 0;
  padding: 0;
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.lv-script-studio__cast-card {
  display: flex;
  gap: 10px;
  padding: 10px 12px;
  border-radius: 10px;
  background: color-mix(in srgb, var(--dq-fill-control) 55%, transparent);
  border: 0.5px solid var(--dq-border-subtle);
}

.lv-script-studio__cast-avatar {
  flex-shrink: 0;
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 10px;
  font-size: var(--dq-font-size-body);
  font-weight: 700;
  color: var(--dq-accent);
  background: color-mix(in srgb, var(--dq-accent) 14%, transparent);
  border: 0.5px solid color-mix(in srgb, var(--dq-accent) 25%, transparent);
}

.lv-script-studio__cast-info {
  flex: 1;
  min-width: 0;
}

.lv-script-studio__cast-name {
  display: block;
  font-size: var(--dq-font-size-body);
  font-weight: 650;
  color: var(--dq-label-primary);
}

.lv-script-studio__look-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin: 6px 0 0;
  padding: 0;
  list-style: none;
}

.lv-script-studio__look-tag {
  font-size: var(--dq-font-size-caption);
  padding: 2px 7px;
  border-radius: 999px;
  color: var(--dq-label-secondary);
  background: color-mix(in srgb, var(--dq-surface-elevated) 80%, transparent);
  border: 0.5px solid var(--dq-border-subtle);
}

.lv-script-studio__cast-body {
  margin: 6px 0 0;
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
  color: var(--dq-label-tertiary);
}

.lv-script-studio__next {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 14px 16px;
  margin-top: auto;
  border-radius: 12px;
  background: linear-gradient(
    135deg,
    color-mix(in srgb, var(--dq-accent) 16%, transparent) 0%,
    color-mix(in srgb, var(--dq-accent) 6%, transparent) 100%
  );
  border: 1px solid color-mix(in srgb, var(--dq-accent) 30%, var(--dq-border-subtle));
}

.lv-script-studio__next-kicker {
  display: block;
  font-size: var(--dq-font-size-caption);
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--dq-accent);
  margin-bottom: 4px;
}

.lv-script-studio__next-text {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
  color: var(--dq-label-secondary);
}

@media (max-width: 960px) {
  .lv-script-studio__composer-foot {
    flex-wrap: wrap;
  }

  .lv-script-studio__body {
    grid-template-columns: 1fr;
    overflow-y: auto;
  }

  .lv-script-studio__body > .lv-script-studio__composer {
    border-right: none;
    border-bottom: 0.5px solid var(--dq-border-subtle);
    min-height: 360px;
  }

  .lv-script-studio__insights {
    min-height: 280px;
  }
}
</style>
