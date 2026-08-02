<template>
  <div class="copilot-page">
    <!-- Left: task navigation (models / settings pattern) -->
    <div class="copilot-page__sidebar">
      <DqSurfaceCard class="copilot-page__sidebar-card studio-surface-card">
        <div class="card-title">
          <DqIcon><Bot /></DqIcon>
          {{ $t('assistant.title') }}
        </div>
        <p class="copilot-page__sidebar-intro">{{ $t('assistant.pageSubtitle') }}</p>

        <nav class="copilot-page__nav" role="navigation" :aria-label="$t('assistant.title')">
          <div
            v-for="section in navSections"
            :key="section.id"
            class="copilot-page__nav-section"
          >
            <div class="copilot-page__nav-label">{{ section.label }}</div>
            <button
              v-for="item in section.items"
              :key="item.key"
              type="button"
              class="dq-download-menu__item"
              :class="{ 'is-active': activeNavKey === item.key }"
              @click="selectNav(item)"
            >
              <DqIcon class="dq-download-menu__icon"><component :is="item.icon" /></DqIcon>
              <span class="dq-download-menu__label">{{ item.label }}</span>
            </button>
          </div>
        </nav>

        <div v-if="modelInfo" class="copilot-page__status-panel">
          <DqTag size="sm" :type="llmReady ? 'success' : 'warning'" effect="plain">
            {{ llmReady ? $t('assistant.llmReady') : $t('assistant.modelNotReady') }}
          </DqTag>
          <span v-if="llmReady" class="copilot-page__status-model">{{ chatModelLabel }}</span>
          <DqButton v-else type="text" size="sm" @click="goToModels">
            {{ $t('assistant.installLlm') }}
          </DqButton>
          <p v-if="needsVision && visionReady" class="copilot-page__status-vision">
            {{ $t('assistant.visionUsed') }} · {{ visionModelLabel }}
          </p>
          <p v-else-if="needsVision && !visionReady" class="copilot-page__status-vision copilot-page__status-vision--warn">
            {{ visionWarnText }}
            <DqButton type="text" size="sm" @click="goToModels">{{ $t('assistant.installVisionModel') }}</DqButton>
          </p>
        </div>
      </DqSurfaceCard>
    </div>

    <!-- Right: workspace + task cards -->
    <div class="copilot-page__main">
      <header class="page-header copilot-page__page-header">
        <div class="copilot-page__header-main">
          <h2 class="page-title copilot-page__page-title">
            <DqIcon v-if="currentTaskIcon" class="copilot-page__title-icon" aria-hidden="true">
              <component :is="currentTaskIcon" />
            </DqIcon>
            {{ currentTaskTitle }}
          </h2>
          <p class="copilot-page__page-desc">{{ currentTaskDesc }}</p>
        </div>
        <div class="copilot-page__header-actions">
          <DqButton size="sm" type="default" @click="openComposer">
            {{ openComposerLabel }}
          </DqButton>
          <DqCountBadge
            :value="pendingCount"
            :hidden="!cardsPaneCollapsed || pendingCount === 0"
          >
            <DqIconButton
              type="text"
              size="sm"
              :label="cardsPaneCollapsed ? $t('assistant.expandCards') : $t('assistant.collapseCards')"
              class="copilot-page__collapse-btn"
              @click="toggleCardsPane"
            >
              <DqIcon :size="16"><Menu /></DqIcon>
            </DqIconButton>
          </DqCountBadge>
        </div>
      </header>

      <div v-if="!llmReady" class="copilot-page__banner">
        <p>{{ $t('assistant.llmRequiredHint') }}</p>
        <DqButton size="sm" @click="goToModels">{{ $t('assistant.installLlm') }}</DqButton>
      </div>

      <div class="copilot-page__split">
      <DqSurfaceCard class="copilot-page__workspace-card studio-surface-card">
        <div
          v-if="needsAsset"
          class="copilot-page__asset"
        >
          <label class="copilot-page__label">{{ $t('assistant.referenceAsset') }}</label>
          <div v-if="selectedAsset" class="copilot-page__asset-selected">
            <img
              v-if="selectedAsset.preview"
              :src="selectedAsset.preview"
              alt=""
              class="copilot-page__asset-preview"
            />
            <div class="copilot-page__asset-meta">
              <span class="copilot-page__asset-id">{{ selectedAsset.id }}</span>
              <DqButton type="text" size="sm" @click="clearAsset">{{ $t('assistant.clearAsset') }}</DqButton>
            </div>
          </div>
          <AssetPicker
            :accept-kind="media === 'video' ? 'video' : 'image'"
            :recent-gallery="recentGallery"
            @pick="onAssetPick"
          />
        </div>

        <div v-if="showTextInput" class="copilot-page__field">
          <label class="copilot-page__label">{{ inputLabel }}</label>
          <DqInput
            v-model="draftInput"
            type="textarea"
            :rows="taskKind === 'generate_lyrics' ? 4 : 5"
            resize="none"
            :placeholder="inputPlaceholder"
            :disabled="!llmReady || isRunning"
            @keydown.meta.enter.prevent="onRunShortcut"
            @keydown.ctrl.enter.prevent="onRunShortcut"
          />
        </div>

        <div v-if="taskKind === 'analyze_reference' && analyzePresets.length" class="copilot-page__presets">
          <span class="copilot-page__presets-label">{{ $t('assistant.analyzePresets') }}</span>
          <div class="copilot-page__preset-list">
            <button
              v-for="preset in analyzePresets"
              :key="preset.id"
              type="button"
              class="copilot-page__preset"
              :disabled="!llmReady || isRunning"
              @click="draftInput = preset.text"
            >
              {{ preset.label }}
            </button>
          </div>
        </div>

        <div class="copilot-page__actions">
          <DqButton
            :disabled="!canRunNow || isRunning"
            :loading="isRunning"
            @click="runNow"
          >
            {{ $t('assistant.runNow') }}
          </DqButton>
          <span class="copilot-page__shortcut-hint">{{ runShortcutHint }}</span>
          <DqButton
            type="default"
            :disabled="!canQueue || isRunning"
            @click="queueTask"
          >
            {{ $t('assistant.addTaskCard') }}
          </DqButton>
        </div>

        <div v-if="lastResult" class="copilot-page__result">
          <div class="copilot-page__result-head">
            <span class="copilot-page__label">{{ $t('assistant.resultLabel') }}</span>
            <DqTag v-if="lastResult.visionUsed" size="sm" type="info" effect="plain">
              {{ $t('assistant.visionUsed') }}
            </DqTag>
          </div>
          <div class="copilot-page__result-body">{{ lastResult.output }}</div>
          <div class="copilot-page__result-actions">
            <DqButton size="sm" @click="copyText(lastResult.output)">{{ $t('assistant.copy') }}</DqButton>
            <DqButton
              v-if="lastResult.applyRoute"
              size="sm"
              type="primary"
              @click="applyResult(lastResult, 'replace')"
            >
              {{ lastResult.applyLabel }}
            </DqButton>
            <DqButton
              v-if="lastResult.applyRoute && lastResult.appendLabel"
              size="sm"
              @click="applyResult(lastResult, 'append')"
            >
              {{ lastResult.appendLabel }}
            </DqButton>
          </div>
        </div>
      </DqSurfaceCard>

      <aside
        v-show="!cardsPaneCollapsed"
        class="copilot-page__cards-pane"
      >
      <DqSurfaceCard class="copilot-page__cards-card studio-surface-card">
        <header class="copilot-page__cards-head">
          <div>
            <h3 class="copilot-page__cards-title">{{ $t('assistant.taskCardsTitle') }}</h3>
            <p class="copilot-page__cards-desc">{{ $t('assistant.taskCardsDesc') }}</p>
          </div>
          <div class="copilot-page__cards-actions">
            <DqButton
              size="sm"
              :disabled="pendingCount === 0 || isBatchRunning"
              :loading="isBatchRunning"
              @click="runBatch"
            >
              {{ $t('assistant.runAll', { n: pendingCount }) }}
            </DqButton>
            <DqButton
              size="sm"
              type="text"
              :disabled="finishedCount === 0"
              @click="clearFinished"
            >
              {{ $t('assistant.clearDone') }}
            </DqButton>
          </div>
        </header>

        <DqEmpty v-if="tasks.length === 0" :description="$t('assistant.taskCardsEmpty')" />

        <ul v-else class="copilot-page__card-list">
          <li
            v-for="task in tasks"
            :key="task.id"
            class="copilot-page__card"
            :class="[
              `copilot-page__card--${task.status}`,
              { 'copilot-page__card--focused': focusedTaskId === task.id },
            ]"
            role="button"
            tabindex="0"
            @click="focusTaskCard(task)"
            @keydown.enter.prevent="focusTaskCard(task)"
          >
            <div class="copilot-page__card-main">
              <span class="copilot-page__card-title">{{ task.title }}</span>
              <span class="copilot-page__card-input">{{ taskCardPreview(task) }}</span>
              <p v-if="task.status === 'error'" class="copilot-page__card-error">{{ task.error }}</p>
              <p v-else-if="task.status === 'done' && task.output" class="copilot-page__card-output">
                {{ task.output }}
              </p>
            </div>
            <div class="copilot-page__card-tools" @click.stop>
              <DqTag size="sm" effect="plain" :type="taskStatusType(task.status)">
                {{ taskStatusLabel(task.status) }}
              </DqTag>
              <DqIconButton
                v-if="task.status === 'pending'"
                type="text"
                size="sm"
                :label="$t('assistant.runNow')"
                @click="executeTask(task.id)"
              >
                <DqIcon :size="14"><VideoPlay /></DqIcon>
              </DqIconButton>
              <DqIconButton
                v-if="task.status === 'done' && task.output"
                type="text"
                size="sm"
                :label="$t('assistant.copy')"
                @click="copyText(task.output!)"
              >
                <DqIcon :size="14"><CopyDocument /></DqIcon>
              </DqIconButton>
              <DqIconButton
                v-if="task.status === 'done' && taskSupportsAppend(task)"
                type="text"
                size="sm"
                :label="appendLabelForTask(task)"
                @click="applyTaskResult(task, 'append')"
              >
                <DqIcon :size="14"><Plus /></DqIcon>
              </DqIconButton>
              <DqIconButton
                v-if="task.status === 'done'"
                type="text"
                size="sm"
                :label="applyLabelForTask(task)"
                @click="applyTaskResult(task, 'replace')"
              >
                <DqIcon :size="14"><ArrowRight /></DqIcon>
              </DqIconButton>
              <DqIconButton
                type="text"
                size="sm"
                :label="$t('common.delete')"
                @click="removeTask(task.id)"
              >
                <DqIcon :size="14"><Delete /></DqIcon>
              </DqIconButton>
            </div>
          </li>
        </ul>
      </DqSurfaceCard>
      </aside>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, nextTick, type Component } from 'vue';
import { useRouter } from 'vue-router';
import { useI18n } from 'vue-i18n';
import {
  ArrowRight,
  CopyDocument,
  Delete,
  DqSurfaceCard,
  Microphone,
  Picture,
  Plus,
  Search,
  VideoCamera,
  VideoPlay,
} from '@danqing/dq-shell';
import AssetPicker from '@/components/asset/AssetPicker.vue';
import {
  parseAssetId,
  useCreativeAssistant,
  type CreativeMedia,
  type CreativeTask,
  type CreativeTaskKind,
  type CreativeTaskStatus,
} from '@/composables/useCreativeAssistant';
import { api } from '@/utils/api';
import { toast } from '@/utils/feedback';
import { getLocale } from '@/utils/i18n';
import { consumeCopilotHandoff } from '@/utils/copilotHandoff';
import { DQ_STORAGE, getItem, setItem } from '@/utils/storage';
import { setPromptDraft } from '@/utils/promptApply';

interface LLMModelInfo {
  model_id: string;
  name: string | { zh?: string; en?: string };
  available: boolean;
  vision?: {
    model_id: string;
    name: string | { zh?: string; en?: string };
    available: boolean;
    mlx_vlm_installed?: boolean;
  };
}

interface SelectedAsset {
  id: string;
  path: string;
  preview: string;
}

interface LastResult {
  output: string;
  visionUsed?: boolean;
  kind?: CreativeTaskKind;
  applyRoute?: 'image_create' | 'video_create' | 'audio_create';
  applyLabel?: string;
  appendLabel?: string;
  musicBrief?: string;
  lyricsOnly?: boolean;
}

interface CopilotNavItem {
  key: string;
  media: CreativeMedia;
  kind: CreativeTaskKind;
  label: string;
  icon: Component;
}

const router = useRouter();
const { t } = useI18n();

const {
  tasks,
  addTask,
  removeTask,
  clearFinished,
  executeTask,
  runBatch,
  isBatchRunning,
  runCreativeTask,
} = useCreativeAssistant();

const media = ref<CreativeMedia>('image');
const taskKind = ref<CreativeTaskKind>('image_to_prompt');
const draftInput = ref('');
const selectedAsset = ref<SelectedAsset | null>(null);
const isRunning = ref(false);
const modelInfo = ref<LLMModelInfo | null>(null);
const recentGallery = ref<Array<Record<string, unknown>>>([]);
const lastResult = ref<LastResult | null>(null);
const focusedTaskId = ref<string | null>(null);
const cardsPaneCollapsed = ref(getItem(DQ_STORAGE.COPILOT_CARDS_COLLAPSED) === '1');

const navSections = computed(() => [
  {
    id: 'image',
    label: t('assistant.mediaImage'),
    items: [
      {
        key: 'image:image_to_prompt',
        media: 'image' as const,
        kind: 'image_to_prompt' as const,
        label: t('assistant.taskImageToPrompt'),
        icon: Picture,
      },
      {
        key: 'image:analyze_reference',
        media: 'image' as const,
        kind: 'analyze_reference' as const,
        label: t('assistant.taskAnalyzeRef'),
        icon: Search,
      },
    ],
  },
  {
    id: 'video',
    label: t('assistant.mediaVideo'),
    items: [
      {
        key: 'video:image_to_prompt',
        media: 'video' as const,
        kind: 'image_to_prompt' as const,
        label: t('assistant.taskFrameToPrompt'),
        icon: VideoCamera,
      },
    ],
  },
  {
    id: 'audio',
    label: t('assistant.mediaAudio'),
    items: [
      {
        key: 'audio:generate_lyrics',
        media: 'audio' as const,
        kind: 'generate_lyrics' as const,
        label: t('assistant.taskLyrics'),
        icon: Microphone,
      },
    ],
  },
]);

const activeNavKey = computed(() => `${media.value}:${taskKind.value}`);

const currentTaskIcon = computed(() => {
  for (const section of navSections.value) {
    const hit = section.items.find((x) => x.key === activeNavKey.value);
    if (hit) return hit.icon;
  }
  return Picture;
});

const openComposerLabel = computed(() => {
  if (media.value === 'video') return t('assistant.openVideoStudio');
  if (media.value === 'audio') return t('assistant.openAudioStudio');
  return t('assistant.openImageStudio');
});

const runShortcutHint = computed(() => {
  const isMac = typeof navigator !== 'undefined'
    && navigator.platform.toLowerCase().includes('mac');
  return isMac ? t('assistant.runShortcutHint') : t('assistant.runShortcutHintWin');
});

function selectNav(item: CopilotNavItem) {
  media.value = item.media;
  taskKind.value = item.kind;
  focusedTaskId.value = null;
  lastResult.value = null;
  setItem(DQ_STORAGE.COPILOT_NAV, item.key);
}

function restoreNavFromStorage() {
  const raw = getItem(DQ_STORAGE.COPILOT_NAV);
  if (!raw) return;
  for (const section of navSections.value) {
    const hit = section.items.find((x) => x.key === raw);
    if (hit) {
      media.value = hit.media;
      taskKind.value = hit.kind;
      return;
    }
  }
}

let handoffApplying = false;

watch([media, taskKind], () => {
  if (handoffApplying) return;
  lastResult.value = null;
});

const llmReady = computed(() => Boolean(modelInfo.value?.available));
const visionInfo = computed(() => modelInfo.value?.vision ?? null);
const visionReady = computed(() => Boolean(visionInfo.value?.available));

const needsAsset = computed(
  () => taskKind.value === 'image_to_prompt' || taskKind.value === 'analyze_reference',
);
const needsVision = computed(() => needsAsset.value);
const showTextInput = computed(
  () => taskKind.value !== 'image_to_prompt',
);

const pendingCount = computed(() => tasks.value.filter((x) => x.status === 'pending').length);
const finishedCount = computed(
  () => tasks.value.filter((x) => x.status === 'done' || x.status === 'error').length,
);

function resolveModelLabel(
  name: string | { zh?: string; en?: string } | undefined,
  fallback: string,
): string {
  if (!name) return fallback;
  if (typeof name === 'string') return name;
  const locale = getLocale();
  return (locale === 'zh' ? name.zh : name.en) || name.zh || name.en || fallback;
}

const chatModelLabel = computed(() =>
  resolveModelLabel(modelInfo.value?.name, modelInfo.value?.model_id || ''),
);

const visionModelLabel = computed(() =>
  resolveModelLabel(visionInfo.value?.name, visionInfo.value?.model_id || ''),
);

const visionWarnText = computed(() => {
  if (!visionInfo.value) return '';
  if (!visionInfo.value.mlx_vlm_installed) return t('assistant.visionPkgMissing');
  return t('assistant.visionModelRequired');
});

const currentTaskTitle = computed(() => {
  const key = `assistant.taskTitle.${taskKind.value}` as const;
  return t(key);
});

const currentTaskDesc = computed(() => {
  if (taskKind.value === 'image_to_prompt' && media.value === 'video') {
    return t('assistant.desc.image_to_prompt_video');
  }
  const key = `assistant.desc.${taskKind.value}` as const;
  return t(key);
});

const inputLabel = computed(() => {
  const map: Record<CreativeTaskKind, string> = {
    enhance_image: t('assistant.enhanceInputLabel'),
    enhance_video: t('assistant.videoInputLabel'),
    enhance_music_brief: t('assistant.musicBriefLabel'),
    generate_lyrics: t('assistant.lyricsInputLabel'),
    analyze_reference: t('assistant.analyzeQuestionLabel'),
    image_to_prompt: '',
  };
  return map[taskKind.value];
});

const inputPlaceholder = computed(() => {
  const map: Record<CreativeTaskKind, string> = {
    enhance_image: t('assistant.enhancePlaceholder'),
    enhance_video: t('assistant.videoPlaceholder'),
    enhance_music_brief: t('assistant.musicBriefPlaceholder'),
    generate_lyrics: t('assistant.lyricsPlaceholder'),
    analyze_reference: t('assistant.analyzePlaceholder'),
    image_to_prompt: '',
  };
  return map[taskKind.value];
});

const analyzePresets = computed(() => [
  { id: 'style', label: t('assistant.presetStyle'), text: t('assistant.presetStyleText') },
  { id: 'palette', label: t('assistant.presetPalette'), text: t('assistant.presetPaletteText') },
  { id: 'subject', label: t('assistant.presetSubject'), text: t('assistant.presetSubjectText') },
  { id: 'prompt', label: t('assistant.presetKeywords'), text: t('assistant.presetKeywordsText') },
]);

const canRunNow = computed(() => {
  if (!llmReady.value) return false;
  if (needsVision.value && !visionReady.value) return false;
  if (needsAsset.value && !selectedAsset.value) return false;
  if (showTextInput.value && !draftInput.value.trim()) return false;
  return true;
});

const canQueue = computed(() => canRunNow.value);

function buildTaskPayload(): Omit<CreativeTask, 'id' | 'status'> {
  return {
    media: media.value,
    kind: taskKind.value,
    title: t(`assistant.taskTitle.${taskKind.value}`),
    input: draftInput.value.trim(),
    assetId: selectedAsset.value?.id,
    assetPreview: selectedAsset.value?.preview,
  };
}

function appendLabelForRoute(route: LastResult['applyRoute']): string | undefined {
  if (route === 'image_create') return t('assistant.appendToImage');
  if (route === 'video_create') return t('assistant.appendToVideo');
  if (route === 'audio_create') return t('assistant.appendToAudio');
  return undefined;
}

function applyMetaForKind(
  kind: CreativeTaskKind,
  output: string,
  input: string,
  taskMedia: CreativeMedia = media.value,
): LastResult {
  const base: LastResult = { output, kind };
  if (kind === 'image_to_prompt') {
    const isVideo = taskMedia === 'video';
    const applyRoute = isVideo ? 'video_create' : 'image_create';
    return {
      ...base,
      applyRoute,
      applyLabel: isVideo ? t('assistant.useInVideo') : t('assistant.useInImage'),
      appendLabel: appendLabelForRoute(applyRoute),
    };
  }
  if (kind === 'enhance_image') {
    return {
      ...base,
      applyRoute: 'image_create',
      applyLabel: t('assistant.useInImage'),
      appendLabel: appendLabelForRoute('image_create'),
    };
  }
  if (kind === 'enhance_video') {
    return {
      ...base,
      applyRoute: 'video_create',
      applyLabel: t('assistant.useInVideo'),
      appendLabel: appendLabelForRoute('video_create'),
    };
  }
  if (kind === 'generate_lyrics') {
    return {
      ...base,
      applyRoute: 'audio_create',
      applyLabel: t('assistant.useInAudio'),
      appendLabel: appendLabelForRoute('audio_create'),
      musicBrief: input,
      lyricsOnly: false,
    };
  }
  if (kind === 'enhance_music_brief') {
    return {
      ...base,
      applyRoute: 'audio_create',
      applyLabel: t('assistant.useInAudio'),
      appendLabel: appendLabelForRoute('audio_create'),
      musicBrief: output,
      lyricsOnly: false,
    };
  }
  if (kind === 'analyze_reference') {
    return {
      ...base,
      applyRoute: 'image_create',
      applyLabel: t('assistant.useInImage'),
      appendLabel: appendLabelForRoute('image_create'),
    };
  }
  return base;
}

async function runNow() {
  if (!canRunNow.value || isRunning.value) return;
  isRunning.value = true;
  lastResult.value = null;
  try {
    const payload = buildTaskPayload();
    const result = await runCreativeTask({
      ...payload,
      id: 'inline',
      status: 'running',
    });
    if (result.status === 'error') {
      toast.error(result.error || t('assistant.error', { msg: '' }));
      return;
    }
    lastResult.value = applyMetaForKind(
      payload.kind,
      result.output || '',
      payload.input,
      payload.media,
    );
    if (result.visionUsed) {
      lastResult.value.visionUsed = true;
    }
  } finally {
    isRunning.value = false;
  }
}

function queueTask() {
  if (!canQueue.value) return;
  const row = addTask(buildTaskPayload());
  focusedTaskId.value = row.id;
  toast.success(t('assistant.taskQueued'));
}

function toggleCardsPane() {
  cardsPaneCollapsed.value = !cardsPaneCollapsed.value;
  setItem(DQ_STORAGE.COPILOT_CARDS_COLLAPSED, cardsPaneCollapsed.value ? '1' : '0');
}

function openComposer() {
  const name = media.value === 'video'
    ? 'video_create'
    : media.value === 'audio'
      ? 'audio_create'
      : 'image_create';
  void router.push({ name });
}

function onRunShortcut() {
  if (!canRunNow.value || isRunning.value) return;
  void runNow();
}

function focusTaskCard(task: CreativeTask) {
  focusedTaskId.value = task.id;
  handoffApplying = true;
  try {
    media.value = task.media;
    taskKind.value = task.kind;
    draftInput.value = task.input;
    setItem(DQ_STORAGE.COPILOT_NAV, `${task.media}:${task.kind}`);
    if (task.assetId) {
      selectedAsset.value = {
        id: task.assetId,
        path: `asset:${task.assetId}`,
        preview: task.assetPreview || `/api/assets/${task.assetId}/thumbnail`,
      };
    } else {
      selectedAsset.value = null;
    }
    if (task.status === 'done' && task.output) {
      const meta = applyMetaForKind(task.kind, task.output, task.input, task.media);
      lastResult.value = task.visionUsed ? { ...meta, visionUsed: true } : meta;
    } else {
      lastResult.value = null;
    }
  } finally {
    handoffApplying = false;
  }
}

function taskCardPreview(task: CreativeTask): string {
  if (task.assetId) return task.input || task.assetId;
  return task.input;
}

function taskStatusType(status: CreativeTaskStatus): 'info' | 'success' | 'warning' | 'danger' {
  if (status === 'done') return 'success';
  if (status === 'error') return 'danger';
  if (status === 'running') return 'info';
  return 'warning';
}

function taskStatusLabel(status: CreativeTaskStatus): string {
  return t(`assistant.taskStatus.${status}`);
}

function applyLabelForTask(task: CreativeTask): string {
  if (task.media === 'video') return t('assistant.useInVideo');
  if (task.media === 'audio') return t('assistant.useInAudio');
  return t('assistant.useInImage');
}

function taskSupportsAppend(task: CreativeTask): boolean {
  return !!task.output && [
    'enhance_image',
    'enhance_video',
    'enhance_music_brief',
    'image_to_prompt',
    'analyze_reference',
    'generate_lyrics',
  ].includes(task.kind);
}

function appendLabelForTask(task: CreativeTask): string {
  if (task.media === 'video') return t('assistant.appendToVideo');
  if (task.media === 'audio') return t('assistant.appendToAudio');
  return t('assistant.appendToImage');
}

function applyTaskResult(task: CreativeTask, mode: 'replace' | 'append' = 'replace') {
  if (!task.output) return;
  const meta = applyMetaForKind(task.kind, task.output, task.input, task.media);
  applyResult(meta, mode);
}

function applyResult(meta: LastResult, mode: 'replace' | 'append' = 'replace') {
  if (!meta.output?.trim()) return;
  const text = meta.output.trim();
  if (meta.applyRoute === 'image_create') {
    setPromptDraft(DQ_STORAGE.IMAGE_CREATE_PROMPT_DRAFT, text, mode);
    void router.push({ name: 'image_create' });
    return;
  }
  if (meta.applyRoute === 'video_create') {
    setPromptDraft(DQ_STORAGE.VIDEO_CREATE_PROMPT_DRAFT, text, mode);
    void router.push({ name: 'video_create' });
    return;
  }
  if (meta.applyRoute === 'audio_create') {
    if (meta.kind === 'enhance_music_brief' && meta.musicBrief?.trim()) {
      setPromptDraft(DQ_STORAGE.AUDIO_CREATE_PROMPT_DRAFT, meta.musicBrief.trim(), mode);
    } else if (meta.kind === 'generate_lyrics') {
      setPromptDraft(DQ_STORAGE.AUDIO_CREATE_LYRICS_DRAFT, text, mode);
    } else if (!meta.lyricsOnly) {
      setPromptDraft(DQ_STORAGE.AUDIO_CREATE_LYRICS_DRAFT, text, mode);
    }
    void router.push({ name: 'audio_create' });
  }
}

async function copyText(text: string) {
  try {
    await navigator.clipboard.writeText(text);
    toast.success(t('assistant.copied'));
  } catch {
    toast.error(t('assistant.copyFailed'));
  }
}

function onAssetPick(payload: { path: string; previewUrl: string }) {
  const id = parseAssetId(payload.path);
  if (!id) return;
  selectedAsset.value = {
    id,
    path: payload.path,
    preview: payload.previewUrl || `/api/assets/${id}/thumbnail`,
  };
}

function clearAsset() {
  selectedAsset.value = null;
}

function goToModels() {
  setItem(DQ_STORAGE.MODELS_CATEGORY, 'llm_models');
  void router.push({ name: 'models' });
}

function applyHandoff(): { autoRun: boolean } | null {
  const handoff = consumeCopilotHandoff();
  if (!handoff) return null;
  handoffApplying = true;
  try {
    media.value = handoff.media;
    taskKind.value = handoff.task;
    if (handoff.prompt) {
      draftInput.value = handoff.prompt;
    }
    if (handoff.assetId) {
      selectedAsset.value = {
        id: handoff.assetId,
        path: `asset:${handoff.assetId}`,
        preview: handoff.assetPreview || `/api/assets/${handoff.assetId}/thumbnail`,
      };
    }
    setItem(DQ_STORAGE.COPILOT_NAV, `${handoff.media}:${handoff.task}`);
    const autoRun =
      (handoff.task === 'image_to_prompt' && Boolean(handoff.assetId))
      || (handoff.task === 'analyze_reference'
        && Boolean(handoff.assetId)
        && Boolean(handoff.prompt?.trim()));
    return { autoRun };
  } finally {
    handoffApplying = false;
  }
}

async function loadRecentGallery() {
  try {
    const kind = media.value === 'video' ? 'video' : 'image';
    const rows = await api.gallery.listImages(16, 0, {
      kind,
      exclude_step_previews: true,
      exclude_upload_refs: true,
    });
    recentGallery.value = rows as unknown as Array<Record<string, unknown>>;
  } catch {
    recentGallery.value = [];
  }
}

watch(media, () => {
  void loadRecentGallery();
});

onMounted(async () => {
  try {
    const [llm, vision] = await Promise.all([
      api.gen.getLlmModelInfo(),
      api.gen.getVisionModelInfo(),
    ]);
    modelInfo.value = { ...llm, vision };
  } catch {
    // ignore
  }
  await loadRecentGallery();
  const handoff = applyHandoff();
  if (!handoff) {
    restoreNavFromStorage();
  }
  if (handoff?.autoRun) {
    await nextTick();
    if (canRunNow.value) {
      void runNow();
    }
  }
});
</script>

<style scoped>
.copilot-page {
  display: flex;
  gap: 20px;
  width: 100%;
  height: 100%;
  overflow: hidden;
}

.copilot-page__sidebar {
  width: 220px;
  flex-shrink: 0;
}

.copilot-page__sidebar-card {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.copilot-page__sidebar-card.dq-surface-card > :deep(.dq-surface-card__body) {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.copilot-page__sidebar-intro {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
  margin: 0 0 12px;
  line-height: 1.45;
}

.copilot-page__nav {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.copilot-page__nav-section {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.copilot-page__nav-label {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  letter-spacing: var(--dq-tracking-wider);
  text-transform: uppercase;
  color: var(--dq-label-tertiary);
  padding: 0 8px 4px;
}

.copilot-page__status-panel {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 0.5px solid var(--dq-border-subtle);
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 6px;
  flex-shrink: 0;
}

.copilot-page__status-model {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
  line-height: 1.35;
  word-break: break-word;
}

.copilot-page__status-vision {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
  line-height: 1.4;
}

.copilot-page__status-vision--warn {
  color: var(--dq-label-secondary);
}

.copilot-page__main {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  padding: 4px 8px 16px 0;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.copilot-page__split {
  flex: 1;
  min-height: 0;
  display: flex;
  gap: 16px;
  align-items: stretch;
}

.copilot-page__cards-pane {
  width: min(360px, 34vw);
  flex-shrink: 0;
  min-height: 0;
  display: flex;
  flex-direction: column;
}

.copilot-page__header-main {
  min-width: 0;
  flex: 1 1 280px;
}

.copilot-page__header-actions {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-shrink: 0;
  margin-left: auto;
}

.copilot-page__title-icon {
  color: var(--dq-accent);
  flex-shrink: 0;
}

.copilot-page__page-desc {
  margin: 0;
  font-size: var(--dq-font-size-body);
  line-height: 1.5;
  color: var(--dq-label-tertiary);
  max-width: 720px;
}

.copilot-page__shortcut-hint {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
  align-self: center;
}

.copilot-page__banner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 10px 12px;
  border-radius: 10px;
  background: var(--dq-warning-surface);
  border: 0.5px solid var(--dq-warning-surface-border);
  flex-shrink: 0;
}

.copilot-page__banner p {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
}

.copilot-page__workspace-card {
  flex: 1;
  min-width: 0;
  min-height: 0;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.copilot-page__cards-card {
  flex: 1;
  min-height: 0;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.copilot-page__workspace-card :deep(.dq-surface-card__body) {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.copilot-page__cards-card :deep(.dq-surface-card__body) {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.copilot-page__label {
  font-size: var(--dq-font-size-caption);
  font-weight: 500;
  color: var(--dq-label-secondary);
}

.copilot-page__field {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.copilot-page__asset {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.copilot-page__asset-selected {
  display: flex;
  gap: 12px;
  align-items: flex-start;
}

.copilot-page__asset-preview {
  width: min(200px, 42%);
  max-height: 160px;
  object-fit: contain;
  border-radius: 10px;
  border: 0.5px solid var(--dq-border);
  background: var(--dq-bg-base);
  flex-shrink: 0;
}

.copilot-page__asset-meta {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 6px;
  min-width: 0;
  flex: 1;
}

.copilot-page__asset-id {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
  word-break: break-all;
}

.copilot-page__presets-label {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
}

.copilot-page__preset-list {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 6px;
}

.copilot-page__preset {
  padding: 5px 10px;
  border-radius: 999px;
  border: 0.5px solid var(--dq-border);
  background: transparent;
  color: var(--dq-label-secondary);
  font-size: var(--dq-font-size-caption);
  cursor: pointer;
}

.copilot-page__preset:hover:not(:disabled) {
  border-color: var(--dq-accent-surface-border);
  color: var(--dq-label-primary);
}

.copilot-page__preset:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.copilot-page__actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.copilot-page__result {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding-top: 4px;
  border-top: 0.5px solid var(--dq-border-subtle);
}

.copilot-page__result-head {
  display: flex;
  align-items: center;
  gap: 8px;
}

.copilot-page__result-body {
  padding: 12px;
  border-radius: 10px;
  background: var(--dq-surface-inset);
  border: 0.5px solid var(--dq-glass-border);
  font-size: var(--dq-font-size-body);
  line-height: 1.55;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: min(40vh, 360px);
  overflow-y: auto;
}

.copilot-page__result-actions {
  display: flex;
  gap: 8px;
}

.copilot-page__cards-head {
  display: flex;
  flex-direction: column;
  gap: 10px;
  flex-shrink: 0;
}

.copilot-page__cards-actions {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}

.copilot-page__cards-title {
  margin: 0 0 2px;
  font-size: var(--dq-font-size-body);
  font-weight: 600;
  color: var(--dq-label-primary);
}

.copilot-page__cards-desc {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
}

.copilot-page__card-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.copilot-page__card {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 10px 12px;
  border-radius: 10px;
  border: 0.5px solid var(--dq-glass-border);
  background: var(--dq-surface-inset);
  -webkit-backdrop-filter: var(--dq-glass-blur-light);
  backdrop-filter: var(--dq-glass-blur-light);
}

.copilot-page__card--running {
  border-color: var(--dq-accent-surface-border);
}

.copilot-page__card--done {
  border-color: var(--dq-success-surface-border);
}

.copilot-page__card--error {
  border-color: var(--dq-danger-surface-border);
}

.copilot-page__card--focused {
  border-color: var(--dq-accent-surface-border);
  background: color-mix(in srgb, var(--dq-accent) 8%, var(--dq-surface-inset));
}

.copilot-page__card[role='button'] {
  cursor: pointer;
}

.copilot-page__card[role='button']:hover {
  border-color: var(--dq-accent-surface-border);
}

.copilot-page__card-main {
  min-width: 0;
  flex: 1;
}

.copilot-page__card-title {
  display: block;
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  color: var(--dq-label-primary);
  margin-bottom: 4px;
}

.copilot-page__card-input,
.copilot-page__card-output {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
  color: var(--dq-label-secondary);
  overflow: hidden;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
}

.copilot-page__card-error {
  margin: 4px 0 0;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-danger);
}

.copilot-page__card-tools {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  flex-wrap: wrap;
  gap: 2px;
}

@media (max-width: 1024px) {
  .copilot-page__split {
    flex-direction: column;
    overflow-y: auto;
  }

  .copilot-page__main {
    overflow-y: auto;
  }

  .copilot-page__cards-pane {
    width: 100%;
    max-height: 320px;
  }
}
</style>
