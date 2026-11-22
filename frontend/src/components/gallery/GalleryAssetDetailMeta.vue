<template>
  <div class="gallery-asset-detail-meta">
    <div v-if="showPrompt && promptText" class="gallery-asset-detail-meta__section">
      <div class="gallery-asset-detail-meta__head">
        <span class="gallery-asset-detail-meta__label">{{ $t('gallery.prompt') }}</span>
        <DqIconButton
          type="text"
          size="sm"
          :label="$t('gallery.copy')"
          @click="copyPrompt"
        >
          <DqIcon><CopyDocument /></DqIcon>
        </DqIconButton>
      </div>
      <p class="gallery-asset-detail-meta__prompt">{{ promptText }}</p>
    </div>

    <div v-if="showPrompt && revisedPrompt" class="gallery-asset-detail-meta__section">
      <div class="gallery-asset-detail-meta__head">
        <span class="gallery-asset-detail-meta__label">{{ $t('gallery.revisedPrompt') }}</span>
        <DqIconButton type="text" size="sm" :label="$t('gallery.copy')" @click="copyRevisedPrompt">
          <DqIcon><CopyDocument /></DqIcon>
        </DqIconButton>
      </div>
      <p class="gallery-asset-detail-meta__prompt">{{ revisedPrompt }}</p>
    </div>

    <div v-if="showPrompt && negativePrompt" class="gallery-asset-detail-meta__section">
      <div class="gallery-asset-detail-meta__head">
        <span class="gallery-asset-detail-meta__label">{{ $t('gallery.negativePrompt') }}</span>
      </div>
      <p class="gallery-asset-detail-meta__prompt gallery-asset-detail-meta__prompt--muted">
        {{ negativePrompt }}
      </p>
    </div>

    <dl class="gallery-asset-detail-meta__rows">
      <div v-if="taskId" class="gallery-asset-detail-meta__row">
        <dt>{{ $t('gallery.taskId') }}</dt>
        <dd class="gallery-asset-detail-meta__mono">
          <code :title="taskId">{{ taskId }}</code>
          <DqIconButton
            type="text"
            size="sm"
            :label="$t('gallery.copy')"
            @click="copyTaskId"
          >
            <DqIcon :size="14"><CopyDocument /></DqIcon>
          </DqIconButton>
          <DqIconButton
            type="text"
            size="sm"
            :label="$t('gallery.viewTaskTrace')"
            @click="onViewTaskTrace"
          >
            <DqIcon :size="14"><ListOrdered /></DqIcon>
          </DqIconButton>
        </dd>
      </div>
      <div v-if="assetId" class="gallery-asset-detail-meta__row">
        <dt>{{ $t('gallery.assetId') }}</dt>
        <dd class="gallery-asset-detail-meta__mono">
          <code :title="assetId">{{ assetId }}</code>
          <DqIconButton
            type="text"
            size="sm"
            :label="$t('gallery.copy')"
            @click="copyAssetId"
          >
            <DqIcon :size="14"><CopyDocument /></DqIcon>
          </DqIconButton>
        </dd>
      </div>
      <div v-if="sourceActionLabelText" class="gallery-asset-detail-meta__row">
        <dt>{{ $t('gallery.sourceAction') }}</dt>
        <dd>{{ sourceActionLabelText }}</dd>
      </div>
      <div v-if="relationLabel" class="gallery-asset-detail-meta__row">
        <dt>{{ $t('gallery.relationType') }}</dt>
        <dd>{{ relationLabel }}</dd>
      </div>
      <div v-if="audioOperation" class="gallery-asset-detail-meta__row">
        <dt>{{ $t('gallery.operation') }}</dt>
        <dd>{{ audioOperation }}</dd>
      </div>
      <div v-if="audioTrack" class="gallery-asset-detail-meta__row">
        <dt>{{ $t('audio.tracks') }}</dt>
        <dd>{{ audioTrack }}</dd>
      </div>
      <div v-if="audioRegion" class="gallery-asset-detail-meta__row">
        <dt>{{ $t('gallery.region') }}</dt>
        <dd>{{ audioRegion }}</dd>
      </div>
      <div v-if="audioFormat" class="gallery-asset-detail-meta__row">
        <dt>{{ $t('audio.audioFormat') }}</dt>
        <dd>{{ audioFormat }}</dd>
      </div>
      <div v-if="item.model" class="gallery-asset-detail-meta__row">
        <dt>{{ $t('gallery.model') }}</dt>
        <dd>{{ item.model }}</dd>
      </div>
      <div v-if="adaptersLabel" class="gallery-asset-detail-meta__row gallery-asset-detail-meta__row--wide">
        <dt>{{ loraFieldLabel }}</dt>
        <dd :title="adaptersLabel">{{ adaptersLabel }}</dd>
      </div>
      <div v-if="controlNetLabel" class="gallery-asset-detail-meta__row gallery-asset-detail-meta__row--wide">
        <dt>{{ controlNetFieldLabel }}</dt>
        <dd :title="controlNetLabel">{{ controlNetLabel }}</dd>
      </div>
      <div v-if="resolutionLabel" class="gallery-asset-detail-meta__row">
        <dt>{{ $t('gallery.resolution') }}</dt>
        <dd>{{ resolutionLabel }}</dd>
      </div>
      <div v-if="steps != null" class="gallery-asset-detail-meta__row">
        <dt>{{ $t('gallery.steps') }}</dt>
        <dd>{{ steps }}</dd>
      </div>
      <div v-if="guidance != null" class="gallery-asset-detail-meta__row">
        <dt>{{ $t('gallery.cfg') }}</dt>
        <dd>{{ guidance }}</dd>
      </div>
      <div v-if="scheduler" class="gallery-asset-detail-meta__row">
        <dt>{{ schedulerFieldLabel }}</dt>
        <dd>{{ scheduler }}</dd>
      </div>
      <div v-if="seed != null" class="gallery-asset-detail-meta__row">
        <dt>{{ $t('gallery.seed') }}</dt>
        <dd class="gallery-asset-detail-meta__mono">{{ seed }}</dd>
      </div>
      <div v-if="durationLabel" class="gallery-asset-detail-meta__row">
        <dt>{{ $t('gallery.durationLabel') }}</dt>
        <dd>{{ durationLabel }}</dd>
      </div>
      <div v-if="item.created_at" class="gallery-asset-detail-meta__row">
        <dt>{{ $t('gallery.createdAt') }}</dt>
        <dd>{{ createdAtLabel }}</dd>
      </div>
    </dl>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { CopyDocument, ListOrdered } from '@danqing/dq-shell';
import { useI18n } from 'vue-i18n';
import type { GalleryItem } from '@/types';
import { api } from '@/utils/api';
import { openGlobalTaskQueue } from '@/utils/appEvents';
import { copyTextToClipboard } from '@/utils/clipboard';
import { toast } from '@/utils/feedback';
import {
  formatAdapterLabel,
  formatGalleryClock,
  formatGalleryDate,
  formatGalleryResolution,
  galleryAdapters,
  galleryAssetId,
  gallerySourceAction,
  galleryTaskId,
  metaNumber,
  metaText,
  sourceActionLabel,
} from '@/utils/galleryAssetMeta';
import { isAudioGalleryItem } from '@/utils/canvasAssets';
import { lineageRelationLabel } from '@/utils/lineageRelationLabel';
import { $tt } from '@/utils/i18n';

const props = defineProps<{
  item: GalleryItem;
  showPrompt?: boolean;
}>();

const { t: $t } = useI18n();

/** Prefer gallery.* ; fall back if locale HMR hasn't merged new keys yet. */
function tr(primary: string, fallback: string): string {
  const text = String($t(primary));
  return text === primary ? String($t(fallback)) : text;
}

const loraFieldLabel = computed(() => tr('gallery.lora', 'create.lora'));
const controlNetFieldLabel = computed(() => tr('gallery.controlNet', 'studio.controlNet'));
const schedulerFieldLabel = computed(() => tr('gallery.scheduler', 'loraTrain.scheduler'));

/** Task-params overlay for older assets that only stored a thin metadata blob. */
const taskOverlay = ref<Record<string, unknown>>({});

const mergedItem = computed((): GalleryItem => {
  if (!Object.keys(taskOverlay.value).length) return props.item;
  return {
    ...props.item,
    metadata: {
      ...(props.item.metadata || {}),
      ...taskOverlay.value,
    },
  };
});

const promptText = computed(() => (props.item.prompt || '').trim());
const revisedPrompt = computed(() => metaText(mergedItem.value, 'revised_prompt'));
const negativePrompt = computed(() =>
  metaText(mergedItem.value, 'negative_prompt', 'negativePrompt'),
);
const taskId = computed(() => galleryTaskId(props.item));
const assetId = computed(() => galleryAssetId(props.item));
const sourceActionLabelText = computed(() => sourceActionLabel(gallerySourceAction(props.item)));
const relationLabel = computed(() => {
  const rt = String(
    mergedItem.value.metadata?.relation_type ||
      (mergedItem.value as { relation_type?: string }).relation_type ||
      '',
  ).trim();
  if (!rt || rt === 'create') return '';
  return lineageRelationLabel(rt);
});
/** Audio-only operation row — never map image relation_type=create through action.audio.*. */
const audioOperation = computed(() => {
  if (!isAudioGalleryItem(props.item)) return '';
  const operation = metaText(mergedItem.value, 'operation');
  if (!operation || operation === 'create') return '';
  const key = `action.audio.${operation === 'cover-nofsq' ? 'cover' : operation}`;
  const label = $tt(key);
  return label === key ? operation : label;
});
const audioTrack = computed(() => metaText(mergedItem.value, 'track'));
const audioRegion = computed(() => {
  const start = metaNumber(mergedItem.value, 'region_start_sec');
  const end = metaNumber(mergedItem.value, 'region_end_sec');
  return start != null && end != null ? `${formatGalleryClock(start)} – ${formatGalleryClock(end)}` : '';
});
const audioFormat = computed(() => metaText(mergedItem.value, 'audio_format', 'format').toUpperCase());
const resolutionLabel = computed(() => formatGalleryResolution(props.item));
const steps = computed(() => metaNumber(mergedItem.value, 'steps'));
const guidance = computed(() =>
  metaNumber(mergedItem.value, 'guidance', 'guidance_scale'),
);
const scheduler = computed(() => metaText(mergedItem.value, 'scheduler'));
const seed = computed(() => metaNumber(mergedItem.value, 'seed'));
const adaptersLabel = computed(() => {
  const rows = galleryAdapters(mergedItem.value);
  if (!rows.length) return '';
  return rows.map(formatAdapterLabel).join(' · ');
});
const controlNetLabel = computed(() => {
  const guide = mergedItem.value.metadata?.structural_guide;
  if (!guide || typeof guide !== 'object') return '';
  const rec = guide as Record<string, unknown>;
  const model = String(rec.model_id || rec.model || '').trim();
  if (!model) return '';
  const weight = rec.weight != null && Number.isFinite(Number(rec.weight)) ? Number(rec.weight) : null;
  const type = String(rec.type || rec.guide_type || '').trim();
  const inpaint = rec.inpaint === true;
  const bits = [model];
  if (weight != null) bits.push(`× ${weight}`);
  if (type) bits.push(type);
  if (inpaint) {
    const tag = $t('gallery.inpaint');
    bits.push(tag === 'gallery.inpaint' ? $t('action.image.retouch') : tag);
  }
  return bits.join(' · ');
});
const durationLabel = computed(() => {
  const raw = props.item.duration_seconds ?? metaNumber(mergedItem.value, 'duration_seconds');
  return raw != null ? formatGalleryClock(raw) : '';
});
const createdAtLabel = computed(() => formatGalleryDate(props.item.created_at));

watch(
  () => props.item.path,
  async () => {
    taskOverlay.value = {};
    const tid = galleryTaskId(props.item);
    if (!tid) return;
    // New assets persist steps (+ adapters when used). Old tips only have seed/size.
    if (metaNumber(props.item, 'steps') != null) return;
    try {
      const task = (await api.gen.getMediaTask(tid)) as {
        params?: { request?: Record<string, unknown> } & Record<string, unknown>;
      };
      // Image create stores knobs under params.request; some rows flatten onto params.
      const req = (
        task?.params?.request && typeof task.params.request === 'object'
          ? task.params.request
          : task?.params
      ) as Record<string, unknown> | undefined;
      if (!req || typeof req !== 'object') return;
      const next: Record<string, unknown> = {};
      if (req.steps != null) next.steps = req.steps;
      if (req.guidance != null) next.guidance = req.guidance;
      const scheduler = String(req.scheduler || '').trim();
      if (scheduler) next.scheduler = scheduler;
      if (req.scheduler_shift != null) next.scheduler_shift = req.scheduler_shift;
      const negative = String(req.negative_prompt || '').trim();
      if (negative) next.negative_prompt = negative;
      if (Array.isArray(req.adapters) && req.adapters.length) next.adapters = req.adapters;
      if (req.structural_guide && typeof req.structural_guide === 'object') {
        const g = req.structural_guide as Record<string, unknown>;
        const modelId = String(g.model_id || '').trim();
        if (modelId) {
          next.structural_guide = {
            model_id: modelId,
            type: g.type || g.guide_type,
            weight: g.weight,
            inpaint: Boolean(g.inpaint_mask_asset_id),
          };
        }
      }
      if (Object.keys(next).length) taskOverlay.value = next;
    } catch {
      /* tip stays on asset metadata */
    }
  },
  { immediate: true },
);

async function copyPrompt() {
  if (!promptText.value) return;
  const ok = await copyTextToClipboard(promptText.value);
  if (ok) toast.success($t('gallery.copied'));
  else toast.error($t('gallery.copyFailed'));
}

async function copyRevisedPrompt() {
  if (!revisedPrompt.value) return;
  const ok = await copyTextToClipboard(revisedPrompt.value);
  if (ok) toast.success($t('gallery.copied'));
  else toast.error($t('gallery.copyFailed'));
}

async function copyAssetId() {
  if (!assetId.value) return;
  const ok = await copyTextToClipboard(assetId.value);
  if (ok) toast.success($t('gallery.assetIdCopied'));
  else toast.error($t('gallery.copyFailed'));
}

async function copyTaskId() {
  if (!taskId.value) return;
  const ok = await copyTextToClipboard(taskId.value);
  if (ok) toast.success($t('studio.taskIdCopied'));
  else toast.error($t('gallery.copyFailed'));
}

function onViewTaskTrace() {
  if (!taskId.value) return;
  openGlobalTaskQueue(taskId.value);
}
</script>

<style scoped>
.gallery-asset-detail-meta__section {
  margin-bottom: 10px;
}

.gallery-asset-detail-meta__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 6px;
}

.gallery-asset-detail-meta__label {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  letter-spacing: var(--dq-tracking-wide);
  text-transform: uppercase;
  color: var(--dq-label-tertiary);
}

.gallery-asset-detail-meta__prompt {
  margin: 0;
  font-size: var(--dq-font-size-body);
  line-height: 1.5;
  color: var(--dq-label-primary);
  word-break: break-word;
}

.gallery-asset-detail-meta__prompt--muted {
  color: var(--dq-label-secondary);
}

.gallery-asset-detail-meta__rows {
  display: flex;
  flex-wrap: wrap;
  gap: 10px 20px;
  margin: 0;
}

.gallery-asset-detail-meta__row {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: var(--dq-font-size-caption);
}

.gallery-asset-detail-meta__row--wide {
  flex: 1 1 100%;
  align-items: flex-start;
}

.gallery-asset-detail-meta__row--wide dd {
  word-break: break-word;
}

.gallery-asset-detail-meta__row dt {
  color: var(--dq-label-tertiary);
  flex-shrink: 0;
}

.gallery-asset-detail-meta__row dd {
  margin: 0;
  color: var(--dq-label-secondary);
}

.gallery-asset-detail-meta__mono {
  display: inline-flex;
  align-items: center;
  gap: 2px;
  min-width: 0;
}

.gallery-asset-detail-meta__mono code {
  font-family: var(--dq-font-mono);
  font-size: inherit;
  max-width: min(220px, 42vw);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
