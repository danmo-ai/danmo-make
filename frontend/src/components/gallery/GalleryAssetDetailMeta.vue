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

    <div v-if="taskId" class="gallery-asset-detail-meta__task">
      <TaskIdBadge :task-id="taskId" compact />
      <DqButton type="text" size="sm" @click="onViewTaskLog">
        {{ $t('gallery.viewTaskLog') }}
      </DqButton>
    </div>

    <dl class="gallery-asset-detail-meta__rows">
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
      <div v-if="item.model" class="gallery-asset-detail-meta__row">
        <dt>{{ $t('gallery.model') }}</dt>
        <dd>{{ item.model }}</dd>
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
import { computed } from 'vue';
import { CopyDocument } from '@danqing/dq-shell';
import { useI18n } from 'vue-i18n';
import TaskIdBadge from '@/components/studio/TaskIdBadge.vue';
import type { GalleryItem } from '@/types';
import { openTaskLog } from '@/utils/appEvents';
import { copyTextToClipboard } from '@/utils/clipboard';
import { toast } from '@/utils/feedback';
import {
  formatGalleryClock,
  formatGalleryDate,
  formatGalleryResolution,
  galleryAssetId,
  gallerySourceAction,
  galleryTaskId,
  metaNumber,
  sourceActionLabel,
} from '@/utils/galleryAssetMeta';
import { lineageRelationLabel } from '@/utils/lineageRelationLabel';

const props = defineProps<{
  item: GalleryItem;
  showPrompt?: boolean;
}>();

const { t: $t } = useI18n();

const promptText = computed(() => (props.item.prompt || '').trim());
const taskId = computed(() => galleryTaskId(props.item));
const assetId = computed(() => galleryAssetId(props.item));
const sourceActionLabelText = computed(() => sourceActionLabel(gallerySourceAction(props.item)));
const relationLabel = computed(() => {
  const rt = String(props.item.metadata?.relation_type || '').trim();
  if (!rt || rt === 'create') return '';
  return lineageRelationLabel(rt);
});
const resolutionLabel = computed(() => formatGalleryResolution(props.item));
const steps = computed(() => metaNumber(props.item, 'steps'));
const guidance = computed(() => metaNumber(props.item, 'guidance', 'guidance_scale'));
const seed = computed(() => metaNumber(props.item, 'seed'));
const durationLabel = computed(() => {
  const raw = props.item.duration_seconds ?? metaNumber(props.item, 'duration_seconds');
  return raw != null ? formatGalleryClock(raw) : '';
});
const createdAtLabel = computed(() => formatGalleryDate(props.item.created_at));

async function copyPrompt() {
  if (!promptText.value) return;
  const ok = await copyTextToClipboard(promptText.value);
  if (ok) toast.success($t('gallery.copied'));
  else toast.error($t('gallery.copyFailed'));
}

async function copyAssetId() {
  if (!assetId.value) return;
  const ok = await copyTextToClipboard(assetId.value);
  if (ok) toast.success($t('gallery.assetIdCopied'));
  else toast.error($t('gallery.copyFailed'));
}

function onViewTaskLog() {
  if (!taskId.value) return;
  openTaskLog(taskId.value);
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

.gallery-asset-detail-meta__task {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px 10px;
  margin-bottom: 10px;
  padding-bottom: 10px;
  border-bottom: 0.5px solid var(--dq-border-subtle);
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

.gallery-asset-detail-meta__row dt {
  color: var(--dq-label-tertiary);
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
