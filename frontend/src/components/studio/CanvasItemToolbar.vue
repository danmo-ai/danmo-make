<template>
  <div
    v-if="visible && item"
    class="canvas-item-toolbar dq-glass--popover"
    :style="toolbarStyle"
    @mousedown.stop
  >
    <span class="canvas-item-toolbar__label">{{ toolbarLabel }}</span>

    <template v-if="pageMedia === 'image' && isImage">
      <DqIconButton
        type="text"
        size="sm"
        :label="$t('action.image.rewrite')"
        @click="$emit('action', 'quick-reference')"
      >
        <DqIcon :size="14"><Picture /></DqIcon>
      </DqIconButton>
      <DqIconButton
        type="text"
        size="sm"
        :label="$t('action.image.retouch')"
        @click="$emit('action', 'retouch')"
      >
        <DqIcon :size="14"><Brush /></DqIcon>
      </DqIconButton>
      <DqIconButton
        type="text"
        size="sm"
        :label="$t('action.image.extend')"
        @click="$emit('action', 'extend')"
      >
        <DqIcon :size="14"><Grid /></DqIcon>
      </DqIconButton>
      <DqIconButton
        type="text"
        size="sm"
        :label="$t('action.image.upscale')"
        @click="$emit('action', 'upscale')"
      >
        <DqIcon :size="14"><ZoomIn /></DqIcon>
      </DqIconButton>
    </template>

    <template v-else-if="pageMedia === 'video' && isImage">
      <DqIconButton
        type="text"
        size="sm"
        :label="$t('action.video.animate')"
        @click="$emit('action', 'quick-animate')"
      >
        <DqIcon :size="14"><VideoCamera /></DqIcon>
      </DqIconButton>
    </template>

    <template v-else-if="pageMedia === 'video' && isVideo">
      <DqIconButton
        type="text"
        size="sm"
        :label="$t('action.video.edit')"
        @click="$emit('action', 'quick-animate-video')"
      >
        <DqIcon :size="14"><VideoCamera /></DqIcon>
      </DqIconButton>
      <DqIconButton
        type="text"
        size="sm"
        :label="$t('action.video.upscale')"
        @click="$emit('action', 'upscale')"
      >
        <DqIcon :size="14"><ZoomIn /></DqIcon>
      </DqIconButton>
    </template>

    <template v-else-if="pageMedia === 'audio' && isAudio">
      <DqIconButton
        type="text"
        size="sm"
        :label="$t('action.audio.cover')"
        @click="$emit('action', 'quick-cover')"
      >
        <DqIcon :size="14"><Headset /></DqIcon>
      </DqIconButton>
    </template>

    <DqIconButton
      v-if="hasWorkflow"
      type="text"
      size="sm"
      :label="$t('canvas.branchGenerate')"
      @click="$emit('action', 'branch')"
    >
      <DqIcon :size="14"><Plus /></DqIcon>
    </DqIconButton>

    <DqDropdown trigger="click" size="sm" @command="(cmd: string) => $emit('action', cmd)">
      <DqIconButton type="text" size="sm" :label="$t('canvas.moreMenu')" @click.stop>
        <DqIcon :size="14"><Menu /></DqIcon>
      </DqIconButton>
      <template #dropdown>
        <DqDropdownMenu>
          <template v-if="pageMedia === 'image' && isImage">
            <DqDropdownItem command="quick-control">
              {{ $t('canvas.guideBranch') }}
            </DqDropdownItem>
            <DqDropdownItem command="use-reference">
              {{ $t('canvas.useAsReference') }}
            </DqDropdownItem>
            <DqDropdownItem command="use-control">
              {{ $t('canvas.useAsControl') }}
            </DqDropdownItem>
            <DqDropdownItem divided command="export-png">
              {{ $t('canvas.exportNodePng') }}
            </DqDropdownItem>
          </template>
          <template v-else-if="pageMedia === 'video' && isImage">
            <DqDropdownItem command="use-start-frame">
              {{ $t('canvas.useAsStartFrame') }}
            </DqDropdownItem>
            <DqDropdownItem command="use-tail-frame">
              {{ $t('canvas.useAsTailFrame') }}
            </DqDropdownItem>
          </template>
          <template v-else-if="pageMedia === 'video' && isVideo">
            <DqDropdownItem command="use-animate-source">
              {{ $t('canvas.useAsAnimateSource') }}
            </DqDropdownItem>
            <DqDropdownItem command="use-video-source">
              {{ $t('canvas.useAsVideoSource') }}
            </DqDropdownItem>
          </template>
          <template v-else-if="pageMedia === 'audio' && isAudio">
            <DqDropdownItem command="use-cover-source">
              {{ $t('canvas.useAsCoverSource') }}
            </DqDropdownItem>
          </template>
          <DqDropdownItem command="snap-staging">
            {{ $t('canvas.snapStagingToNode') }}
          </DqDropdownItem>
          <DqDropdownItem command="rename">
            {{ $t('canvas.renameNode') }}
          </DqDropdownItem>
          <DqDropdownItem command="save-note">
            {{ $t('canvas.savePromptAsNote') }}
          </DqDropdownItem>
          <DqDropdownItem command="lineage">
            {{ $t('gallery.lineage') }}
          </DqDropdownItem>
          <DqDropdownItem
            v-if="(pageMedia === 'image' && isImage) || (pageMedia === 'video' && (isImage || isVideo))"
            command="copilot-image-to-prompt"
          >
            {{ $t('canvas.copilotImageToPrompt') }}
          </DqDropdownItem>
          <DqDropdownItem
            v-if="pageMedia === 'image' && isImage"
            command="copilot-analyze"
          >
            {{ $t('canvas.copilotAnalyze') }}
          </DqDropdownItem>
          <DqDropdownItem command="ai-describe">
            {{ $t('canvas.aiDescribe') }}
          </DqDropdownItem>
          <DqDropdownItem command="ai-describe-text">
            {{ $t('canvas.aiDescribeTextOnly') }}
          </DqDropdownItem>
            <DqDropdownItem
              v-if="pageMedia === 'image' && isImage"
              command="train-lora"
            >
              {{ $t('loraTrain.saveToDataset') }}
            </DqDropdownItem>
            <DqDropdownItem divided command="download">
            {{ $t('gallery.download') }}
          </DqDropdownItem>
        </DqDropdownMenu>
      </template>
    </DqDropdown>

    <span class="canvas-item-toolbar__sep" aria-hidden="true" />

    <DqIconButton
      type="text"
      size="sm"
      :label="$t('canvas.removeFromCanvas')"
      @click="$emit('action', 'remove')"
    >
      <DqIcon :size="14"><Close /></DqIcon>
    </DqIconButton>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useI18n } from 'vue-i18n';
import { lineageRelationLabel } from '@/utils/lineageRelationLabel';
import {
  Brush,
  Close,
  Grid,
  Headset,
  Menu,
  Picture,
  Plus,
  VideoCamera,
  ZoomIn,
} from '@danqing/dq-shell';
import type { GalleryItem } from '@/types';
import {
  isAudioGalleryItem,
  isImageGalleryItem,
  isVideoGalleryItem,
} from '@/utils/canvasAssets';

const props = defineProps<{
  visible: boolean;
  item: GalleryItem | null;
  left: number;
  top: number;
  placement?: 'above' | 'below';
  nodeHeight?: number;
  media?: import('@/composables/useCanvasStore').CanvasMedia;
  describing?: boolean;
}>();

defineEmits<{
  (e: 'action', action: string): void;
}>();

const { t: $t } = useI18n();

const pageMedia = computed(() => props.media || 'image');

const toolbarLabel = computed(() => {
  const rt = String(props.item?.metadata?.relation_type || '').trim();
  if (rt && rt !== 'create') return lineageRelationLabel(rt);
  return $t('canvas.nodeToolbarLabel');
});

const isAudio = computed(() => (props.item ? isAudioGalleryItem(props.item) : false));
const isVideo = computed(() => (props.item ? isVideoGalleryItem(props.item) : false));
const isImage = computed(() => (props.item ? isImageGalleryItem(props.item) : false));

const hasWorkflow = computed(
  () =>
    (pageMedia.value === 'image' && isImage.value) ||
    (pageMedia.value === 'video' && (isImage.value || isVideo.value)) ||
    (pageMedia.value === 'audio' && isAudio.value),
);

const toolbarStyle = computed(() => {
  const below = props.placement === 'below';
  const offset = below ? (props.nodeHeight ?? 0) + 10 : 0;
  return {
    left: `${props.left}px`,
    top: `${props.top}px`,
    transform: below
      ? `translate(-50%, ${offset}px)`
      : 'translate(-50%, calc(-100% - 10px))',
  };
});
</script>

<style scoped>
.canvas-item-toolbar {
  position: absolute;
  z-index: 60;
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 2px;
  max-width: min(96vw, 420px);
  padding: 4px 6px;
  border-radius: 10px;
  pointer-events: auto;
  box-shadow: var(--dq-shadow-md);
}

.canvas-item-toolbar__label {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  color: var(--dq-label-tertiary);
  padding: 0 4px 0 2px;
  white-space: nowrap;
}

.canvas-item-toolbar__sep {
  width: 1px;
  height: 18px;
  margin: 0 2px;
  background: var(--dq-border-subtle);
  opacity: 0.7;
}

.canvas-item-toolbar__menu-hint {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
  white-space: normal;
  line-height: 1.35;
}
</style>
