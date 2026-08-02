<template>
  <DqDrawer
    :open="modelValue"
    :title="$t('canvas.layers')"
    direction="rtl"
    size="320px"
    class="canvas-layer-drawer"
    @update:open="$emit('update:modelValue', $event)"
  >
    <div v-if="staging" class="canvas-layer__section">
      <div class="canvas-layer__section-title">{{ $t('canvas.stagingLayer') }}</div>
      <div
        class="canvas-layer__staging-row"
        :class="{ 'canvas-layer__staging-row--hidden': !staging.visible }"
      >
        <div class="canvas-layer__staging-icon" aria-hidden="true">◎</div>
        <div class="canvas-layer__info">
          <span class="canvas-layer__name">{{ $t('canvas.stagingLabel') }}</span>
          <span class="canvas-layer__dim">
            {{ Math.round(staging.width) }}×{{ Math.round(staging.height) }}
          </span>
        </div>
        <div class="canvas-layer__controls">
          <DqCheckbox
            :model-value="staging.visible"
            :aria-label="staging.visible ? $t('canvas.hideLayer') : $t('canvas.showLayer')"
            @update:model-value="$emit('staging-update', { visible: $event })"
            @click.stop
          />
          <DqIconButton
            v-if="selectedPath"
            type="text"
            size="sm"
            :label="$t('canvas.snapStagingToNode')"
            @click.stop="$emit('snap-staging')"
          >
            <DqIcon :size="12"><Grid /></DqIcon>
          </DqIconButton>
          <DqIconButton
            type="text"
            size="sm"
            :label="$t('canvas.focusStaging')"
            @click.stop="$emit('focus-staging')"
          >
            <DqIcon :size="12"><Aim /></DqIcon>
          </DqIconButton>
        </div>
      </div>
    </div>

    <div v-if="showOverlays && overlayRows.length" class="canvas-layer__section">
      <div class="canvas-layer__section-title">{{ $t('canvas.overlaySection') }}</div>
      <div
        v-for="row in overlayRows"
        :key="row.kind"
        class="canvas-layer__overlay-row"
        :class="{ 'canvas-layer__overlay-row--hidden': !row.visible }"
      >
        <div class="canvas-layer__thumb">
          <img :src="row.thumb" :alt="row.label" />
        </div>
        <div class="canvas-layer__overlay-body">
          <div class="canvas-layer__overlay-head">
            <span class="canvas-layer__name">{{ row.label }}</span>
            <DqCheckbox
              :model-value="row.visible"
              :aria-label="row.visible ? $t('canvas.hideLayer') : $t('canvas.showLayer')"
              @update:model-value="$emit('overlay-update', row.kind, { visible: $event })"
              @click.stop
            />
          </div>
          <div class="canvas-layer__overlay-opacity">
            <span class="canvas-layer__opacity-label">{{ $t('canvas.overlayOpacity') }}</span>
            <DqSlider
              :model-value="row.opacity"
              :min="0.1"
              :max="1"
              :step="0.05"
              @update:model-value="$emit('overlay-update', row.kind, { opacity: $event })"
            />
          </div>
          <DqButton
            type="text"
            size="sm"
            class="canvas-layer__overlay-clear"
            @click="$emit('overlay-clear', row.kind)"
          >
            {{ $t('canvas.clearOverlay') }}
          </DqButton>
        </div>
      </div>
    </div>

    <div v-if="layerList.length === 0 && !overlayRows.length" class="canvas-layer__empty">
      <DqEmpty :description="$t('canvas.noLayers')" />
      <DqButton type="primary" @click="$emit('import-works')">
        {{ $t('canvas.importWorks') }}
      </DqButton>
    </div>
    <div v-else-if="layerList.length" class="canvas-layer__section">
      <div v-if="showOverlays && overlayRows.length" class="canvas-layer__section-title">
        {{ $t('canvas.assetLayers') }}
      </div>
      <p class="canvas-layer__rename-hint">{{ $t('canvas.renameNodeHint') }}</p>
      <div class="canvas-layer__list">
        <div
          v-for="entry in layerList"
          :key="entry.path"
          class="canvas-layer__row"
          :class="{
            'canvas-layer__row--selected': entry.path === selectedPath,
            'canvas-layer__row--hidden': !entry.visible,
          }"
          @click="$emit('select', entry.path)"
        >
          <div class="canvas-layer__thumb">
            <div v-if="entry.isAudio" class="canvas-layer__thumb-audio" aria-hidden="true">
              <DqIcon :size="18"><Headset /></DqIcon>
            </div>
            <img v-else :src="entry.thumb" :alt="entry.displayName" />
          </div>
          <div class="canvas-layer__info" @dblclick.stop="startRename(entry)">
            <DqInput
              v-if="renamingPath === entry.path"
              ref="renameInputRef"
              v-model="renameDraft"
              size="sm"
              class="canvas-layer__rename-input"
              :placeholder="$t('canvas.renameNodePlaceholder')"
              @keydown.enter.stop="commitRename(entry.path)"
              @keydown.esc.stop="cancelRename"
              @blur="commitRename(entry.path)"
              @click.stop
            />
            <span v-else class="canvas-layer__name" :title="entry.displayName">
              {{ entry.displayName }}
            </span>
            <span class="canvas-layer__dim">{{ entry.subtitle }}</span>
            <span v-if="entry.note" class="canvas-layer__note">{{ entry.note }}</span>
          </div>
          <div class="canvas-layer__controls">
            <DqIconButton
              type="text"
              size="sm"
              :label="$t('canvas.renameNode')"
              @click.stop="startRename(entry)"
            >
              <DqIcon :size="12"><Document /></DqIcon>
            </DqIconButton>
            <DqCheckbox
              :model-value="entry.visible"
              :aria-label="entry.visible ? $t('canvas.hideLayer') : $t('canvas.showLayer')"
              @update:model-value="$emit('toggle-visibility', entry.path, $event)"
              @click.stop
            />
            <DqIconButton
              type="text"
              size="sm"
              :label="$t('canvas.removeFromCanvas')"
              @click.stop="$emit('remove', entry.path)"
            >
              <DqIcon :size="12"><Delete /></DqIcon>
            </DqIconButton>
          </div>
        </div>
      </div>
    </div>

    <div class="canvas-layer__footer">
      <DqButton type="primary" @click="$emit('import-works')">
        {{ $t('canvas.importWorks') }}
      </DqButton>
    </div>
  </DqDrawer>
</template>

<script setup lang="ts">
import { computed, ref, nextTick } from 'vue';
import { Aim, Delete, Document, Headset } from '@danqing/dq-shell';
import { useI18n } from 'vue-i18n';
import type {
  GalleryItem,
  CanvasItemState,
  CanvasOverlaysState,
  CanvasOverlayLayer,
  CanvasOverlayKind,
  CanvasStagingState,
} from '@/types';
import { api } from '@/utils/api';
import {
  canvasNodeDisplayName,
  formatGalleryDuration,
  isAudioGalleryItem,
  isVideoGalleryItem,
  previewUrlForGalleryItem,
} from '@/utils/canvasAssets';
import { OVERLAY_KINDS_BY_MEDIA, OVERLAY_LABEL_KEYS } from '@/utils/canvasOverlays';
import type { CanvasMedia } from '@/composables/useCanvasStore';

const props = defineProps<{
  modelValue: boolean;
  items: Record<string, CanvasItemState>;
  galleryItems: GalleryItem[];
  selectedPath: string;
  showOverlays?: boolean;
  overlays?: CanvasOverlaysState | null;
  staging?: CanvasStagingState | null;
  media?: CanvasMedia;
}>();

const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void;
  (e: 'select', path: string): void;
  (e: 'remove', path: string): void;
  (e: 'rename', path: string, label: string): void;
  (e: 'toggle-visibility', path: string, visible: boolean): void;
  (e: 'overlay-update', kind: CanvasOverlayKind, patch: Partial<CanvasOverlayLayer>): void;
  (e: 'overlay-clear', kind: CanvasOverlayKind): void;
  (e: 'staging-update', patch: Partial<CanvasStagingState>): void;
  (e: 'focus-staging'): void;
  (e: 'import-works'): void;
  (e: 'snap-staging'): void;
}>();

const { t } = useI18n();

const renamingPath = ref('');
const renameDraft = ref('');
const renameInputRef = ref<{ $el?: HTMLElement } | null>(null);

type LayerEntry = {
  path: string;
  visible: boolean;
  zIndex: number;
  isAudio: boolean;
  displayName: string;
  customLabel: string;
  thumb: string;
  subtitle: string;
  note: string;
};

async function startRename(entry: LayerEntry) {
  renamingPath.value = entry.path;
  renameDraft.value = entry.customLabel || entry.displayName;
  await nextTick();
  const input = renameInputRef.value?.$el?.querySelector('input');
  input?.focus();
  input?.select();
}

function commitRename(path: string) {
  if (renamingPath.value !== path) return;
  const trimmed = renameDraft.value.trim();
  emit('rename', path, trimmed);
  renamingPath.value = '';
}

function cancelRename() {
  renamingPath.value = '';
}

function cancelRenameIfActive(): boolean {
  if (!renamingPath.value) return false;
  cancelRename();
  return true;
}

async function startRenameForPath(path: string) {
  const entry = layerList.value.find((e) => e.path === path);
  if (entry) await startRename(entry);
}

defineExpose({ startRenameForPath, cancelRenameIfActive });

function truncateNote(text: string, max = 72): string {
  const s = text.replace(/\s+/g, ' ').trim();
  if (s.length <= max) return s;
  return `${s.slice(0, max - 1)}…`;
}

const overlayRows = computed(() => {
  if (!props.showOverlays || !props.overlays) return [];
  const media = props.media || 'image';
  const kinds = OVERLAY_KINDS_BY_MEDIA[media];
  const rows: Array<{
    kind: CanvasOverlayKind;
    label: string;
    thumb: string;
    visible: boolean;
    opacity: number;
  }> = [];

  for (const kind of kinds) {
    const layer = props.overlays[kind];
    if (!layer?.path) continue;
    const gi = props.galleryItems.find((g) => g.path === layer.path);
    rows.push({
      kind,
      label: t(OVERLAY_LABEL_KEYS[kind]),
      thumb: gi ? previewUrlForGalleryItem(gi) : api.gallery.getImageUrl(layer.path),
      visible: layer.visible !== false,
      opacity: layer.opacity ?? 0.45,
    });
  }

  return rows;
});

const layerList = computed(() => {
  const entries = Object.entries(props.items).sort((a, b) => b[1].zIndex - a[1].zIndex);

  return entries.map(([path, state]) => {
    const gi = props.galleryItems.find((g) => g.path === path);
    const isAudio = gi ? isAudioGalleryItem(gi) : false;
    const isVideo = gi ? isVideoGalleryItem(gi) : false;

    let subtitle = '';
    if (isAudio) {
      const dur = gi?.duration_seconds ? formatGalleryDuration(gi.duration_seconds) : '';
      subtitle = dur
        ? `${t('gallery.audioLabel')} · ${dur}`
        : t('gallery.audioLabel');
    } else if (isVideo) {
      const parts: string[] = [t('gallery.filterVideo')];
      if (gi?.duration_seconds) parts.push(formatGalleryDuration(gi.duration_seconds));
      if (gi?.width && gi?.height) parts.push(`${gi.width}×${gi.height}`);
      subtitle = parts.join(' · ');
    } else if (gi?.width && gi?.height) {
      subtitle = `${gi.width}×${gi.height}`;
    }

    const customLabel = (state.label || '').trim();
    const displayName = canvasNodeDisplayName(path, state, gi);

    return {
      path,
      visible: state.visible,
      zIndex: state.zIndex,
      isAudio,
      displayName,
      customLabel,
      thumb: gi ? previewUrlForGalleryItem(gi) : api.gallery.getImageUrl(path),
      subtitle,
      note: state.note ? truncateNote(state.note) : '',
    };
  });
});
</script>

<style scoped>
.canvas-layer__empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  padding: 32px 24px;
}

.canvas-layer__footer {
  margin-top: auto;
  padding: 12px 16px 16px;
  border-top: 1px solid var(--dq-border-subtle);
}

.canvas-layer__footer :deep(.dq-button) {
  width: 100%;
}

.canvas-layer__section {
  padding: 0 0 8px;
}

.canvas-layer__section-title {
  padding: 12px 16px 6px;
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  letter-spacing: var(--dq-tracking-wider);
  text-transform: uppercase;
  color: var(--dq-label-secondary);
}

.canvas-layer__rename-hint {
  margin: 0;
  padding: 0 16px 8px;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
  line-height: 1.4;
}

.canvas-layer__list {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 0 12px 12px;
}

.canvas-layer__row,
.canvas-layer__overlay-row {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 8px;
  border-radius: 8px;
  cursor: pointer;
  transition: background 0.15s;
}

.canvas-layer__row:hover,
.canvas-layer__overlay-row:hover {
  background: var(--dq-fill-secondary);
}

.canvas-layer__row--selected {
  background: color-mix(in srgb, var(--dq-accent) 10%, transparent);
}

.canvas-layer__row--hidden,
.canvas-layer__overlay-row--hidden {
  opacity: 0.45;
}

.canvas-layer__thumb {
  width: 44px;
  height: 44px;
  border-radius: 6px;
  overflow: hidden;
  flex-shrink: 0;
  background: var(--dq-fill-secondary);
}

.canvas-layer__thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.canvas-layer__thumb-audio {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--dq-accent);
  background: color-mix(in srgb, var(--dq-accent) 12%, var(--dq-fill-secondary));
}

.canvas-layer__info {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.canvas-layer__name {
  font-size: var(--dq-font-size-body);
  font-weight: 500;
  color: var(--dq-label-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.canvas-layer__rename-input {
  width: 100%;
}

.canvas-layer__dim {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
}

.canvas-layer__note {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
  line-height: 1.35;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.canvas-layer__controls {
  display: flex;
  align-items: center;
  gap: 2px;
  flex-shrink: 0;
}

.canvas-layer__overlay-row {
  margin: 0 12px 6px;
  padding: 10px;
  border: 1px solid var(--dq-border-subtle);
  cursor: default;
}

.canvas-layer__overlay-body {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.canvas-layer__overlay-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.canvas-layer__overlay-opacity {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.canvas-layer__opacity-label {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
}

.canvas-layer__overlay-clear {
  align-self: flex-start;
  padding: 0;
  min-height: auto;
}

.canvas-layer__staging-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 0 12px 8px;
  padding: 10px;
  border: 1px dashed var(--dq-border-subtle);
  border-radius: 8px;
}

.canvas-layer__staging-row--hidden {
  opacity: 0.45;
}

.canvas-layer__staging-icon {
  width: 44px;
  height: 44px;
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: var(--dq-font-size-body);
  color: var(--dq-accent);
  background: color-mix(in srgb, var(--dq-accent) 10%, transparent);
  flex-shrink: 0;
}
</style>
