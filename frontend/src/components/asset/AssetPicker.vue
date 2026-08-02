<template>
  <div class="asset-picker">
    <div class="asset-picker__actions">
      <input
        ref="fileInputRef"
        type="file"
        class="asset-picker__file-input"
        :accept="acceptAttr"
        @change="onFileChange"
      />
      <DqButton type="primary" @click="triggerUpload">
        <DqIcon><upload /></DqIcon>
        {{ $t('assetPicker.upload') }}
      </DqButton>
      <DqButton size="sm" @click="openLibrary">
        <DqIcon><folder-opened /></DqIcon>
        {{ $t('assetPicker.library') }}
      </DqButton>
    </div>

    <div v-if="recentFiltered.length" class="asset-picker__recent">
      <div class="asset-picker__recent-label">{{ $t('assetPicker.recent') }}</div>
      <div class="asset-picker__recent-grid">
        <div
          v-for="item in recentFiltered"
          :key="String(item.path)"
          class="asset-picker__thumb"
          role="button"
          tabindex="0"
          @click.stop="emitPickFromRecent(item)"
          @keydown.enter.prevent="emitPickFromRecent(item)"
        >
          <img
            v-if="!thumbFailed[String(item.path)]"
            :key="`${String(item.path)}-${thumbRetryFile[String(item.path)] ? 'file' : 'thumb'}`"
            :src="thumbUrlForRecent(item)"
            alt=""
            @error="onThumbError(item)"
          />
          <div v-else class="asset-picker__thumb-fallback" />
        </div>
      </div>
    </div>

    <DqDialog
      v-model:open="libraryOpen"
      class="asset-picker-library-dialog"
      :title="$t('assetPicker.dialogTitle')"
      width="min(640px, 92vw)"
      destroy-on-close
      @update:open="(v: boolean) => { if (v) onLibraryOpened(); }"
    >
      <div v-dq-loading="libraryLoading" class="asset-picker__library">
        <DqEmpty
          v-if="!libraryLoading && libraryExhausted && libraryRows.length === 0"
          :description="$t('assetPicker.emptyLibrary')"
        />
        <div v-if="libraryRows.length" class="asset-picker__library-grid">
          <button
            v-for="row in libraryRows"
            :key="row.id"
            type="button"
            class="asset-picker__library-cell"
            @click="selectLibraryRow(row)"
          >
            <img v-if="acceptKind === 'image'" :src="row.thumbUrl" alt="" class="asset-picker__library-img" />
            <template v-else>
              <img
                v-if="acceptKind !== 'audio'"
                :src="row.thumbUrl"
                alt=""
                class="asset-picker__library-img"
              />
              <div v-else class="asset-picker__library-audio-fallback" aria-hidden="true" />
              <span class="asset-picker__library-badge">{{
                acceptKind === 'audio'
                  ? $t('assetPicker.filterAudio')
                  : $t('gallery.filterVideo')
              }}</span>
            </template>
          </button>
        </div>
        <div v-if="!libraryExhausted || libraryRows.length" class="asset-picker__library-more">
          <DqButton
            v-if="!libraryExhausted"
            size="sm"
            :loading="libraryLoading"
            @click="fetchLibraryPage(false)"
          >
            {{ $t('assetPicker.loadMore') }}
          </DqButton>
          <span v-else-if="libraryRows.length" class="asset-picker__library-end">{{ $t('assetPicker.noMore') }}</span>
        </div>
      </div>
    </DqDialog>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { useI18n } from 'vue-i18n';
import { toast } from '@/utils/feedback';
import { api } from '@/utils/api';
import type { AssetRow } from '@/types';

const props = withDefaults(
  defineProps<{
    acceptKind?: 'image' | 'video' | 'audio' | 'image_or_video';
    recentGallery?: Array<Record<string, unknown>>;
  }>(),
  {
    acceptKind: 'image',
    recentGallery: () => [],
  },
);

const emit = defineEmits<{
  pick: [payload: { path: string; previewUrl: string }];
}>();

const { t: $t } = useI18n();

const fileInputRef = ref<HTMLInputElement | null>(null);
const thumbFailed = ref<Record<string, boolean>>({});
/** After thumbnail 404, retry once with full image file (image assets only). */
const thumbRetryFile = ref<Record<string, boolean>>({});

const acceptAttr = computed(() => {
  if (props.acceptKind === 'video') return 'video/*,.mp4,.webm,.mov,.mkv,.avi';
  if (props.acceptKind === 'audio') {
    return 'audio/*,.mp3,.wav,.m4a,.aac,.flac,.ogg,.opus';
  }
  if (props.acceptKind === 'image_or_video') {
    return 'image/*,video/*,.mp4,.webm,.mov,.mkv,.avi';
  }
  return 'image/*';
});

function isVideoAsset(row: AssetRow): boolean {
  const k = String(row.kind || '');
  const mime = String(row.mime_type || '');
  if (k === 'video') return true;
  if (mime.startsWith('video/')) return true;
  const base = String(row.path || '')
    .split(/[/\\]/)
    .pop()
    ?.toLowerCase();
  if (base && /\.(mp4|mov|webm|mkv|avi)$/.test(base)) return true;
  return false;
}

function isAudioAsset(row: AssetRow): boolean {
  const k = String(row.kind || '');
  const mime = String(row.mime_type || '');
  if (k === 'audio') return true;
  if (mime.startsWith('audio/')) return true;
  const base = String(row.path || '')
    .split(/[/\\]/)
    .pop()
    ?.toLowerCase();
  return !!base && /\.(wav|mp3|flac|m4a|aac|opus|ogg)$/.test(base);
}

function isImageAsset(row: AssetRow): boolean {
  const k = String(row.kind || '');
  const mime = String(row.mime_type || '');
  if (k === 'image') return true;
  return mime.startsWith('image/');
}

const recentFiltered = computed(() => {
  const list = props.recentGallery || [];
  return list.filter((raw) => {
    const item = raw as Record<string, unknown>;
    const meta = item.metadata as Record<string, unknown> | undefined;
    const ext = String(item.name || '')
      .split('.')
      .pop()
      ?.toLowerCase() || '';
    if (props.acceptKind === 'video') {
      if (meta?.asset_kind === 'video') return true;
      return ['mp4', 'mov', 'avi', 'mkv', 'webm'].includes(ext);
    }
    if (props.acceptKind === 'audio') {
      if (meta?.asset_kind === 'audio') return true;
      return ['wav', 'mp3', 'flac', 'm4a', 'aac', 'opus', 'ogg'].includes(ext);
    }
    if (props.acceptKind === 'image_or_video') {
      if (meta?.asset_kind === 'audio') return false;
      if (meta?.asset_kind === 'video') return true;
      if (meta?.asset_kind === 'image') return true;
      if (['mp4', 'mov', 'avi', 'mkv', 'webm'].includes(ext)) return true;
      return !['wav', 'mp3', 'flac', 'm4a', 'aac', 'opus', 'ogg'].includes(ext);
    }
    if (meta?.asset_kind === 'video' || meta?.asset_kind === 'audio') return false;
    return !['mp4', 'mov', 'avi', 'mkv', 'webm', 'wav', 'mp3', 'flac', 'm4a', 'aac', 'opus', 'ogg'].includes(ext);
  });
});

function thumbUrlForRecent(item: Record<string, unknown>): string {
  const p = String(item.path || '');
  if (props.acceptKind === 'image' && thumbRetryFile.value[p] && p.startsWith('asset:')) {
    return api.gallery.getImageUrl(p);
  }
  const th = item.thumbnail as string | undefined;
  if (typeof th === 'string' && th.length) return th;
  if (p.startsWith('asset:')) {
    const id = p.slice('asset:'.length);
    return `/api/assets/${id}/thumbnail`;
  }
  return '';
}

function onThumbError(item: Record<string, unknown>) {
  const p = String(item.path || '');
  if (!p) return;
  if (props.acceptKind === 'image' && !thumbRetryFile.value[p] && p.startsWith('asset:')) {
    thumbRetryFile.value = { ...thumbRetryFile.value, [p]: true };
    return;
  }
  markThumbFailed(item);
}

function previewUrlForRecentPick(item: Record<string, unknown>, path: string): string {
  const meta = item.metadata as Record<string, unknown> | undefined;
  const ext = String(item.name || '')
    .split('.')
    .pop()
    ?.toLowerCase() || '';
  const isVideo = meta?.asset_kind === 'video' || ['mp4', 'mov', 'avi', 'mkv', 'webm'].includes(ext);
  if (isVideo && path.startsWith('asset:')) {
    return `/api/assets/${path.slice('asset:'.length)}/file`;
  }
  return api.gallery.getImageUrl(path);
}

function emitPickFromRecent(item: Record<string, unknown>) {
  const path = String(item.path || '');
  if (!path.startsWith('asset:')) {
    toast.warning(
      props.acceptKind === 'video'
        ? $t('assetPicker.needVideo')
        : props.acceptKind === 'audio'
          ? $t('assetPicker.needAudio')
          : props.acceptKind === 'image_or_video'
            ? $t('assetPicker.needImageOrVideo')
            : $t('assetPicker.needImage'),
    );
    return;
  }
  emit('pick', {
    path,
    previewUrl: previewUrlForRecentPick(item, path),
  });
}

function markThumbFailed(item: Record<string, unknown>) {
  const p = String(item.path || '');
  if (!p) return;
  thumbFailed.value = { ...thumbFailed.value, [p]: true };
}

function triggerUpload() {
  fileInputRef.value?.click();
}

async function onFileChange(ev: Event) {
  const input = ev.target as HTMLInputElement;
  const file = input.files?.[0];
  input.value = '';
  if (!file) return;

  if (props.acceptKind === 'image' && !file.type.startsWith('image/')) {
    toast.warning($t('assetPicker.needImage'));
    return;
  }
  if (props.acceptKind === 'video' && !file.type.startsWith('video/')) {
    toast.warning($t('assetPicker.needVideo'));
    return;
  }
  if (props.acceptKind === 'audio' && !file.type.startsWith('audio/')) {
    toast.warning($t('assetPicker.needAudio'));
    return;
  }
  if (
    props.acceptKind === 'image_or_video'
    && !file.type.startsWith('image/')
    && !file.type.startsWith('video/')
  ) {
    toast.warning($t('assetPicker.needImageOrVideo'));
    return;
  }

  try {
    const data = (await api.gen.uploadAsset(file)) as { id: string };
    const path = `asset:${data.id}`;
    const previewUrl = file.type.startsWith('video/')
      ? `/api/assets/${data.id}/file`
      : file.type.startsWith('audio/')
        ? `/api/assets/${data.id}/file`
        : api.gallery.getImageUrl(path);
    emit('pick', { path, previewUrl });
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e);
    toast.error($t('studio.uploadFailed', { msg }));
  }
}

/** Library dialog */
const libraryOpen = ref(false);
const libraryLoading = ref(false);
const libraryRows = ref<{ id: string; thumbUrl: string; previewUrl: string }[]>([]);
const libraryApiOffset = ref(0);
const libraryExhausted = ref(false);
const LIB_FETCH = 48;

function mapAssetRow(row: AssetRow) {
  const aid = row.id;
  const path = `asset:${aid}`;
  const video = isVideoAsset(row);
  const audio = isAudioAsset(row);
  return {
    id: aid,
    thumbUrl: row.thumbnail_url || `/api/assets/${aid}/thumbnail`,
    previewUrl: video || audio ? `/api/assets/${aid}/file` : api.gallery.getImageUrl(path),
  };
}

async function fetchLibraryPage(reset: boolean) {
  if (reset) {
    libraryApiOffset.value = 0;
    libraryRows.value = [];
    libraryExhausted.value = false;
  }
  if (libraryExhausted.value && !reset) return;

  libraryLoading.value = true;
  try {
    const kind =
      props.acceptKind === 'image'
        ? 'image'
        : props.acceptKind === 'video'
          ? 'video'
          : props.acceptKind === 'audio'
            ? 'audio'
            : null;
    const data = await api.gen.listAssets(kind, LIB_FETCH, libraryApiOffset.value, {
      exclude_upload_refs: false,
    });
    const items = (data.items || []) as AssetRow[];
    if (items.length < LIB_FETCH) libraryExhausted.value = true;
    libraryApiOffset.value += items.length;

    const filtered =
      props.acceptKind === 'image'
        ? items.filter((r) => isImageAsset(r))
        : props.acceptKind === 'video'
          ? items.filter((r) => isVideoAsset(r))
          : props.acceptKind === 'audio'
            ? items.filter((r) => isAudioAsset(r))
            : items.filter((r) => isImageAsset(r) || isVideoAsset(r));

    const mapped = filtered.map(mapAssetRow);
    if (reset) libraryRows.value = mapped;
    else libraryRows.value = libraryRows.value.concat(mapped);

    if (items.length === 0) libraryExhausted.value = true;
  } catch (e: unknown) {
    console.error(e);
    toast.error($t('gallery.loadFailed'));
    libraryExhausted.value = true;
  } finally {
    libraryLoading.value = false;
  }
}

function openLibrary() {
  libraryOpen.value = true;
}

function onLibraryOpened() {
  void fetchLibraryPage(true);
}

function selectLibraryRow(row: { id: string; previewUrl: string }) {
  const path = `asset:${row.id}`;
  emit('pick', { path, previewUrl: row.previewUrl });
  libraryOpen.value = false;
}
</script>

<style scoped>
.asset-picker__file-input {
  position: absolute;
  width: 0;
  height: 0;
  opacity: 0;
  pointer-events: none;
}

.asset-picker__thumb-fallback {
  width: 100%;
  height: 100%;
  background: var(--dq-bg-base);
}

.asset-picker__library {
  min-height: 120px;
}

.asset-picker__library-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.asset-picker__library-cell {
  position: relative;
  width: 72px;
  height: 72px;
  padding: 0;
  border: 1px solid var(--dq-border);
  border-radius: 8px;
  overflow: hidden;
  cursor: pointer;
  background: var(--dq-bg-base);
}


.asset-picker__library-audio-fallback {
  width: 100%;
  height: 100%;
  background: color-mix(in srgb, var(--dq-accent) 18%, var(--dq-bg-base));
}

.asset-picker__library-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.asset-picker__library-badge {
  position: absolute;
  right: 2px;
  bottom: 2px;
  font-size: var(--dq-font-size-caption);
  padding: 1px 4px;
  border-radius: 4px;
  background: var(--dq-mask);
  color: var(--dq-label-primary);
}

.asset-picker__library-more {
  margin-top: 12px;
  display: flex;
  justify-content: center;
}
</style>
