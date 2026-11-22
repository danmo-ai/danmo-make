<!-- @ts-nocheck -->
<template>
  <div
    class="gallery-audio-detail"
    :class="[`gallery-audio-detail--${variant}`, { 'gallery-audio-detail--has-timeline': hasTimeline }]"
  >
    <div class="gallery-audio-detail__layout">
      <div class="gallery-audio-detail__main">
        <DqSegmented
          v-if="compareSrc"
          v-model="compareMode"
          size="sm"
          :options="[
            { label: $t('audio.resultCompare'), value: 'result' },
            { label: $t('audio.sourceOriginal'), value: 'source' },
          ]"
        />
        <AudioMusicPlayer
          ref="playerRef"
          :key="`${item.path}:${compareMode}`"
          :src="activeSrc"
          :title="displayTitle"
          :subtitle="displaySubtitle"
          :hue="artHue"
          :duration-seconds="durationSeconds"
          :layout="variant === 'lightbox' ? 'featured' : 'inline'"
          :autoplay="false"
          :show-download="variant === 'lightbox'"
          @download="emit('download')"
          @timeupdate="onTimeUpdate"
        />

        <template v-if="variant === 'lightbox' && showMeta">
          <GalleryAssetDetailMeta :item="item" show-prompt />
        </template>
      </div>

      <div class="gallery-audio-detail__lyrics-block studio-audio-effective-lyrics">
        <div class="studio-audio-effective-lyrics__head">
          <span class="studio-audio-effective-lyrics__label">
            {{ hasTimeline ? $t('gallery.syncedLyrics') : $t('audio.effectiveLyrics') }}
            <span
              v-if="hasTimeline"
              class="gallery-audio-detail__timeline-badge"
              :class="{ 'is-aligned': isAligned }"
              :title="isAligned ? $t('gallery.lyricsAlignedHint') : $t('gallery.lyricsEstimatedHint')"
            >
              {{ alignmentBadge }}
            </span>
          </span>
          <div class="gallery-audio-detail__lyrics-actions">
            <DqButton v-if="hasTimeline" type="text" size="sm" @click="toggleTimelineEdit">
              {{ editMode ? $t('common.cancel') : $t('audio.editTimeline') }}
            </DqButton>
            <DqButton v-if="editMode" type="text" size="sm" :disabled="timelineSaving" @click="saveTimeline">
              {{ $t('audio.saveTimeline') }}
            </DqButton>
            <DqButton v-if="editMode" type="text" size="sm" :disabled="timelineSaving" @click="resetTimeline">
              {{ $t('audio.resetTimeline') }}
            </DqButton>
            <DqButton
              v-if="hasTimeline && !editMode"
              type="text"
              size="sm"
              @click="showFullText = !showFullText"
            >
              {{ showFullText ? $t('gallery.lyricsShowSynced') : $t('gallery.lyricsShowFull') }}
            </DqButton>
            <DqButton v-if="lyricsText" type="text" size="sm" @click="copyLyrics">
              {{ $t('gallery.copy') }}
            </DqButton>
            <DqButton v-if="lyricsText" type="text" size="sm" @click="downloadLyrics">
              {{ $t('audio.downloadLyrics') }}
            </DqButton>
            <DqButton v-if="hasTimeline && !editMode" type="text" size="sm" @click="downloadLrc">
              {{ $t('gallery.downloadLrc') }}
            </DqButton>
          </div>
        </div>
        <div v-if="editMode" class="gallery-audio-detail__editor">
          <p class="studio-field-footnote">{{ $t('audio.timelineEditHint') }}</p>
          <div class="gallery-audio-detail__offset">
            <DqInput v-model="offsetInput" size="sm" inputmode="decimal" :placeholder="$t('audio.shiftTimeline')" />
            <DqButton size="sm" @click="applyOffset">{{ $t('audio.shiftTimeline') }}</DqButton>
          </div>
          <div v-if="durationSeconds" class="gallery-audio-detail__track">
            <div
              v-for="(line, index) in draftLines"
              :key="index"
              class="gallery-audio-detail__segment"
              :style="segmentStyle(line)"
              @click="onSeek(line.start)"
            >
              <button
                type="button"
                class="gallery-audio-detail__handle gallery-audio-detail__handle--start"
                :aria-label="$t('audio.regionStart')"
                @pointerdown.stop="startBoundaryDrag($event, index, 'start')"
              />
              <button
                type="button"
                class="gallery-audio-detail__handle gallery-audio-detail__handle--end"
                :aria-label="$t('audio.regionEnd')"
                @pointerdown.stop="startBoundaryDrag($event, index, 'end')"
              />
            </div>
          </div>
          <div v-for="(line, index) in draftLines" :key="`edit-${index}`" class="gallery-audio-detail__edit-row">
            <DqInput :model-value="String(line.start)" size="sm" inputmode="decimal" @update:model-value="updateLine(index, { start: Number($event) })" />
            <DqInput :model-value="String(line.end)" size="sm" inputmode="decimal" @update:model-value="updateLine(index, { end: Number($event) })" />
            <DqInput :model-value="line.text" size="sm" @update:model-value="updateLine(index, { text: String($event) })" />
          </div>
        </div>
        <AudioLyricsScroller
          v-else-if="hasTimeline && !showFullText"
          class="gallery-audio-detail__scroller"
          :lines="timelineLines"
          :current-time="currentTime"
          :compact="variant !== 'lightbox'"
          :karaoke="isAligned"
          @seek="onSeek"
        />
        <pre v-else-if="lyricsText" class="studio-audio-effective-lyrics__body">{{ lyricsText }}</pre>
        <p v-else-if="isInstrumental" class="gallery-audio-detail__lyrics-empty">
          {{ $t('gallery.lyricsInstrumental') }}
        </p>
        <p v-else class="gallery-audio-detail__lyrics-empty">
          {{ $t('gallery.lyricsUnavailable') }}
        </p>
        <p v-if="lyricsText" class="studio-field-footnote studio-audio-effective-lyrics__hint">
          {{
            !hasTimeline
              ? $t('audio.effectiveLyricsHint')
              : isAligned
                ? $t('gallery.lyricsAlignedHint')
                : $t('gallery.lyricsEstimatedHint')
          }}
        </p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import AudioLyricsScroller from '@/components/audio/AudioLyricsScroller.vue';
import AudioMusicPlayer from '@/components/audio/AudioMusicPlayer.vue';
import GalleryAssetDetailMeta from '@/components/gallery/GalleryAssetDetailMeta.vue';
import type { GalleryItem } from '@/types';
import { assetDisplayLabel } from '@/utils/assetDisplay';
import { toast } from '@/utils/feedback';
import { api } from '@/utils/api';
import { $tt } from '@/utils/i18n';
import {
  linesFromSegments,
  linesToLrc,
  lyricsToPlainText,
  shiftLyricLines,
  updateLyricLine,
  type LyricLine,
} from '@/utils/lyricsTimeline';

const props = defineProps({
  item: { type: Object as () => GalleryItem, required: true },
  src: { type: String, required: true },
  compareSrc: { type: String, default: '' },
  variant: { type: String, default: 'sidebar' },
  durationLabel: { type: String, default: '' },
  showMeta: { type: Boolean, default: true },

});

const emit = defineEmits(['download', 'saved']);

const { t: $t } = useI18n();

const playerRef = ref<InstanceType<typeof AudioMusicPlayer> | null>(null);
const currentTime = ref(0);
const compareMode = ref('result');
const activeSrc = computed(() => compareMode.value === 'source' && props.compareSrc ? props.compareSrc : props.src);
const showFullText = ref(false);
const editMode = ref(false);
const timelineSaving = ref(false);
const draftLines = ref<LyricLine[]>([]);
const offsetInput = ref('');

const displayTitle = computed(() => assetDisplayLabel(props.item));

const displaySubtitle = computed(() => {
  const parts: string[] = [];
  if (props.item.model) parts.push(props.item.model);
  if (props.durationLabel) parts.push(props.durationLabel);
  return parts.join(' · ');
});

const durationSeconds = computed(() => {
  const raw =
    props.item.duration_seconds ??
    (props.item.metadata?.duration_seconds as number | undefined);
  const n = Number(raw);
  return Number.isFinite(n) && n > 0 ? n : 0;
});

const lyricsText = computed(() => {
  const meta = props.item.metadata || {};
  const eff = String(meta.lyrics_effective || '').trim();
  if (eff) return eff;
  return String(meta.lyrics_input || '').trim();
});

const isInstrumental = computed(
  () => props.item.metadata?.lyrics_alignment === 'instrumental',
);

const alignmentMode = computed(() => String(props.item.metadata?.lyrics_alignment || ''));
const isAligned = computed(() => alignmentMode.value === 'dit_cross_attention');
const alignmentBadge = computed(() => {
  if (alignmentMode.value === 'user_edit') return $tt('audio.lyricsUserEdited');
  return isAligned.value ? $tt('gallery.lyricsAligned') : $tt('gallery.lyricsEstimated');
});

const timelineLines = computed(() =>
  linesFromSegments(props.item.metadata?.lyrics_segments, durationSeconds.value),
);
const hasTimeline = computed(() => timelineLines.value.length > 0);

watch(
  () => props.item.path,
  () => {
    currentTime.value = 0;
    compareMode.value = 'result';
    showFullText.value = false;
    editMode.value = false;
    draftLines.value = [];
    offsetInput.value = '';
  },
);

function linesFromText(text: string, duration: number): LyricLine[] {
  const rows = lyricsToPlainText(text).split('\n').map((line) => line.trim()).filter((line) => line && !/^\[[^]]+\]$/.test(line));
  const span = rows.length ? duration / rows.length : 0;
  return rows.map((row, index) => ({ start: index * span, end: (index + 1) * span, text: row }));
}

function toggleTimelineEdit() {
  if (editMode.value) {
    editMode.value = false;
    return;
  }
  draftLines.value = timelineLines.value.map((line) => ({ ...line }));
  if (!draftLines.value.length) draftLines.value = linesFromText(lyricsText.value, durationSeconds.value);
  editMode.value = true;
}

function updateLine(index: number, patch: Partial<LyricLine>) {
  draftLines.value = updateLyricLine(draftLines.value, index, patch, durationSeconds.value);
}

function applyOffset() {
  const offset = Number(offsetInput.value);
  if (!Number.isFinite(offset)) return;
  draftLines.value = shiftLyricLines(draftLines.value, offset, durationSeconds.value);
}

function segmentStyle(line: LyricLine) {
  const total = Math.max(0.001, durationSeconds.value);
  return {
    left: `${Math.min(100, Math.max(0, line.start / total * 100))}%`,
    width: `${Math.min(100, Math.max(1, (line.end - line.start) / total * 100))}%`,
  };
}

function startBoundaryDrag(event: PointerEvent, index: number, field: 'start' | 'end') {
  const handle = event.currentTarget as HTMLElement;
  const track = handle.parentElement?.parentElement;
  if (!track) return;
  const rect = track.getBoundingClientRect();
  const original = draftLines.value[index]?.[field] || 0;
  const startX = event.clientX;
  handle.setPointerCapture(event.pointerId);
  const onMove = (move: PointerEvent) => {
    const seconds = (move.clientX - startX) / Math.max(1, rect.width) * durationSeconds.value;
    updateLine(index, { [field]: Math.min(durationSeconds.value, Math.max(0, original + seconds)) });
  };
  const onUp = () => {
    handle.removeEventListener('pointermove', onMove);
    handle.removeEventListener('pointerup', onUp);
    handle.removeEventListener('pointercancel', onUp);
  };
  handle.addEventListener('pointermove', onMove);
  handle.addEventListener('pointerup', onUp);
  handle.addEventListener('pointercancel', onUp);
}

function originalTimeline() {
  const meta = props.item.metadata || {};
  const segments = meta.engine_lyrics_segments || meta.lyrics_segments;
  return {
    lines: linesFromSegments(segments, durationSeconds.value),
    alignment: String(meta.engine_lyrics_alignment || meta.lyrics_alignment || 'structure_estimate'),
  };
}

async function saveTimeline(lines = draftLines.value, alignment = 'user_edit') {
  const assetId = props.item.path.startsWith('asset:') ? props.item.path.slice(6) : '';
  if (!assetId || timelineSaving.value) return;
  timelineSaving.value = true;
  try {
    const original = originalTimeline();
    const text = lines.map((line) => line.text.trim()).filter(Boolean).join('\n');
    const body = {
      title: props.item.title,
      lyrics_effective: text,
      lyrics_input: text,
      lyrics_segments: lines.map((line) => ({
        label: line.label || 'body',
        start_sec: line.start,
        end_sec: line.end,
        text: line.text,
      })),
      lyrics_alignment: alignment,
      engine_lyrics_segments: original.lines.map((line) => ({
        label: line.label || 'body', start_sec: line.start, end_sec: line.end, text: line.text,
      })),
      engine_lyrics_alignment: original.alignment,
      lrc: linesToLrc(lines),
    };
    await api.gallery.updateAsset(assetId, body);
    Object.assign(props.item.metadata, body);
    props.item.title = String(body.title || '');
    draftLines.value = lines.map((line) => ({ ...line }));
    editMode.value = false;
    toast.success($tt('audio.timelineSaved'));
    emit('saved', { item: props.item, body });
  } catch (error) {
    toast.error($tt('audio.timelineSaveFailed', { msg: error instanceof Error ? error.message : String(error) }));
  } finally {
    timelineSaving.value = false;
  }
}

async function resetTimeline() {
  const original = originalTimeline();
  await saveTimeline(original.lines, original.alignment);
}

function onTimeUpdate(t: number) {
  currentTime.value = t;
}

function onSeek(seconds: number) {
  playerRef.value?.seek(seconds);
  currentTime.value = seconds;
}

const artHue = computed(() => artHueForPrompt(displayTitle.value));

function artHueForPrompt(text: string) {
  let h = 0;
  const s = String(text || 'audio');
  for (let i = 0; i < s.length; i += 1) {
    h = (h * 31 + s.charCodeAt(i)) % 360;
  }
  return h;
}

async function copyLyrics() {
  if (!lyricsText.value) return;
  try {
    await navigator.clipboard.writeText(lyricsText.value);
    toast.success($tt('gallery.copied'));
  } catch {
    const ta = document.createElement('textarea');
    ta.value = lyricsText.value;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    toast.success($tt('gallery.copied'));
  }
}

function saveText(text: string, suffix: string) {
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  const base = (props.item.name || 'lyrics').replace(/\.[^.]+$/, '');
  a.href = url;
  a.download = base + suffix;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function downloadLyrics() {
  if (!lyricsText.value) return;
  saveText(lyricsText.value + '\n', '_lyrics.txt');
}

function downloadLrc() {
  if (!hasTimeline.value) return;
  saveText(linesToLrc(timelineLines.value), '.lrc');
}
</script>

<style scoped>
/* Lightbox overrides for dark theme readability */
.gallery-audio-detail--lightbox .gallery-audio-detail__lyrics-block {
  background: var(--dq-surface-inset, var(--dq-fill-tertiary));
  border: 0.5px solid var(--dq-border-subtle);
}

.gallery-audio-detail--lightbox .studio-audio-effective-lyrics__body {
  color: var(--dq-label-primary);
}

/* Pop-player layout when lightbox is widened for timed lyrics (dialog ≥ ~960px). */
.gallery-audio-detail--lightbox.gallery-audio-detail--has-timeline .gallery-audio-detail__layout {
  display: grid;
  grid-template-columns: minmax(260px, 0.95fr) minmax(340px, 1.05fr);
  gap: 24px;
  align-items: start;
  width: 100%;
  min-width: 0;
}

.gallery-audio-detail--lightbox.gallery-audio-detail--has-timeline .gallery-audio-detail__main,
.gallery-audio-detail--lightbox.gallery-audio-detail--has-timeline .gallery-audio-detail__lyrics-block {
  min-width: 0;
}

.gallery-audio-detail--lightbox.gallery-audio-detail--has-timeline .gallery-audio-detail__lyrics-block {
  margin-top: 0;
}

.gallery-audio-detail .studio-audio-effective-lyrics__head {
  flex-wrap: nowrap;
  align-items: center;
  gap: 6px;
  min-width: 0;
}

.gallery-audio-detail .studio-audio-effective-lyrics__label {
  flex: 0 1 auto;
  min-width: 0;
  white-space: nowrap;
}

.gallery-audio-detail .gallery-audio-detail__lyrics-actions {
  flex: 1 1 auto;
  min-width: 0;
  flex-wrap: nowrap;
  justify-content: flex-end;
}

.gallery-audio-detail .gallery-audio-detail__lyrics-actions :deep(.dq-btn) {
  padding-inline: 4px;
  flex-shrink: 0;
}

.gallery-audio-detail__timeline-badge {
  display: inline-block;
  white-space: nowrap;
  margin-left: 6px;
  padding: 1px 6px;
  border-radius: 999px;
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  letter-spacing: 0;
  text-transform: none;
  vertical-align: 1px;
  color: var(--dq-label-secondary);
  background: var(--dq-fill-tertiary);
}

.gallery-audio-detail__timeline-badge.is-aligned {
  color: var(--dq-accent);
  background: color-mix(in srgb, var(--dq-accent) 14%, transparent);
}

.gallery-audio-detail__scroller {
  margin-top: 6px;
}

.gallery-audio-detail__editor {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.gallery-audio-detail__offset {
  display: grid;
  grid-template-columns: minmax(120px, 180px) auto;
  gap: 8px;
}

.gallery-audio-detail__track {
  position: relative;
  height: 42px;
  border-radius: var(--dq-radius-control);
  background: var(--dq-fill-secondary);
  overflow: hidden;
}

.gallery-audio-detail__segment {
  position: absolute;
  top: 6px;
  bottom: 6px;
  min-width: 8px;
  border-radius: 3px;
  background: color-mix(in srgb, var(--dq-accent) 35%, transparent);
}

.gallery-audio-detail__handle {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 8px;
  padding: 0;
  border: 0;
  background: var(--dq-accent);
  cursor: ew-resize;
  touch-action: none;
}

.gallery-audio-detail__handle--start {
  left: 0;
}

.gallery-audio-detail__handle--end {
  right: 0;
}

.gallery-audio-detail__edit-row {
  display: grid;
  grid-template-columns: 76px 76px minmax(0, 1fr);
  gap: 6px;
}
</style>
