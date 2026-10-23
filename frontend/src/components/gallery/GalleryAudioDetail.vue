<!-- @ts-nocheck -->
<template>
  <div
    class="gallery-audio-detail"
    :class="[`gallery-audio-detail--${variant}`, { 'gallery-audio-detail--has-timeline': hasTimeline }]"
  >
    <div class="gallery-audio-detail__layout">
      <div class="gallery-audio-detail__main">
        <AudioMusicPlayer
          ref="playerRef"
          :key="item.path"
          :src="src"
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

        <template v-if="variant === 'lightbox'">
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
              {{ isAligned ? $t('gallery.lyricsAligned') : $t('gallery.lyricsEstimated') }}
            </span>
          </span>
          <div v-if="lyricsText" class="gallery-audio-detail__lyrics-actions">
            <DqButton
              v-if="hasTimeline"
              type="text"
              size="sm"
              @click="showFullText = !showFullText"
            >
              {{ showFullText ? $t('gallery.lyricsShowSynced') : $t('gallery.lyricsShowFull') }}
            </DqButton>
            <DqButton type="text" size="sm" @click="copyLyrics">
              {{ $t('gallery.copy') }}
            </DqButton>
            <DqButton type="text" size="sm" @click="downloadLyrics">
              {{ $t('audio.downloadLyrics') }}
            </DqButton>
            <DqButton v-if="hasTimeline" type="text" size="sm" @click="downloadLrc">
              {{ $t('gallery.downloadLrc') }}
            </DqButton>
          </div>
        </div>
        <AudioLyricsScroller
          v-if="hasTimeline && !showFullText"
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
import { $tt } from '@/utils/i18n';
import { linesFromSegments, linesToLrc } from '@/utils/lyricsTimeline';

const props = defineProps({
  item: { type: Object as () => GalleryItem, required: true },
  src: { type: String, required: true },
  variant: { type: String, default: 'sidebar' },
  durationLabel: { type: String, default: '' },

});

const emit = defineEmits(['download']);

const { t: $t } = useI18n();

const playerRef = ref<InstanceType<typeof AudioMusicPlayer> | null>(null);
const currentTime = ref(0);
const showFullText = ref(false);

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

const timelineLines = computed(() =>
  linesFromSegments(props.item.metadata?.lyrics_segments, durationSeconds.value),
);
const hasTimeline = computed(() => timelineLines.value.length > 0);

watch(
  () => props.item.path,
  () => {
    currentTime.value = 0;
    showFullText.value = false;
  },
);

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

.gallery-audio-detail--lightbox {
  container-type: inline-size;
}

.gallery-audio-detail__layout {
  display: block;
}

/* Pop-player layout: Now Playing on the left, synced lyrics on the right. */
@container (min-width: 680px) {
  .gallery-audio-detail--has-timeline .gallery-audio-detail__layout {
    display: grid;
    grid-template-columns: minmax(260px, 0.9fr) minmax(340px, 1.1fr);
    gap: 24px;
    align-items: start;
  }

  .gallery-audio-detail--has-timeline .gallery-audio-detail__lyrics-block {
    margin-top: 0;
  }
}

.gallery-audio-detail .studio-audio-effective-lyrics__head {
  flex-wrap: wrap;
  row-gap: 4px;
}

.gallery-audio-detail .gallery-audio-detail__lyrics-actions {
  flex-wrap: wrap;
}

.gallery-audio-detail__timeline-badge {
  display: inline-block;
  white-space: nowrap;
  margin-left: 6px;
  padding: 1px 6px;
  border-radius: 999px;
  font-size: 10px;
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
</style>
