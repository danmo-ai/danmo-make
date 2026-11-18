<!-- @ts-nocheck -->
<template>
  <div
    class="dq-music-player dq-media-player dq-media-player--video"
    :class="{
      'is-playing': isPlaying,
      'is-loading': isLoading,
      'dq-media-player--gallery': layout === 'gallery',
      'dq-media-player--create': layout === 'create',
    }"
    :style="playerStyle"
  >
    <div class="dq-media-player__screen">
      <video
        ref="videoEl"
        class="dq-media-player__video"
        :src="src"
        preload="metadata"
        playsinline
        @loadedmetadata="onLoadedMetadata"
        @durationchange="onDurationChange"
        @timeupdate="onTimeUpdate"
        @ended="onEnded"
        @play="onPlay"
        @pause="onPause"
        @waiting="isLoading = true"
        @canplay="isLoading = false"
      />
      <div class="dq-media-player__screen-vignette" />
      <button
        type="button"
        class="dq-media-player__screen-play"
        :disabled="!src"
        :aria-label="isPlaying ? $t('audio.pause') : $t('audio.play')"
        @click="togglePlay"
      >
        <DqIcon :size="36">
          <Pause v-if="isPlaying" />
          <Play v-else />
        </DqIcon>
      </button>
    </div>

    <div v-if="layout === 'create'" class="dq-media-player__meta dq-media-player__meta--below">
      <p class="dq-music-player__eyebrow">{{ $t('studio.previewNow') }}</p>
      <h3 class="dq-music-player__title" :title="title">{{ displayTitle }}</h3>
      <p v-if="subtitle" class="dq-music-player__subtitle">{{ subtitle }}</p>
    </div>

    <div
      class="dq-music-player__scrub"
      :class="{ 'dq-music-player__scrub--featured': layout === 'gallery' }"
    >
      <div
        class="dq-music-player__progress"
        role="slider"
        :aria-valuenow="Math.round(progressPct)"
        aria-valuemin="0"
        aria-valuemax="100"
        :aria-label="$t('audio.seek')"
        tabindex="0"
        @click="onSeek"
        @keydown="onSeekKeydown"
      >
        <div class="dq-music-player__progress-track">
          <div class="dq-music-player__progress-buffer" :style="{ width: bufferPct + '%' }" />
          <div class="dq-music-player__progress-fill" :style="{ width: progressPct + '%' }" />
        </div>
      </div>
      <div class="dq-music-player__times">
        <span>{{ formatClock(currentTime) }}</span>
        <span>{{ formatClock(duration) }}</span>
      </div>
    </div>

    <div class="dq-music-player__transport">
      <button
        type="button"
        class="dq-music-player__btn-play"
        :class="{ 'dq-music-player__btn-play--featured': layout === 'gallery' }"
        :disabled="!src"
        :aria-label="isPlaying ? $t('audio.pause') : $t('audio.play')"
        @click="togglePlay"
      >
        <DqIcon :size="layout === 'gallery' ? 22 : 26" class="dq-music-player__btn-icon">
          <Pause v-if="isPlaying" />
          <Play v-else />
        </DqIcon>
      </button>
      <DqIconButton
        v-if="showDownload && src"
        type="text"
        size="sm"
        class="dq-music-player__btn-dl"
        :class="{ 'dq-music-player__btn-dl--featured': layout === 'gallery' }"
        :label="$t('gallery.download')"
        @click="emit('download')"
      >
        <DqIcon><Download /></DqIcon>
      </DqIconButton>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { Download, Pause, Play } from '@danqing/dq-shell';
import { useMediaTimeline } from '@/composables/useMediaTimeline';

const props = defineProps({
  src: { type: String, default: '' },
  title: { type: String, default: '' },
  subtitle: { type: String, default: '' },
  layout: { type: String, default: 'create' },
  aspectWidth: { type: Number, default: 0 },
  aspectHeight: { type: Number, default: 0 },
  durationSeconds: { type: Number, default: 0 },
  showDownload: { type: Boolean, default: true },
  hue: { type: Number, default: 268 },
});

const emit = defineEmits(['download', 'play', 'pause', 'duration']);

const videoEl = ref<HTMLVideoElement | null>(null);
const isPlaying = ref(false);
const isLoading = ref(false);
const intrinsicAspect = ref<{ w: number; h: number } | null>(null);

const {
  currentTime,
  duration,
  bufferEnd,
  resetTimeline,
  onTimeUpdate,
  onDurationChange,
  onPlay: onTimelinePlay,
  onPause: onTimelinePause,
  onEnded: onTimelineEnded,
  syncFromElement,
} = useMediaTimeline(videoEl, {
  durationHint: () => props.durationSeconds || null,
});

const displayTitle = computed(() => (props.title || '').trim() || 'Untitled');

const playerStyle = computed(() => {
  const styles: Record<string, string> = {
    '--dq-music-hue': String(props.hue),
  };
  const w = props.aspectWidth || intrinsicAspect.value?.w || 0;
  const h = props.aspectHeight || intrinsicAspect.value?.h || 0;
  if (w > 0 && h > 0) {
    styles['--dq-video-aspect-ratio'] = `${w} / ${h}`;
  }
  return styles;
});

const progressPct = computed(() => {
  if (!duration.value) return 0;
  return Math.min(100, (currentTime.value / duration.value) * 100);
});

const bufferPct = computed(() => {
  if (!duration.value) return 0;
  return Math.min(100, (bufferEnd.value / duration.value) * 100);
});

function formatClock(sec: number) {
  const s = Math.max(0, Math.floor(sec || 0));
  const m = Math.floor(s / 60);
  return m + ':' + String(s % 60).padStart(2, '0');
}

function onLoadedMetadata() {
  const el = videoEl.value;
  if (!el) return;
  syncFromElement();
  isLoading.value = false;
  if (el.videoWidth > 0 && el.videoHeight > 0) {
    intrinsicAspect.value = { w: el.videoWidth, h: el.videoHeight };
  }
  if (duration.value > 0) emit('duration', duration.value);
}

function onEnded() {
  isPlaying.value = false;
  onTimelineEnded();
  emit('pause');
}

function onPlay() {
  isPlaying.value = true;
  onTimelinePlay();
  emit('play');
}

function onPause() {
  isPlaying.value = false;
  onTimelinePause();
  emit('pause');
}

function onSeek(ev: MouseEvent) {
  const el = videoEl.value;
  const track = ev.currentTarget as HTMLElement;
  if (!el || !duration.value || !track) return;
  const rect = track.getBoundingClientRect();
  const ratio = Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width));
  el.currentTime = ratio * duration.value;
  syncFromElement();
}

function onSeekKeydown(ev: KeyboardEvent) {
  const el = videoEl.value;
  if (!el || !duration.value) return;
  let next = currentTime.value;
  if (ev.key === 'ArrowLeft' || ev.key === 'ArrowDown') next -= 5;
  else if (ev.key === 'ArrowRight' || ev.key === 'ArrowUp') next += 5;
  else if (ev.key === 'Home') next = 0;
  else if (ev.key === 'End') next = duration.value;
  else return;
  ev.preventDefault();
  el.currentTime = Math.min(duration.value, Math.max(0, next));
  syncFromElement();
}

function togglePlay() {
  const el = videoEl.value;
  if (!el || !props.src) return;
  if (isPlaying.value) el.pause();
  else el.play().catch(() => {});
}

function load() {
  const el = videoEl.value;
  if (!el) return;
  resetTimeline();
  isPlaying.value = false;
  intrinsicAspect.value = null;
  try {
    el.load();
  } catch {
    /* ignore */
  }
}

watch(
  () => props.src,
  () => load(),
);

watch(
  () => props.durationSeconds,
  () => syncFromElement(),
);

defineExpose({ load, togglePlay, pause: () => videoEl.value?.pause() });
</script>
