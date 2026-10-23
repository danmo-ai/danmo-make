<!-- @ts-nocheck -->
<template>
  <div
    class="dq-music-player"
    :class="{
      'dq-music-player--compact': compact,
      'dq-music-player--featured': layout === 'featured',
      'is-playing': isPlaying,
      'is-loading': isLoading,
    }"
  >
    <audio
      ref="audioEl"
      class="dq-music-player__audio"
      :src="src"
      preload="auto"
      playsinline
      @loadedmetadata="onLoadedMetadata"
      @durationchange="onDurationChange"
      @timeupdate="onTimeUpdate"
      @ended="onEnded"
      @play="onPlay"
      @pause="onPause"
      @waiting="isLoading = true"
      @canplay="onCanPlay"
      @error="onAudioError"
    />

    <!-- Featured: Now Playing card (create page / gallery detail) -->
    <template v-if="layout === 'featured' && !compact">
      <div class="dq-music-player__ambient" :style="ambientStyle" aria-hidden="true" />

      <div class="dq-music-player__featured-top">
        <button
          type="button"
          class="dq-music-player__art dq-music-player__art--featured dq-music-player__art-btn"
          :class="{ 'is-active': isPlaying }"
          :style="artStyle"
          :disabled="!src"
          :aria-label="isPlaying ? $t('audio.pause') : $t('audio.play')"
          @click.stop="togglePlay"
        >
          <DqIcon class="dq-music-player__art-icon" :size="36"><Headset /></DqIcon>
          <span class="dq-music-player__art-shine" aria-hidden="true" />
        </button>

        <div
          class="dq-music-player__wave"
          :class="{ 'is-playing': isPlaying && !isLoading }"
          aria-hidden="true"
        >
          <span
            v-for="i in waveBars"
            :key="i"
            class="dq-music-player__wave-bar"
            :style="{ '--wave-i': i }"
          />
        </div>
      </div>

      <div class="dq-music-player__featured-meta">
        <h3 class="dq-music-player__title dq-music-player__title--featured" :title="title">
          {{ displayTitle }}
        </h3>
        <p v-if="subtitle" class="dq-music-player__subtitle dq-music-player__subtitle--featured">
          {{ subtitle }}
        </p>
      </div>

      <div class="dq-music-player__scrub dq-music-player__scrub--featured">
        <div
          class="dq-music-player__progress"
          role="slider"
          :aria-valuenow="Math.round(progressPct)"
          aria-valuemin="0"
          aria-valuemax="100"
          :aria-label="$t('audio.seek')"
          @click="onSeek"
        >
          <div class="dq-music-player__progress-track">
            <div class="dq-music-player__progress-buffer" :style="{ width: bufferPct + '%' }" />
            <div class="dq-music-player__progress-fill" :style="{ width: progressPct + '%' }">
              <span class="dq-music-player__progress-knob" aria-hidden="true" />
            </div>
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
          class="dq-music-player__btn-play dq-music-player__btn-play--featured"
          :disabled="!src"
          :aria-busy="isLoading && !isPlaying"
          :aria-label="isPlaying ? $t('audio.pause') : $t('audio.play')"
          @click.stop="togglePlay"
        >
          <DqIcon :size="22" class="dq-music-player__btn-icon">
            <Pause v-if="isPlaying" />
            <Play v-else-if="!isLoading" />
            <Headset v-else />
          </DqIcon>
        </button>
        <DqIconButton
          v-if="showDownload && src"
          type="text"
          size="sm"
          class="dq-music-player__btn-dl dq-music-player__btn-dl--featured"
          :label="$t('audio.download')"
          @click.stop="emit('download')"
        >
          <DqIcon><Download /></DqIcon>
        </DqIconButton>
      </div>
    </template>

    <!-- Inline: compact row (recent list / sidebar) -->
    <template v-else>
      <div class="dq-music-player__hero">
        <div class="dq-music-player__art" :class="{ 'is-active': isPlaying }" :style="artStyle">
          <DqIcon class="dq-music-player__art-icon" :size="compact ? 20 : 28">
            <Headset />
          </DqIcon>
        </div>

        <div class="dq-music-player__meta">
          <h3 class="dq-music-player__title" :title="title">{{ displayTitle }}</h3>
          <p v-if="subtitle" class="dq-music-player__subtitle">{{ subtitle }}</p>
        </div>

        <div class="dq-music-player__hero-actions">
          <button
            type="button"
            class="dq-music-player__btn-play"
            :disabled="!src"
            :aria-label="isPlaying ? $t('audio.pause') : $t('audio.play')"
            @click.stop="togglePlay"
          >
            <DqIcon :size="compact ? 18 : 20" class="dq-music-player__btn-icon">
              <Pause v-if="isPlaying" />
              <Play v-else />
            </DqIcon>
          </button>
          <DqIconButton
            v-if="showDownload && src && !compact"
            type="text"
            size="sm"
            class="dq-music-player__btn-dl"
            :label="$t('audio.download')"
            @click.stop="emit('download')"
          >
            <DqIcon><Download /></DqIcon>
          </DqIconButton>
        </div>
      </div>

      <div v-if="!compact" class="dq-music-player__scrub">
        <div
          class="dq-music-player__progress"
          role="slider"
          :aria-valuenow="Math.round(progressPct)"
          aria-valuemin="0"
          aria-valuemax="100"
          :aria-label="$t('audio.seek')"
          @click="onSeek"
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
    </template>

    <div v-if="$slots.default" class="dq-music-player__extra">
      <slot />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { Download, Headset, Pause, Play } from '@danqing/dq-shell';
import { toast } from '@/utils/feedback';
import { $tt } from '@/utils/i18n';
import { useMediaTimeline } from '@/composables/useMediaTimeline';

const props = defineProps({
  src: { type: String, default: '' },
  title: { type: String, default: '' },
  subtitle: { type: String, default: '' },
  compact: { type: Boolean, default: false },
  layout: { type: String, default: 'inline' },
  showDownload: { type: Boolean, default: true },
  autoplay: { type: Boolean, default: false },
  durationSeconds: { type: Number, default: 0 },
  hue: { type: Number, default: 0 },
});

const emit = defineEmits(['download', 'play', 'pause', 'duration', 'error', 'timeupdate']);

const waveBars = 32;

const audioEl = ref<HTMLAudioElement | null>(null);
const isPlaying = ref(false);
const isLoading = ref(false);
const pendingPlay = ref(false);
const mediaReady = ref(false);
const loadingToastShown = ref(false);

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
} = useMediaTimeline(audioEl, {
  durationHint: () => props.durationSeconds || null,
});

const displayTitle = computed(() => {
  const t = (props.title || '').trim();
  return t || '—';
});

const progressPct = computed(() => {
  if (!duration.value || duration.value <= 0) return 0;
  return Math.min(100, (currentTime.value / duration.value) * 100);
});

const bufferPct = computed(() => {
  if (!duration.value || duration.value <= 0) return 0;
  return Math.min(100, (bufferEnd.value / duration.value) * 100);
});

const artStyle = computed(() => ({
  '--dq-music-hue': String(props.hue),
}));

const ambientStyle = computed(() => ({
  '--dq-music-hue': String(props.hue),
  opacity: isPlaying.value ? 1 : 0.72,
}));

function formatClock(sec: number) {
  const s = Math.max(0, Math.floor(sec || 0));
  const m = Math.floor(s / 60);
  return m + ':' + String(s % 60).padStart(2, '0');
}

function onLoadedMetadata() {
  const el = audioEl.value;
  if (!el) return;
  syncFromElement();
  if (duration.value > 0) emit('duration', duration.value);
  if (el.readyState >= HTMLMediaElement.HAVE_FUTURE_DATA) {
    mediaReady.value = true;
    isLoading.value = false;
    loadingToastShown.value = false;
    flushPendingPlay(el);
  }
}

function onEnded() {
  isPlaying.value = false;
  onTimelineEnded();
  emit('pause');
}

function onPlay() {
  isPlaying.value = true;
  isLoading.value = false;
  pendingPlay.value = false;
  onTimelinePlay();
  emit('play');
}

function onPause() {
  isPlaying.value = false;
  onTimelinePause();
  emit('pause');
}

function onSeek(ev: MouseEvent) {
  const el = audioEl.value;
  const track = ev.currentTarget as HTMLElement;
  if (!el || !duration.value || !track) return;
  const rect = track.getBoundingClientRect();
  const ratio = Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width));
  el.currentTime = ratio * duration.value;
  syncFromElement();
}

function onCanPlay() {
  const el = audioEl.value;
  if (!el) return;
  mediaReady.value = true;
  isLoading.value = false;
  loadingToastShown.value = false;
  flushPendingPlay(el);
}

function flushPendingPlay(el: HTMLAudioElement) {
  if (!pendingPlay.value) return;
  if (el.readyState < HTMLMediaElement.HAVE_FUTURE_DATA) return;
  requestPlay(el, false);
}

function onAudioError() {
  const el = audioEl.value;
  const code = el?.error?.code;
  pendingPlay.value = false;
  isLoading.value = false;
  isPlaying.value = false;
  const msg =
    code === MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED
      ? $tt('audio.playbackFormatUnsupported')
      : code === MediaError.MEDIA_ERR_NETWORK
        ? $tt('audio.playbackLoadFailed')
        : $tt('audio.playbackFailed');
  emit('error', msg);
  toast.error(msg);
}

/** play() must run synchronously inside the click handler (macOS WebKit user activation). */
function requestPlay(el: HTMLAudioElement, fromUserGesture: boolean) {
  if (!props.src) {
    pendingPlay.value = false;
    return;
  }
  const p = el.play();
  if (!p) {
    pendingPlay.value = false;
    return;
  }
  p.then(() => {
    pendingPlay.value = false;
    loadingToastShown.value = false;
  }).catch(() => {
    pendingPlay.value = false;
    if (fromUserGesture) {
      if (el.readyState < HTMLMediaElement.HAVE_FUTURE_DATA) {
        pendingPlay.value = true;
        isLoading.value = true;
        if (!loadingToastShown.value) {
          loadingToastShown.value = true;
          toast.info($tt('audio.playbackLoading'));
        }
      } else {
        toast.error($tt('audio.playbackFailed'));
      }
      return;
    }
    toast.info($tt('audio.playbackTapAgain'));
  });
}

function togglePlay() {
  const el = audioEl.value;
  if (!el || !props.src) return;
  if (!el.paused) {
    el.pause();
    pendingPlay.value = false;
    isLoading.value = false;
    return;
  }
  if (el.readyState < HTMLMediaElement.HAVE_FUTURE_DATA) {
    pendingPlay.value = true;
    isLoading.value = true;
    if (!loadingToastShown.value) {
      loadingToastShown.value = true;
      toast.info($tt('audio.playbackLoading'));
    }
    return;
  }
  pendingPlay.value = true;
  requestPlay(el, true);
}

function playWhenReady() {
  const el = audioEl.value;
  if (!el || !props.src) return;
  pendingPlay.value = true;
  if (el.readyState >= HTMLMediaElement.HAVE_FUTURE_DATA) {
    requestPlay(el, true);
  } else {
    isLoading.value = true;
  }
}

function pause() {
  pendingPlay.value = false;
  audioEl.value?.pause();
}

function resetForNewSrc(clearPending: boolean) {
  resetTimeline();
  isPlaying.value = false;
  mediaReady.value = false;
  isLoading.value = !!props.src;
  loadingToastShown.value = false;
  if (clearPending) pendingPlay.value = false;
}

function load(clearPending = true) {
  resetForNewSrc(clearPending);
  const el = audioEl.value;
  if (!el || !props.src) return;
  try {
    el.load();
  } catch {
    /* ignore */
  }
}

watch(
  () => props.src,
  (url) => {
    const wantAutoplay = !!props.autoplay && !!url;
    resetForNewSrc(!wantAutoplay);
    if (wantAutoplay) pendingPlay.value = true;
  },
);

watch(
  () => props.durationSeconds,
  () => syncFromElement(),
);

watch(currentTime, (t) => emit('timeupdate', t));

function seek(seconds: number) {
  const el = audioEl.value;
  if (!el || !Number.isFinite(seconds)) return;
  const max = duration.value > 0 ? duration.value : Number.POSITIVE_INFINITY;
  el.currentTime = Math.min(max, Math.max(0, seconds));
  syncFromElement();
}

defineExpose({ pause, load, togglePlay, playWhenReady, seek, isPlaying });
</script>
