<template>
  <div class="audio-waveform" :class="{ 'is-selectable': selectable }">
    <canvas
      ref="canvasRef"
      class="audio-waveform__canvas"
      role="slider"
      tabindex="0"
      :aria-label="$t('audio.seek')"
      :aria-valuemin="0"
      :aria-valuemax="Math.round(duration)"
      :aria-valuenow="Math.round(currentTime)"
      @pointerdown="onPointerDown"
      @keydown="onKeydown"
    />
    <span v-if="loading" class="audio-waveform__loading">{{ $t('audio.waveformLoading') }}</span>
    <span v-if="error" class="audio-waveform__loading">{{ error }}</span>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { loadAudioPeaks } from '@/utils/audioWaveform';
import { $tt } from '@/utils/i18n';

const props = withDefaults(defineProps<{
  src: string;
  currentTime?: number;
  duration?: number;
  regionStart?: number | null;
  regionEnd?: number | null;
  selectable?: boolean;
}>(), {
  currentTime: 0,
  duration: 0,
  regionStart: null,
  regionEnd: null,
  selectable: false,
});
const emit = defineEmits<{
  seek: [seconds: number];
  'update:region': [region: { start: number; end: number }];
}>();

const canvasRef = ref<HTMLCanvasElement | null>(null);
const peaks = ref<Float32Array | null>(null);
const loading = ref(false);
const error = ref('');
let ticket = 0;
let observer: ResizeObserver | null = null;
let drag: { start: number; pointer: number } | null = null;

const hasRegion = computed(() =>
  props.regionStart != null && props.regionEnd != null && props.regionEnd > props.regionStart,
);

async function load() {
  const current = ++ticket;
  peaks.value = null;
  error.value = '';
  if (!props.src) return;
  loading.value = true;
  try {
    const decoded = await loadAudioPeaks(props.src);
    if (current === ticket) peaks.value = decoded;
  } catch (cause) {
    if (current === ticket) error.value = cause instanceof Error ? cause.message : String(cause);
  } finally {
    if (current === ticket) loading.value = false;
  }
}

function color(name: string, fallback: string): string {
  return getComputedStyle(canvasRef.value!).getPropertyValue(name).trim() || fallback;
}

function draw() {
  const canvas = canvasRef.value;
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * ratio));
  const height = Math.max(1, Math.round(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, width, height);
  const values = peaks.value;
  if (!values?.length) {
    ctx.strokeStyle = color('--dq-border-subtle', '#666');
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();
    return;
  }
  const duration = Math.max(0, props.duration || 0);
  const regionX = hasRegion.value && duration > 0
    ? [props.regionStart! / duration * width, props.regionEnd! / duration * width]
    : null;
  if (regionX) {
    ctx.fillStyle = color('--dq-fill-tertiary', '#333');
    ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = color('--dq-accent', '#888');
    ctx.globalAlpha = 0.12;
    ctx.fillRect(regionX[0], 0, regionX[1] - regionX[0], height);
    ctx.globalAlpha = 1;
  }
  ctx.strokeStyle = color('--dq-accent', '#888');
  ctx.lineWidth = Math.max(1, ratio);
  ctx.beginPath();
  const middle = height / 2;
  const maxHeight = height * 0.44;
  for (let x = 0; x < width; x += 1) {
    const peak = values[Math.min(values.length - 1, Math.floor(x / width * values.length))] || 0;
    const amplitude = Math.max(ratio, peak * maxHeight);
    ctx.moveTo(x + 0.5, middle - amplitude);
    ctx.lineTo(x + 0.5, middle + amplitude);
  }
  ctx.stroke();
  if (duration > 0) {
    const playhead = Math.min(width, Math.max(0, props.currentTime / duration * width));
    ctx.strokeStyle = color('--dq-label-primary', '#fff');
    ctx.lineWidth = ratio * 1.5;
    ctx.beginPath();
    ctx.moveTo(playhead, 0);
    ctx.lineTo(playhead, height);
    ctx.stroke();
  }
  if (regionX) {
    ctx.strokeStyle = color('--dq-accent', '#888');
    ctx.lineWidth = ratio * 2;
    for (const x of regionX) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
  }
}

function secondsAt(event: PointerEvent): number {
  const canvas = canvasRef.value!;
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(1, Math.max(0, (event.clientX - rect.left) / Math.max(1, rect.width)));
  return ratio * Math.max(0, props.duration || 0);
}

function onPointerDown(event: PointerEvent) {
  if (props.duration <= 0) return;
  const canvas = canvasRef.value!;
  canvas.setPointerCapture(event.pointerId);
  const time = secondsAt(event);
  if (!props.selectable) {
    emit('seek', time);
    return;
  }
  drag = { start: time, pointer: event.pointerId };
  emit('update:region', { start: time, end: Math.min(props.duration, time + Math.max(0.04, props.duration * 0.01)) });
  canvas.onpointermove = (move) => {
    if (!drag) return;
    const current = secondsAt(move);
    emit('update:region', {
      start: Math.min(drag.start, current),
      end: Math.max(drag.start, current),
    });
  };
  canvas.onpointerup = canvas.onpointercancel = () => {
    drag = null;
    canvas.onpointermove = null;
  };
}

function onKeydown(event: KeyboardEvent) {
  let next = props.currentTime;
  if (event.key === 'ArrowLeft') next -= 5;
  else if (event.key === 'ArrowRight') next += 5;
  else if (event.key === 'Home') next = 0;
  else if (event.key === 'End') next = props.duration;
  else return;
  event.preventDefault();
  emit('seek', Math.min(props.duration, Math.max(0, next)));
}

watch([() => props.src], load, { immediate: true });
watch([peaks, () => props.currentTime, () => props.duration, () => props.regionStart, () => props.regionEnd], draw);
onMounted(() => {
  observer = new ResizeObserver(draw);
  if (canvasRef.value) observer.observe(canvasRef.value);
  draw();
});
onBeforeUnmount(() => {
  ticket += 1;
  observer?.disconnect();
});
</script>

<style scoped>
.audio-waveform {
  position: relative;
  width: 100%;
  min-width: 0;
}

.audio-waveform__canvas {
  display: block;
  width: 100%;
  height: 88px;
  border-radius: var(--dq-radius-control);
  background: var(--dq-fill-secondary);
  touch-action: none;
}

.audio-waveform.is-selectable .audio-waveform__canvas {
  cursor: crosshair;
}

.audio-waveform__canvas:focus-visible {
  outline: 2px solid var(--dq-focus-ring);
  outline-offset: 2px;
}

.audio-waveform__loading {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  padding: 8px;
  color: var(--dq-label-secondary);
  font-size: var(--dq-font-size-caption);
  pointer-events: none;
}
</style>
