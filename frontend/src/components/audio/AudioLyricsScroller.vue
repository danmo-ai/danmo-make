<template>
  <div
    class="dq-lyrics-scroller"
    :class="{
      'dq-lyrics-scroller--compact': compact,
      'is-following': !userHold,
    }"
  >
    <div ref="viewport" class="dq-lyrics-scroller__viewport" @wheel.passive="holdAutoFollow"
      @touchmove.passive="holdAutoFollow" @keydown="onViewportKey">
      <div class="dq-lyrics-scroller__pad" aria-hidden="true" />
      <button
        v-for="(line, i) in lines"
        :key="i + ':' + line.start"
        :ref="(el) => setLineEl(i, el)"
        type="button"
        class="dq-lyrics-scroller__line"
        :class="{
          'is-active': i === activeIdx,
          'is-past': i < activeIdx,
          'is-near': Math.abs(i - activeIdx) === 1,
        }"
        :aria-current="i === activeIdx ? 'true' : undefined"
        :title="formatClock(line.start)"
        @click="onLineClick(line)"
      >
        <span
          class="dq-lyrics-scroller__text"
          :class="{ 'is-karaoke': karaoke && i === activeIdx }"
          :style="i === activeIdx && karaoke ? { '--dq-lyric-fill': fillPct + '%' } : undefined"
        >{{ line.text }}</span>
      </button>
      <div class="dq-lyrics-scroller__pad" aria-hidden="true" />
    </div>
    <button
      v-if="userHold && activeIdx >= 0"
      type="button"
      class="dq-lyrics-scroller__recenter"
      @click="resumeAutoFollow"
    >
      {{ $t('gallery.lyricsBackToCurrent') }}
    </button>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, watch, type PropType } from 'vue';
import { activeLineIndex, lineProgress, type LyricLine } from '@/utils/lyricsTimeline';

const props = defineProps({
  lines: { type: Array as PropType<LyricLine[]>, default: () => [] },
  currentTime: { type: Number, default: 0 },
  compact: { type: Boolean, default: false },
  /** Fill the active line left→right over its duration (only meaningful for aligned timelines). */
  karaoke: { type: Boolean, default: false },
});

const emit = defineEmits<{ (e: 'seek', seconds: number): void }>();

const HOLD_MS = 3000;

const viewport = ref<HTMLElement | null>(null);
const lineEls: HTMLElement[] = [];
const userHold = ref(false);
let holdTimer: ReturnType<typeof setTimeout> | null = null;

const activeIdx = computed(() => activeLineIndex(props.lines, props.currentTime));
const fillPct = computed(() =>
  Math.round(lineProgress(props.lines[activeIdx.value], props.currentTime) * 1000) / 10,
);

function setLineEl(i: number, el: unknown) {
  if (el instanceof HTMLElement) lineEls[i] = el;
}

function prefersReducedMotion(): boolean {
  return typeof window !== 'undefined' && !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
}

function centerActive(smooth = true) {
  const vp = viewport.value;
  const idx = activeIdx.value;
  const el = idx >= 0 ? lineEls[idx] : null;
  if (!vp) return;
  const top = el ? el.offsetTop - vp.clientHeight / 2 + el.clientHeight / 2 : 0;
  vp.scrollTo({ top: Math.max(0, top), behavior: smooth && !prefersReducedMotion() ? 'smooth' : 'auto' });
}

function holdAutoFollow() {
  userHold.value = true;
  if (holdTimer) clearTimeout(holdTimer);
  holdTimer = setTimeout(resumeAutoFollow, HOLD_MS);
}

function resumeAutoFollow() {
  if (holdTimer) clearTimeout(holdTimer);
  holdTimer = null;
  userHold.value = false;
  centerActive();
}

function onViewportKey(ev: KeyboardEvent) {
  if (['ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'Home', 'End'].includes(ev.key)) holdAutoFollow();
}

function onLineClick(line: LyricLine) {
  emit('seek', line.start);
  resumeAutoFollow();
}

function formatClock(sec: number) {
  const s = Math.max(0, Math.floor(sec || 0));
  return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0');
}

watch(activeIdx, () => {
  if (!userHold.value) centerActive();
});

watch(
  () => props.lines,
  () => {
    lineEls.length = 0;
    nextTick(() => centerActive(false));
  },
  { immediate: true },
);

onBeforeUnmount(() => {
  if (holdTimer) clearTimeout(holdTimer);
});
</script>

<style scoped>
.dq-lyrics-scroller {
  position: relative;
  --dq-lyric-line-gap: 6px;
}

.dq-lyrics-scroller__viewport {
  position: relative;
  height: 360px;
  overflow-y: auto;
  scrollbar-width: none;
  mask-image: linear-gradient(to bottom, transparent 0, #000 18%, #000 82%, transparent 100%);
  -webkit-mask-image: linear-gradient(to bottom, transparent 0, #000 18%, #000 82%, transparent 100%);
}

.dq-lyrics-scroller__viewport::-webkit-scrollbar {
  display: none;
}

.dq-lyrics-scroller--compact .dq-lyrics-scroller__viewport {
  height: 112px;
}

.dq-lyrics-scroller__pad {
  height: 45%;
}

.dq-lyrics-scroller__line {
  display: block;
  width: 100%;
  margin: var(--dq-lyric-line-gap) 0;
  padding: 6px 10px;
  border: 0;
  border-radius: var(--dq-radius-control, 8px);
  background: transparent;
  text-align: left;
  cursor: pointer;
  color: var(--dq-label-tertiary);
  font-size: calc(var(--dq-font-size-body, 14px) + 3px);
  font-weight: 600;
  line-height: 1.45;
  letter-spacing: var(--dq-tracking-tight, 0);
  opacity: 0.55;
  transform-origin: left center;
  transition:
    opacity 240ms ease,
    transform 240ms ease,
    color 240ms ease;
}

.dq-lyrics-scroller--compact .dq-lyrics-scroller__line {
  padding: 2px 6px;
  margin: 2px 0;
  font-size: var(--dq-font-size-body, 14px);
}

.dq-lyrics-scroller__line:hover {
  opacity: 0.85;
  background: var(--dq-fill-tertiary);
}

.dq-lyrics-scroller__line.is-past {
  opacity: 0.4;
}

.dq-lyrics-scroller__line.is-near {
  opacity: 0.72;
}

.dq-lyrics-scroller__line.is-active {
  opacity: 1;
  color: var(--dq-label-primary);
  transform: scale(1.06);
}

.dq-lyrics-scroller--compact .dq-lyrics-scroller__line.is-active {
  transform: scale(1.03);
}

.dq-lyrics-scroller__line:focus-visible {
  outline: 2px solid var(--dq-accent);
  outline-offset: 1px;
}

.dq-lyrics-scroller__text.is-karaoke {
  background-image: linear-gradient(
    90deg,
    var(--dq-accent) var(--dq-lyric-fill, 0%),
    var(--dq-label-primary) var(--dq-lyric-fill, 0%)
  );
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

.dq-lyrics-scroller__recenter {
  position: absolute;
  right: 8px;
  bottom: 8px;
  padding: 4px 10px;
  border: 0.5px solid var(--dq-border-subtle);
  border-radius: 999px;
  background: var(--dq-bg-base);
  color: var(--dq-label-secondary);
  font-size: var(--dq-font-size-caption, 12px);
  cursor: pointer;
  box-shadow: var(--dq-shadow-md, none);
}

@media (prefers-reduced-motion: reduce) {
  .dq-lyrics-scroller__line {
    transition: none;
  }
}
</style>
