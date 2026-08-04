<template>
  <div
    ref="layoutRef"
    class="studio-layout"
    :class="{
      'studio-layout--composer-collapsed': collapsible && composerCollapsed,
      'studio-layout--drawer-mode': hideComposerBar,
    }"
    :style="layoutStyle"
  >
    <!-- Filter bar -->
    <header class="studio-filter-bar dq-glass--bar">
      <div class="studio-filter-bar__inner">
        <slot name="filters" />
      </div>
    </header>

    <!-- Canvas area -->
    <div ref="canvasRef" class="studio-canvas-area" :class="{ 'studio-canvas-area--freeform': freeform }" @scroll="onScroll">
      <slot name="canvas" />
    </div>

    <!-- Composer bar; hidden in unified drawer mode -->
    <div
      v-if="!hideComposerBar"
      class="studio-composer-bar"
      :class="{ 'studio-composer-bar--collapsed': collapsible && composerCollapsed }"
    >
      <div class="studio-composer-bar__scrim" aria-hidden="true" />
      <button
        v-if="collapsible"
        type="button"
        class="studio-composer-bar__toggle dq-glass--popover"
        :aria-label="composerCollapsed ? $t('canvas.composerExpand') : $t('canvas.composerCollapse')"
        :title="composerCollapsed ? $t('canvas.composerExpand') : $t('canvas.composerCollapse')"
        @click="toggleComposerCollapsed"
      >
        <span class="studio-composer-bar__toggle-icon" aria-hidden="true">
          {{ composerCollapsed ? '▲' : '▼' }}
        </span>
      </button>
      <div class="studio-composer-bar__inner">
        <slot name="composer" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount } from 'vue';
import { useI18n } from 'vue-i18n';
import {
  COMPOSER_RESERVE_CSS_DRAWER_MODE,
  COMPOSER_SCRIM_CSS_DRAWER_MODE,
  COMPOSER_RESERVE_CSS_COLLAPSED,
  COMPOSER_RESERVE_CSS_EXPANDED,
  COMPOSER_SCRIM_CSS_COLLAPSED,
  COMPOSER_SCRIM_CSS_EXPANDED,
  composerReservePx,
} from '@/utils/composerReserve';

const props = defineProps<{
  freeform?: boolean;
  collapsible?: boolean;
  composerCollapsed?: boolean;
  hideComposerBar?: boolean;
  /** Taller composer reserve for dense toolbars. */
  composerTall?: boolean;
}>();

const emit = defineEmits<{
  (e: 'scroll', event: Event): void;
  (e: 'update:composerCollapsed', value: boolean): void;
}>();

const { t: $t } = useI18n();

const canvasRef = ref<HTMLElement | null>(null);
const layoutRef = ref<HTMLElement | null>(null);
const viewportHeight = ref(typeof window !== 'undefined' ? window.innerHeight : 800);

const composerCollapsed = computed(() => props.composerCollapsed === true);

const layoutStyle = computed(() => {
  const collapsed = props.collapsible && composerCollapsed.value;
  const vh = viewportHeight.value;
  const tall = Boolean(props.composerTall) && !collapsed;
  const drawerMode = Boolean(props.hideComposerBar);
  return {
    '--dq-composer-reserve': drawerMode
      ? COMPOSER_RESERVE_CSS_DRAWER_MODE
      : collapsed
        ? COMPOSER_RESERVE_CSS_COLLAPSED
        : tall
          ? COMPOSER_RESERVE_CSS_EXPANDED
          : COMPOSER_RESERVE_CSS_EXPANDED,
    '--dq-composer-scrim-height': drawerMode
      ? COMPOSER_SCRIM_CSS_DRAWER_MODE
      : collapsed
        ? COMPOSER_SCRIM_CSS_COLLAPSED
        : tall
          ? COMPOSER_SCRIM_CSS_EXPANDED
          : COMPOSER_SCRIM_CSS_EXPANDED,
    '--dq-composer-reserve-px': `${composerReservePx(vh, collapsed, tall, drawerMode)}px`,
  };
});

function onResize() {
  viewportHeight.value = window.innerHeight;
}

function onScroll(event: Event) {
  if (!props.freeform) {
    emit('scroll', event);
  }
}

function toggleComposerCollapsed() {
  emit('update:composerCollapsed', !composerCollapsed.value);
}

function scrollToTop() {
  canvasRef.value?.scrollTo({ top: 0, behavior: 'smooth' });
}

onMounted(() => {
  window.addEventListener('resize', onResize);
});

onBeforeUnmount(() => {
  window.removeEventListener('resize', onResize);
});

defineExpose({ scrollToTop, canvasRef, layoutRef });
</script>

<style scoped>
.studio-layout {
  display: flex;
  flex-direction: column;
  height: 100vh;
  width: 100%;
  position: relative;
  overflow: hidden;
  background-color: var(--dq-bg-page);
  background-image:
    radial-gradient(ellipse 120% 420px at 50% -8%, rgba(10, 132, 255, 0.07) 0%, transparent 55%),
    var(--dq-bg-page-glow);
}

.studio-filter-bar {
  flex-shrink: 0;
  z-index: 10;
}

.studio-filter-bar__inner {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  min-width: 0;
  padding: 10px 12px;
  box-sizing: border-box;
  overflow-x: auto;
  overflow-y: hidden;
  scrollbar-width: none;
}

.studio-filter-bar__inner::-webkit-scrollbar {
  display: none;
}

.studio-canvas-area {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  width: 100%;
  min-width: 0;
  box-sizing: border-box;
  padding: 16px 12px 220px;
  scroll-behavior: smooth;
}

.studio-layout--drawer-mode .studio-canvas-area:not(.studio-canvas-area--freeform) {
  padding-bottom: 88px;
}

.studio-canvas-area--freeform {
  overflow: hidden;
  padding: 0;
}

.studio-composer-bar {
  position: fixed;
  bottom: 0;
  left: var(--dq-shell-sidebar-width, 60px);
  right: 0;
  padding: 0 24px 24px;
  z-index: 100;
  pointer-events: none;
}

.studio-composer-bar__scrim {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  height: var(--dq-composer-scrim-height, min(220px, 38vh));
  pointer-events: none;
  background: linear-gradient(
    to top,
    color-mix(in srgb, var(--dq-bg-page) 96%, transparent) 0%,
    color-mix(in srgb, var(--dq-bg-page) 72%, transparent) 42%,
    transparent 100%
  );
  transition: height 0.2s ease;
}

.studio-composer-bar__toggle {
  position: absolute;
  left: 50%;
  top: -14px;
  transform: translateX(-50%);
  z-index: 2;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 22px;
  padding: 0;
  border: 1px solid var(--dq-border-subtle);
  border-radius: 999px;
  background: var(--dq-surface);
  color: var(--dq-label-secondary);
  cursor: pointer;
  pointer-events: auto;
  transition: color 0.15s ease, border-color 0.15s ease;
}

.studio-composer-bar__toggle:hover {
  color: var(--dq-label-primary);
  border-color: var(--dq-border-strong);
}

.studio-composer-bar__toggle-icon {
  font-size: var(--dq-font-size-caption);
  line-height: 1;
}

.studio-composer-bar__inner {
  position: relative;
  max-width: 920px;
  margin: 0 auto;
  pointer-events: auto;
  transition: opacity 0.15s ease;
}

.studio-composer-bar--collapsed {
  padding-bottom: 16px;
}

.studio-canvas-area::-webkit-scrollbar {
  width: 6px;
}

.studio-canvas-area::-webkit-scrollbar-track {
  background: transparent;
}

.studio-canvas-area::-webkit-scrollbar-thumb {
  background: var(--dq-scrollbar-thumb);
  border-radius: 3px;
}

.studio-canvas-area::-webkit-scrollbar-thumb:hover {
  background: var(--dq-scrollbar-thumb-hover);
}

@media (prefers-reduced-transparency: reduce) {
  .studio-layout {
    background-image: none;
  }

  .studio-composer-bar__scrim {
    background: linear-gradient(to top, var(--dq-bg-page) 0%, transparent 100%);
  }
}
</style>
