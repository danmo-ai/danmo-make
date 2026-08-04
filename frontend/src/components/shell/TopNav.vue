<template>
  <div class="dq-sidebar-nav">
    <!-- Brand Logo -->
    <div class="dq-sidebar-nav__brand">
      <DqIcon :size="22"><MagicStick /></DqIcon>
    </div>

    <template v-for="group in navGroups" :key="group.id">
      <div v-if="group.dividerBefore" class="dq-sidebar-nav__divider" aria-hidden="true" />

      <nav
        class="dq-sidebar-nav__group"
        role="navigation"
        :aria-label="group.ariaLabel"
      >
        <template v-for="item in group.items" :key="item.id">
          <DqCountBadge
            v-if="item.isQueue"
            :value="queueCount"
            :hidden="queueCount === 0"
            class="queue-badge"
          >
            <button
              type="button"
              class="dq-sidebar-nav__item"
              @click="openQueue"
            >
              <DqIcon :size="22"><component :is="item.icon" /></DqIcon>
              <span class="dq-sidebar-nav__tooltip">{{ item.label }}</span>
            </button>
          </DqCountBadge>
          <button
            v-else
            type="button"
            class="dq-sidebar-nav__item"
            :class="{ 'is-active': activePage === item.id }"
            @click="onNavSelect(item.id)"
          >
            <DqIcon :size="22"><component :is="item.icon" /></DqIcon>
            <span class="dq-sidebar-nav__tooltip">{{ item.label }}</span>
          </button>
        </template>
      </nav>

      <div v-if="group.spacerAfter" class="dq-sidebar-nav__spacer" aria-hidden="true" />
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, type Component } from 'vue';
import { useI18n } from 'vue-i18n';
import {
  Box,
  Brush,
  MagicStick,
  Microphone,
  Picture,
  Setting,
  VideoCamera,
} from '@danqing/dq-shell';

defineProps<{
  activePage: string;
  queueCount: number;
}>();

const emit = defineEmits<{
  (e: 'navigate', page: string): void;
  (e: 'open-queue'): void;
}>();

const { t } = useI18n();

interface NavItem {
  id: string;
  /** Global icon name from registerDqIcons (dq-ui registry). */
  icon: Component | string;
  label: string;
  isQueue?: boolean;
}

interface NavGroup {
  id: string;
  ariaLabel: string;
  items: NavItem[];
  dividerBefore?: boolean;
  spacerAfter?: boolean;
}

/**
 * Sidebar groups (top → bottom):
 * 1. Studio — image / video / audio composers (main battlefield)
 * 2. Workflow — copilot + global generation queue (used while creating)
 * 3. Library — models + prompt templates (assets & presets)
 * 4. System — settings
 */
const navGroups = computed<NavGroup[]>(() => [
  {
    id: 'studio',
    ariaLabel: t('nav.groupStudio'),
    items: [
      { id: 'image_create', icon: Picture, label: t('nav.image_create') },
      { id: 'video_create', icon: VideoCamera, label: t('nav.video_create') },
      { id: 'avatar_create', icon: 'UserRound', label: t('nav.avatar_create') },
      { id: 'audio_create', icon: Microphone, label: t('nav.audio_create') },
      { id: 'lora_train', icon: MagicStick, label: t('nav.lora_train') },
    ],
  },
  {
    id: 'workflow',
    ariaLabel: t('nav.groupWorkflow'),
    dividerBefore: true,
    items: [
      { id: 'assistant', icon: 'Bot', label: t('nav.assistant') },
      { id: 'task_queue', icon: 'ListOrdered', label: t('nav.task_queue'), isQueue: true },
    ],
    spacerAfter: true,
  },
  {
    id: 'library',
    ariaLabel: t('nav.groupLibrary'),
    dividerBefore: true,
    items: [
      { id: 'models', icon: Box, label: t('nav.models') },
      { id: 'prompts', icon: Brush, label: t('nav.prompts') },
    ],
  },
  {
    id: 'system',
    ariaLabel: t('nav.groupSystem'),
    dividerBefore: true,
    items: [{ id: 'settings', icon: Setting, label: t('nav.settings') }],
  },
]);

function onNavSelect(index: string) {
  emit('navigate', index);
}

function openQueue() {
  emit('open-queue');
}
</script>

<style scoped>
.dq-sidebar-nav {
  display: flex;
  flex-direction: column;
  align-items: center;
  height: 100%;
  width: 100%;
  padding: 14px 0 16px;
  box-sizing: border-box;
}

/* ── Brand ── */
.dq-sidebar-nav__brand {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  margin-bottom: 18px;
  color: var(--dq-accent);
  flex-shrink: 0;
}

/* ── Group ── */
.dq-sidebar-nav__group {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  width: 100%;
  flex-shrink: 0;
}

.dq-sidebar-nav__spacer {
  flex: 1;
  width: 100%;
  min-height: 8px;
}

/* ── Divider ── */
.dq-sidebar-nav__divider {
  width: 18px;
  height: 1px;
  background: var(--dq-border-subtle);
  margin: 8px 0;
  flex-shrink: 0;
  opacity: 0.6;
}

/* ── Nav Item ── */
.dq-sidebar-nav__item {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  padding: 0;
  margin: 0;
  border: none;
  border-radius: 10px;
  background: transparent;
  color: var(--dq-label-secondary);
  cursor: pointer;
  transition:
    color 0.2s cubic-bezier(0.4, 0, 0.2, 1),
    background 0.2s cubic-bezier(0.4, 0, 0.2, 1),
    box-shadow 0.2s ease,
    transform 0.15s ease;
  font-family: inherit;
  flex-shrink: 0;
}

.dq-sidebar-nav__item:hover {
  color: var(--dq-label-primary);
  background: var(--dq-fill-tertiary);
}

.dq-sidebar-nav__item.is-active {
  color: var(--dq-accent);
  background: color-mix(in srgb, var(--dq-accent) 16%, transparent);
}

.dq-sidebar-nav__item.is-active:hover {
  background: color-mix(in srgb, var(--dq-accent) 22%, transparent);
}

.dq-sidebar-nav__item:active {
  transform: scale(0.94) translateY(0);
  transition-duration: 0.08s;
}

/* ── Tooltip ── */
.dq-sidebar-nav__tooltip {
  position: absolute;
  left: calc(100% + 12px);
  top: 50%;
  transform: translateY(-50%) scale(0.95);
  padding: 6px 10px;
  background: var(--dq-glass-tooltip-bg);
  border: 0.5px solid var(--dq-glass-border);
  border-radius: 7px;
  font-size: var(--dq-font-size-caption);
  font-weight: 500;
  color: var(--dq-label-primary);
  white-space: nowrap;
  pointer-events: none;
  opacity: 0;
  visibility: hidden;
  transition:
    opacity 0.12s ease,
    transform 0.12s ease,
    visibility 0.12s;
  box-shadow: var(--dq-shadow-md);
  backdrop-filter: var(--dq-glass-blur-light);
  -webkit-backdrop-filter: var(--dq-glass-blur-light);
  z-index: 1000;
}

.dq-sidebar-nav__tooltip::before {
  content: '';
  position: absolute;
  left: -4px;
  top: 50%;
  transform: translateY(-50%) rotate(45deg);
  width: 7px;
  height: 7px;
  background: var(--dq-glass-tooltip-bg);
  border-left: 0.5px solid var(--dq-glass-border);
  border-bottom: 0.5px solid var(--dq-glass-border);
}

.dq-sidebar-nav__item:hover .dq-sidebar-nav__tooltip {
  opacity: 1;
  visibility: visible;
  transform: translateY(-50%) scale(1);
}

/* ── Queue Badge ── */
.queue-badge {
  position: relative;
}

.queue-badge :deep(.dq-badge__content) {
  top: 2px;
  right: 2px;
  min-width: 14px;
  height: 14px;
  padding: 0 3px;
  font-size: var(--dq-font-size-caption);
  font-weight: 700;
  border-radius: 7px;
}
</style>
