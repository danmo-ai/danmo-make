<template>
  <div class="dq-sidebar-nav" :class="{ 'is-collapsed': collapsed }">
    <div class="dq-sidebar-nav__brand">
      <span class="dq-sidebar-nav__brand-mark" aria-hidden="true">
        <DqIcon :size="20"><MagicStick /></DqIcon>
      </span>
      <span class="dq-sidebar-nav__brand-name">Danmo Make</span>
    </div>

    <div class="dq-sidebar-nav__groups">
      <template v-for="group in navGroups" :key="group.id">
        <nav
          class="dq-sidebar-nav__group"
          role="navigation"
          :aria-label="group.ariaLabel"
        >
          <div class="dq-sidebar-nav__group-label">{{ group.ariaLabel }}</div>
          <button
            v-for="item in group.items"
            :key="item.id"
            type="button"
            class="dq-sidebar-nav__item"
            :class="{ 'is-active': !item.isQueue && activePage === item.id }"
            :aria-current="!item.isQueue && activePage === item.id ? 'page' : undefined"
            :aria-label="collapsed ? item.label : undefined"
            @click="item.isQueue ? openQueue() : onNavSelect(item.id)"
          >
            <DqIcon class="dq-sidebar-nav__item-icon" :size="18">
              <component :is="item.icon" />
            </DqIcon>
            <span class="dq-sidebar-nav__label">{{ item.label }}</span>
            <span
              v-if="item.isQueue && queueCount > 0"
              class="dq-sidebar-nav__count"
              :aria-label="$t('nav.queuePending', { count: queueCount })"
            >{{ queueCount > 99 ? '99+' : queueCount }}</span>
            <span v-if="collapsed" class="dq-sidebar-nav__tooltip" role="tooltip">
              {{ item.label }}
            </span>
          </button>
        </nav>
        <div v-if="group.spacerAfter" class="dq-sidebar-nav__spacer" aria-hidden="true" />
      </template>
    </div>

    <button
      type="button"
      class="dq-sidebar-nav__item dq-sidebar-nav__collapse"
      :aria-label="collapsed ? $t('nav.expandSidebar') : $t('nav.collapseSidebar')"
      :title="collapsed ? $t('nav.expandSidebar') : $t('nav.collapseSidebar')"
      @click="$emit('toggle-collapse')"
    >
      <DqIcon class="dq-sidebar-nav__item-icon" :size="18">
        <ArrowRight v-if="collapsed" />
        <ArrowLeft v-else />
      </DqIcon>
      <span class="dq-sidebar-nav__label">{{ $t('nav.collapseSidebar') }}</span>
    </button>
  </div>
</template>

<script setup lang="ts">
import { computed, type Component } from 'vue';
import { useI18n } from 'vue-i18n';
import {
  ArrowLeft,
  ArrowRight,
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
  collapsed: boolean;
}>();

const emit = defineEmits<{
  (e: 'navigate', page: string): void;
  (e: 'open-queue'): void;
  (e: 'toggle-collapse'): void;
}>();

const { t } = useI18n();

interface NavItem {
  id: string;
  icon: Component | string;
  label: string;
  isQueue?: boolean;
}

interface NavGroup {
  id: string;
  ariaLabel: string;
  items: NavItem[];
  spacerAfter?: boolean;
}

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
    items: [
      { id: 'assistant', icon: 'Bot', label: t('nav.assistant') },
      { id: 'task_queue', icon: 'ListOrdered', label: t('nav.task_queue'), isQueue: true },
    ],
  },
  {
    id: 'library',
    ariaLabel: t('nav.groupLibrary'),
    items: [
      { id: 'models', icon: Box, label: t('nav.models') },
      { id: 'prompts', icon: Brush, label: t('nav.prompts') },
    ],
    spacerAfter: true,
  },
  {
    id: 'system',
    ariaLabel: t('nav.groupSystem'),
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
  width: 100%;
  height: 100%;
  padding: 16px 12px 12px;
  box-sizing: border-box;
}

.dq-sidebar-nav__brand {
  display: flex;
  align-items: center;
  gap: 10px;
  height: 36px;
  margin: 0 8px 24px;
  flex-shrink: 0;
}

.dq-sidebar-nav__brand-mark {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  flex-shrink: 0;
  border: 1px solid color-mix(in srgb, var(--dq-accent) 24%, var(--dq-border-subtle));
  border-radius: 9px;
  color: var(--dq-accent);
  background: color-mix(in srgb, var(--dq-accent) 8%, var(--dq-bg-base));
}

.dq-sidebar-nav__brand-name {
  overflow: hidden;
  color: var(--dq-label-primary);
  font-size: var(--dq-font-size-body);
  font-weight: 650;
  letter-spacing: var(--dq-tracking-tight);
  white-space: nowrap;
}

.dq-sidebar-nav__groups {
  display: flex;
  flex: 1;
  flex-direction: column;
  min-height: 0;
  overflow-x: visible;
  overflow-y: auto;
  scrollbar-width: none;
}

.dq-sidebar-nav__groups::-webkit-scrollbar {
  display: none;
}

.dq-sidebar-nav__group {
  display: flex;
  flex-direction: column;
  gap: 3px;
  width: 100%;
  margin-bottom: 20px;
  flex-shrink: 0;
}

.dq-sidebar-nav__group-label {
  height: 24px;
  padding: 0 10px;
  color: var(--dq-label-tertiary);
  font-size: var(--dq-font-size-caption);
  font-weight: 500;
  line-height: 24px;
}

.dq-sidebar-nav__spacer {
  flex: 1;
  min-height: 12px;
}

.dq-sidebar-nav__item {
  position: relative;
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  height: 38px;
  min-height: 38px;
  margin: 0;
  padding: 0 10px;
  border: 0;
  border-radius: var(--dq-radius-control-sm);
  color: var(--dq-label-secondary);
  background: transparent;
  cursor: pointer;
  font: inherit;
  font-size: var(--dq-font-size-body);
  text-align: left;
  transition: color 140ms ease, background-color 140ms ease;
}

.dq-sidebar-nav__item:hover {
  color: var(--dq-label-primary);
  background: var(--dq-fill-tertiary);
}

.dq-sidebar-nav__item.is-active {
  color: var(--dq-label-primary);
  background: var(--dq-fill-secondary);
  font-weight: 600;
}

.dq-sidebar-nav__item.is-active::before {
  position: absolute;
  top: 10px;
  bottom: 10px;
  left: 0;
  width: 2px;
  border-radius: 2px;
  background: var(--dq-accent);
  content: '';
}

.dq-sidebar-nav__item:active {
  background: var(--dq-fill-secondary);
}

.dq-sidebar-nav__item-icon {
  width: 18px;
  flex: 0 0 18px;
  color: var(--dq-label-tertiary);
}

.dq-sidebar-nav__item:hover .dq-sidebar-nav__item-icon,
.dq-sidebar-nav__item.is-active .dq-sidebar-nav__item-icon {
  color: var(--dq-accent);
}

.dq-sidebar-nav__label {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.dq-sidebar-nav__count {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 20px;
  height: 20px;
  padding: 0 5px;
  border-radius: 10px;
  color: var(--dq-on-accent);
  background: var(--dq-accent);
  font-size: var(--dq-font-size-caption);
  font-variant-numeric: tabular-nums;
  line-height: 1;
}

.dq-sidebar-nav__collapse {
  flex-shrink: 0;
  margin-top: 8px;
  border-top: 1px solid var(--dq-border-subtle);
  border-radius: 0;
  color: var(--dq-label-tertiary);
}

.dq-sidebar-nav__tooltip {
  display: none;
}

.is-collapsed {
  padding-right: 10px;
  padding-left: 10px;
}

.is-collapsed .dq-sidebar-nav__brand {
  margin-right: 0;
  margin-left: 0;
  justify-content: center;
}

.is-collapsed .dq-sidebar-nav__brand-name,
.is-collapsed .dq-sidebar-nav__group-label,
.is-collapsed .dq-sidebar-nav__label,
.is-collapsed .dq-sidebar-nav__collapse .dq-sidebar-nav__label {
  display: none;
}

.is-collapsed .dq-sidebar-nav__item {
  justify-content: center;
  padding: 0;
}

.is-collapsed .dq-sidebar-nav__count {
  position: absolute;
  top: 2px;
  right: 2px;
  min-width: 16px;
  height: 16px;
  padding: 0 3px;
  font-size: var(--dq-font-size-caption);
}

.is-collapsed .dq-sidebar-nav__tooltip {
  position: absolute;
  top: 50%;
  left: calc(100% + 12px);
  z-index: 1000;
  display: block;
  padding: 6px 9px;
  border: 1px solid var(--dq-border-subtle);
  border-radius: var(--dq-radius-control-sm);
  color: var(--dq-label-primary);
  background: var(--dq-bg-base);
  box-shadow: var(--dq-shadow-md);
  font-size: var(--dq-font-size-caption);
  font-weight: 500;
  opacity: 0;
  pointer-events: none;
  transform: translate(4px, -50%);
  transition: opacity 120ms ease, transform 120ms ease;
  visibility: hidden;
  white-space: nowrap;
}

.is-collapsed .dq-sidebar-nav__item:hover .dq-sidebar-nav__tooltip,
.is-collapsed .dq-sidebar-nav__item:focus-visible .dq-sidebar-nav__tooltip {
  opacity: 1;
  transform: translate(0, -50%);
  visibility: visible;
}
</style>
