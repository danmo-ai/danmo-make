<template>
  <div class="canvas-toolbar">
    <span class="canvas-toolbar__title">{{ $t('canvas.viewToolsLabel') }}</span>
    <div class="canvas-toolbar__group">
      <DqIconButton
        type="text"
        size="sm"
        :label="$t('canvas.fitAll')"
        @click="$emit('fit-all')"
      >
        <DqIcon :size="14"><Grid /></DqIcon>
      </DqIconButton>
      <DqIconButton
        type="text"
        size="sm"
        :disabled="zoom <= 0.11"
        :label="'-'"
        @click="$emit('update:zoom', Math.max(0.1, zoom - 0.2))"
      >
        <span class="canvas-toolbar__zoom-out-icon">&minus;</span>
      </DqIconButton>
      <span class="canvas-toolbar__zoom-label">{{ Math.round(zoom * 100) }}%</span>
      <DqIconButton
        type="text"
        size="sm"
        :disabled="zoom >= 4.9"
        :label="'+'"
        @click="$emit('update:zoom', Math.min(5, zoom + 0.2))"
      >
        <DqIcon :size="14"><ZoomIn /></DqIcon>
      </DqIconButton>
      <DqIconButton
        type="text"
        size="sm"
        :label="$t('canvas.resetView')"
        @click="$emit('reset-view')"
      >
        <DqIcon :size="14"><Refresh /></DqIcon>
      </DqIconButton>
    </div>
    <div class="canvas-toolbar__group">
      <DqIconButton
        type="text"
        size="sm"
        class="canvas-toolbar__btn--accent"
        :label="$t('studio.openComposer')"
        @click="$emit('open-composer')"
      >
        <DqIcon :size="14"><MagicStick /></DqIcon>
      </DqIconButton>
      <DqIconButton
        type="text"
        size="sm"
        class="canvas-toolbar__btn--accent"
        :label="$t('canvas.importWorks')"
        @click="$emit('open-gallery-picker')"
      >
        <DqIcon :size="14"><Plus /></DqIcon>
      </DqIconButton>
      <DqIconButton
        type="text"
        size="sm"
        :class="{ 'canvas-toolbar__btn--active': graphOpen }"
        :label="$t('canvas.sessionGraph')"
        @click="$emit('toggle-graph')"
      >
        <DqIcon :size="14"><Document /></DqIcon>
      </DqIconButton>
      <DqIconButton
        type="text"
        size="sm"
        :class="{ 'canvas-toolbar__btn--active': edgesOpen }"
        :label="$t('canvas.showEdges')"
        @click="$emit('toggle-edges')"
      >
        <DqIcon :size="14"><Grid /></DqIcon>
      </DqIconButton>
      <DqIconButton
        type="text"
        size="sm"
        :class="{ 'canvas-toolbar__btn--active': guidesOpen }"
        :label="$t('canvas.regionGuides')"
        @click="$emit('toggle-guides')"
      >
        <DqIcon :size="14"><Aim /></DqIcon>
      </DqIconButton>
      <DqIconButton
        type="text"
        size="sm"
        :label="$t('canvas.layers')"
        @click="$emit('toggle-layers')"
      >
        <DqIcon :size="14"><Menu /></DqIcon>
      </DqIconButton>
      <DqDropdown trigger="click" size="sm" @command="onMoreCommand">
        <DqIconButton
          type="text"
          size="sm"
          :label="$t('canvas.moreMenu')"
        >
          <DqIcon :size="14"><Menu /></DqIcon>
        </DqIconButton>
        <template #dropdown>
          <DqDropdownMenu>
            <DqDropdownItem command="import-json">
              {{ $t('canvas.importJson') }}
            </DqDropdownItem>
            <DqDropdownItem disabled>
              <span class="canvas-toolbar__menu-hint">{{ $t('canvas.pasteImportHint') }}</span>
            </DqDropdownItem>
            <DqDropdownItem command="export-json">
              {{ $t('canvas.exportJson') }}
            </DqDropdownItem>
            <DqDropdownItem command="export-png">
              {{ $t('canvas.exportPng') }}
            </DqDropdownItem>
            <DqDropdownItem command="copy-session">
              {{ $t('canvas.copySession') }}
            </DqDropdownItem>
            <DqDropdownItem divided disabled>
              <span class="canvas-toolbar__menu-section">{{ $t('canvas.shortcutsTitle') }}</span>
            </DqDropdownItem>
            <DqDropdownItem disabled>
              <span class="canvas-toolbar__menu-hint">{{ $t('canvas.shortcutRemove') }}</span>
            </DqDropdownItem>
            <DqDropdownItem disabled>
              <span class="canvas-toolbar__menu-hint">{{ $t('canvas.shortcutEscape') }}</span>
            </DqDropdownItem>
            <DqDropdownItem disabled>
              <span class="canvas-toolbar__menu-hint">{{ $t('canvas.shortcutEnter') }}</span>
            </DqDropdownItem>
            <DqDropdownItem disabled>
              <span class="canvas-toolbar__menu-hint">{{ $t('canvas.shortcutFit') }}</span>
            </DqDropdownItem>
            <DqDropdownItem disabled>
              <span class="canvas-toolbar__menu-hint">{{ $t('canvas.shortcutLayers') }}</span>
            </DqDropdownItem>
            <DqDropdownItem disabled>
              <span class="canvas-toolbar__menu-hint">{{ $t('canvas.shortcutPan') }}</span>
            </DqDropdownItem>
            <DqDropdownItem disabled>
              <span class="canvas-toolbar__menu-hint">{{ $t('canvas.shortcutImport') }}</span>
            </DqDropdownItem>
            <DqDropdownItem disabled>
              <span class="canvas-toolbar__menu-hint">{{ $t('canvas.shortcutGuides') }}</span>
            </DqDropdownItem>
            <DqDropdownItem disabled>
              <span class="canvas-toolbar__menu-hint">{{ $t('canvas.shortcutRename') }}</span>
            </DqDropdownItem>
            <DqDropdownItem disabled>
              <span class="canvas-toolbar__menu-hint">{{ $t('canvas.shortcutLineage') }}</span>
            </DqDropdownItem>
            <DqDropdownItem disabled>
              <span class="canvas-toolbar__menu-hint">{{ $t('canvas.shortcutSnapStaging') }}</span>
            </DqDropdownItem>
          </DqDropdownMenu>
        </template>
      </DqDropdown>
    </div>
  </div>
</template>

<script setup lang="ts">
import { Grid, ZoomIn, Refresh, Menu, Document, Plus, Aim, MagicStick } from '@danqing/dq-shell';

defineProps<{
  zoom: number;
  graphOpen?: boolean;
  edgesOpen?: boolean;
  guidesOpen?: boolean;
}>();

const emit = defineEmits<{
  (e: 'update:zoom', zoom: number, cx?: number, cy?: number): void;
  (e: 'fit-all'): void;
  (e: 'reset-view'): void;
  (e: 'toggle-layers'): void;
  (e: 'toggle-graph'): void;
  (e: 'toggle-edges'): void;
  (e: 'toggle-guides'): void;
  (e: 'import-json'): void;
  (e: 'export-json'): void;
  (e: 'export-png'): void;
  (e: 'copy-session'): void;
  (e: 'open-gallery-picker'): void;
  (e: 'open-composer'): void;
}>();

function onMoreCommand(cmd: string) {
  if (cmd === 'import-json') emit('import-json');
  else if (cmd === 'export-json') emit('export-json');
  else if (cmd === 'export-png') emit('export-png');
  else if (cmd === 'copy-session') emit('copy-session');
}
</script>

<style scoped>
.canvas-toolbar {
  position: absolute;
  bottom: calc(16px + var(--dq-composer-reserve, min(200px, 36vh)));
  right: 16px;
  display: flex;
  align-items: center;
  gap: 6px;
  z-index: 50;
  pointer-events: auto;
}

.canvas-toolbar__title {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  color: var(--dq-label-tertiary);
  writing-mode: vertical-rl;
  text-orientation: mixed;
  letter-spacing: var(--dq-tracking-wider);
  user-select: none;
  padding: 2px 0;
}

.canvas-toolbar__group {
  display: flex;
  align-items: center;
  gap: 2px;
  background: var(--dq-bg-base);
  border: 1px solid var(--dq-border-subtle);
  border-radius: 8px;
  padding: 4px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.canvas-toolbar__zoom-label {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
  width: 36px;
  text-align: center;
  font-variant-numeric: tabular-nums;
}

.canvas-toolbar__btn--active {
  color: var(--dq-accent);
  background: color-mix(in srgb, var(--dq-accent) 12%, transparent);
  border-radius: 6px;
}

.canvas-toolbar__btn--accent {
  color: var(--dq-accent);
}

.canvas-toolbar__menu-hint {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
  white-space: normal;
  line-height: 1.35;
  max-width: 200px;
  display: block;
}

.canvas-toolbar__menu-section {
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  color: var(--dq-label-primary);
}

.canvas-toolbar__zoom-out-icon {
  font-size: var(--dq-font-size-title);
  font-weight: 600;
  line-height: 1;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 14px;
  height: 14px;
  color: var(--dq-label-primary);
}
</style>
