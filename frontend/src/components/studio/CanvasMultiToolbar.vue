<template>
  <div
    v-if="visible && count >= 2"
    class="canvas-multi-toolbar dq-glass--popover"
    :style="toolbarStyle"
    @mousedown.stop
  >
    <span class="canvas-multi-toolbar__count">{{ $t('canvas.selectedCount', { n: count }) }}</span>
    <span class="canvas-multi-toolbar__sep" />
    <DqIconButton
      v-for="btn in alignButtons"
      :key="btn.action"
      type="text"
      size="sm"
      :label="btn.label"
      @click="$emit('align', btn.action)"
    >
      <span class="canvas-multi-toolbar__glyph">{{ btn.glyph }}</span>
    </DqIconButton>
    <span class="canvas-multi-toolbar__sep" />
    <DqIconButton
      type="text"
      size="sm"
      :label="$t('canvas.distributeH')"
      @click="$emit('distribute', 'horizontal')"
    >
      <span class="canvas-multi-toolbar__glyph">⇔</span>
    </DqIconButton>
    <DqIconButton
      type="text"
      size="sm"
      :label="$t('canvas.distributeV')"
      @click="$emit('distribute', 'vertical')"
    >
      <span class="canvas-multi-toolbar__glyph">⇕</span>
    </DqIconButton>
    <span class="canvas-multi-toolbar__sep" />
    <DqIconButton
      type="text"
      size="sm"
      :label="$t('canvas.snapStagingToSelection')"
      @click="$emit('snap-staging')"
    >
      <span class="canvas-multi-toolbar__glyph">◎</span>
    </DqIconButton>
    <span class="canvas-multi-toolbar__sep" />
    <DqIconButton
      type="text"
      size="sm"
      :label="$t('loraTrain.saveToDataset')"
      @click="$emit('train-lora')"
    >
      <DqIcon :size="14"><MagicStick /></DqIcon>
    </DqIconButton>
    <span class="canvas-multi-toolbar__sep" />
    <DqIconButton
      type="text"
      size="sm"
      :label="$t('canvas.removeFromCanvas')"
      @click="$emit('remove')"
    >
      <DqIcon :size="14"><Delete /></DqIcon>
    </DqIconButton>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { Delete, MagicStick } from '@danqing/dq-shell';
import { useI18n } from 'vue-i18n';
import type { AlignMode, DistributeMode } from '@/utils/canvasGeometry';

const props = defineProps<{
  visible: boolean;
  count: number;
  centerX: number;
  topY: number;
}>();

defineEmits<{
  (e: 'align', mode: AlignMode): void;
  (e: 'distribute', mode: DistributeMode): void;
  (e: 'remove'): void;
  (e: 'snap-staging'): void;
  (e: 'train-lora'): void;
}>();

const { t } = useI18n();

const alignButtons = computed(() => [
  { action: 'left' as AlignMode, label: t('canvas.alignLeft'), glyph: '⫷' },
  { action: 'center' as AlignMode, label: t('canvas.alignCenter'), glyph: '⫿' },
  { action: 'right' as AlignMode, label: t('canvas.alignRight'), glyph: '⫸' },
  { action: 'top' as AlignMode, label: t('canvas.alignTop'), glyph: '⫠' },
  { action: 'middle' as AlignMode, label: t('canvas.alignMiddle'), glyph: '⫟' },
  { action: 'bottom' as AlignMode, label: t('canvas.alignBottom'), glyph: '⫡' },
]);

const toolbarStyle = computed(() => ({
  left: `${props.centerX}px`,
  top: `${props.topY}px`,
}));
</script>

<style scoped>
.canvas-multi-toolbar {
  position: absolute;
  z-index: 62;
  display: flex;
  align-items: center;
  gap: 2px;
  padding: 4px 8px;
  border-radius: 10px;
  pointer-events: auto;
  transform: translate(-50%, calc(-100% - 12px));
  box-shadow: var(--dq-shadow-md);
}

.canvas-multi-toolbar__count {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
  padding: 0 4px;
  white-space: nowrap;
}

.canvas-multi-toolbar__sep {
  width: 1px;
  height: 18px;
  margin: 0 2px;
  background: var(--dq-border-subtle);
  opacity: 0.7;
}

.canvas-multi-toolbar__glyph {
  font-size: var(--dq-font-size-caption);
  line-height: 1;
  width: 14px;
  text-align: center;
  display: inline-block;
}
</style>
