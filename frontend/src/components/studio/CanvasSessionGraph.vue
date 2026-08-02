<template>
  <aside v-if="open" class="canvas-session-graph dq-glass--popover">
    <header class="canvas-session-graph__head">
      <div class="canvas-session-graph__head-text">
        <span>{{ $t('canvas.sessionGraph') }}</span>
        <span v-if="allNodes.length > 0" class="canvas-session-graph__stats">
          <template v-if="filterQuery.trim()">
            {{
              $t('canvas.sessionGraphStatsFiltered', {
                shown: filteredNodes.length,
                total: allNodes.length,
                edges: edges.length,
              })
            }}
          </template>
          <template v-else>
            {{ $t('canvas.sessionGraphStats', { nodes: allNodes.length, edges: edges.length }) }}
          </template>
        </span>
      </div>
      <DqIconButton type="text" size="sm" :label="$t('gallery.close')" @click="$emit('close')">
        <DqIcon :size="14"><Close /></DqIcon>
      </DqIconButton>
    </header>

    <div v-if="allNodes.length > 0" class="canvas-session-graph__search">
      <DqInput
        v-model="filterQuery"
        size="sm"
        clearable
        :placeholder="$t('canvas.sessionGraphSearch')"
        @keydown.esc.stop="onSearchEsc"
      />
    </div>

    <div v-if="allNodes.length === 0" class="canvas-session-graph__empty">
      <p>{{ $t('canvas.sessionGraphEmpty') }}</p>
      <DqButton type="primary" @click="$emit('import-works')">
        {{ $t('canvas.importWorks') }}
      </DqButton>
    </div>

    <div v-else-if="filteredNodes.length === 0" class="canvas-session-graph__empty">
      <p>{{ $t('canvas.sessionGraphNoMatch') }}</p>
    </div>

    <div v-else class="canvas-session-graph__body">
      <button
        v-for="node in filteredNodes"
        :key="node.path"
        type="button"
        class="canvas-session-graph__node"
        :class="{
          'canvas-session-graph__node--active': node.path === activePath,
          'canvas-session-graph__node--child': node.depth > 0,
        }"
        :style="{ paddingLeft: `${8 + node.depth * 12}px` }"
        @click="$emit('focus-node', node.path)"
      >
        <span v-if="node.depth > 0" class="canvas-session-graph__branch" aria-hidden="true">↳</span>
        <img :src="node.thumb" :alt="node.name" />
        <div class="canvas-session-graph__meta">
          <span class="canvas-session-graph__name">{{ node.name }}</span>
          <span v-if="node.parentName" class="canvas-session-graph__parent">
            {{ $t('canvas.sessionGraphFrom', { name: node.parentName }) }}
          </span>
          <span v-if="node.notePreview" class="canvas-session-graph__note">{{ node.notePreview }}</span>
        </div>
        <span v-if="node.relationLabel" class="canvas-session-graph__relation">
          {{ node.relationLabel }}
        </span>
        <span v-else-if="node.childCount > 0" class="canvas-session-graph__children">
          → {{ node.childCount }}
        </span>
      </button>
    </div>
  </aside>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { Close } from '@danqing/dq-shell';
import { canvasNodeDisplayName, previewUrlForGalleryItem } from '@/utils/canvasAssets';
import { computeNodeDepths } from '@/utils/canvasEdges';
import { lineageRelationLabel } from '@/utils/lineageRelationLabel';
import type { CanvasEdge, CanvasItemState, GalleryItem } from '@/types';

const props = defineProps<{
  open: boolean;
  items: Record<string, CanvasItemState>;
  edges: CanvasEdge[];
  galleryItems: GalleryItem[];
  activePath: string;
}>();

defineEmits<{
  (e: 'close'): void;
  (e: 'focus-node', path: string): void;
  (e: 'import-works'): void;
}>();

const filterQuery = ref('');

watch(
  () => props.open,
  (open) => {
    if (!open) filterQuery.value = '';
  },
);

const allNodes = computed(() => {
  const childCount = new Map<string, number>();
  const parentPathByChild = new Map<string, string>();
  const relationByChild = new Map<string, string>();
  for (const e of props.edges) {
    childCount.set(e.from, (childCount.get(e.from) || 0) + 1);
    parentPathByChild.set(e.to, e.from);
    relationByChild.set(e.to, e.relation);
  }
  const depths = computeNodeDepths(props.items, props.edges);

  return Object.entries(props.items)
    .filter(([, s]) => s.visible)
    .map(([path, state]) => {
      const gi = props.galleryItems.find((g) => g.path === path);
      const parentPath = parentPathByChild.get(path);
      const parentGi = parentPath
        ? props.galleryItems.find((g) => g.path === parentPath)
        : undefined;
      const parentState = parentPath ? props.items[parentPath] : undefined;
      const relation = relationByChild.get(path) || gi?.metadata?.relation_type;
      return {
        path,
        name: canvasNodeDisplayName(path, state, gi),
        thumb: gi ? previewUrlForGalleryItem(gi) : '',
        childCount: childCount.get(path) || 0,
        depth: depths.get(path) || 0,
        zIndex: state.zIndex,
        parentName: parentPath
          ? canvasNodeDisplayName(parentPath, parentState || state, parentGi)
          : '',
        relationLabel: depthOrRelationLabel(depths.get(path) || 0, relation),
        notePreview: notePreview(state.note),
      };
    })
    .sort((a, b) => a.depth - b.depth || a.name.localeCompare(b.name) || b.zIndex - a.zIndex);
});

const filteredNodes = computed(() => {
  const q = filterQuery.value.trim().toLowerCase();
  if (!q) return allNodes.value;
  return allNodes.value.filter(
    (n) =>
      n.name.toLowerCase().includes(q) ||
      n.parentName.toLowerCase().includes(q) ||
      n.notePreview.toLowerCase().includes(q),
  );
});

function onSearchEsc() {
  if (filterQuery.value.trim()) {
    filterQuery.value = '';
    return;
  }
}

function clearFilterIfActive(): boolean {
  if (!filterQuery.value.trim()) return false;
  filterQuery.value = '';
  return true;
}

defineExpose({ clearFilterIfActive });

function depthOrRelationLabel(depth: number, relation: unknown): string {
  if (depth <= 0) return '';
  return lineageRelationLabel(String(relation || 'create'));
}

function notePreview(note: string | undefined, max = 48): string {
  const s = String(note || '').replace(/\s+/g, ' ').trim();
  if (!s) return '';
  if (s.length <= max) return s;
  return `${s.slice(0, max - 1)}…`;
}
</script>

<style scoped>
.canvas-session-graph {
  position: absolute;
  top: 52px;
  left: 12px;
  bottom: calc(16px + var(--dq-composer-reserve, min(200px, 36vh)) + 56px);
  width: min(240px, 32vw);
  z-index: 45;
  display: flex;
  flex-direction: column;
  border: 1px solid var(--dq-border-subtle);
  border-radius: 10px;
  overflow: hidden;
  pointer-events: auto;
}

.canvas-session-graph__head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 8px;
  padding: 10px 12px;
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  border-bottom: 1px solid var(--dq-border-subtle);
}

.canvas-session-graph__head-text {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}

.canvas-session-graph__stats {
  font-size: var(--dq-font-size-caption);
  font-weight: 400;
  color: var(--dq-label-tertiary);
}

.canvas-session-graph__search {
  padding: 8px 10px 0;
}

.canvas-session-graph__body {
  flex: 1;
  overflow: auto;
  padding: 8px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.canvas-session-graph__empty {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  padding: 16px;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
  text-align: center;
}

.canvas-session-graph__empty p {
  margin: 0;
  line-height: 1.45;
}

.canvas-session-graph__node {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  padding: 6px 8px;
  border: 1px solid transparent;
  border-radius: 8px;
  background: transparent;
  cursor: pointer;
  text-align: left;
}

.canvas-session-graph__node:hover {
  background: var(--dq-surface-inset-hover);
}

.canvas-session-graph__node--active {
  border-color: var(--dq-accent);
  background: color-mix(in srgb, var(--dq-accent) 10%, transparent);
}

.canvas-session-graph__branch {
  flex-shrink: 0;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
  width: 10px;
}

.canvas-session-graph__node img {
  width: 36px;
  height: 36px;
  border-radius: 6px;
  object-fit: cover;
  flex-shrink: 0;
}

.canvas-session-graph__meta {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 1px;
}

.canvas-session-graph__name {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.canvas-session-graph__parent {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.canvas-session-graph__note {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
  font-style: italic;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.canvas-session-graph__relation {
  flex-shrink: 0;
  font-size: var(--dq-font-size-caption);
  padding: 2px 5px;
  border-radius: 4px;
  background: color-mix(in srgb, var(--dq-accent) 12%, transparent);
  color: var(--dq-accent);
}

.canvas-session-graph__children {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
  flex-shrink: 0;
}
</style>
