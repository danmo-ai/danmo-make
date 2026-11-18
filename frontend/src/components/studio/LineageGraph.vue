<template>
  <div
    class="lineage-graph"
    ref="containerRef"
    @wheel.prevent="onWheel"
    @mousedown.middle.prevent="onPanStart"
    @mousemove="onPanMove"
    @mouseup="onPanEnd"
    @mouseleave="onPanEnd"
  >
    <div v-if="!hasNodes" class="lineage-graph__empty">
      {{ $t('gallery.lineageEmpty') }}
    </div>

    <template v-else>
      <!-- Floating reset-view button -->
      <button
        v-if="isPanned"
        type="button"
        class="lineage-graph__reset-btn"
        :aria-label="$t('canvas.resetView')"
        @click="resetView"
      >
        <DqIcon :size="14"><Aim /></DqIcon>
      </button>

      <div class="lineage-graph__viewport" :style="viewportStyle">
        <!-- SVG edges layer -->
        <svg
          class="lineage-graph__edges"
          :viewBox="'0 0 ' + graphWidth + ' ' + graphHeight"
          :width="graphWidth"
          :height="graphHeight"
          :style="{ minWidth: graphWidth + 'px', minHeight: graphHeight + 'px' }"
        >
          <path
            v-for="edge in edges"
            :key="edge.key"
            :d="edge.path"
            class="lineage-graph__edge"
          />
          <text
            v-for="el in edgeLabels"
            :key="el.key"
            :x="el.x"
            :y="el.y"
            class="lineage-graph__edge-label"
          >{{ el.text }}</text>
        </svg>

        <!-- Node cards layer -->
        <div
          v-for="pn in positionedNodes"
          :key="pn.node.id"
          class="lineage-graph__node"
          :class="{
            'lineage-graph__node--current': pn.node.id === currentNodeId,
            'lineage-graph__node--root': pn.depth === 0,
            'lineage-graph__node--on-canvas': isOnCanvas(pn.node.id),
          }"
          :style="{
            left: pn.x + 'px',
            top: pn.y + 'px',
            width: NODE_W + 'px',
          }"
        >
          <button
            type="button"
            class="lineage-graph__node-card"
            :title="$t('canvas.lineageJumpHint')"
            @click.stop="onNodeClick(pn.node.id)"
            @dblclick.stop="onNodeDblClick(pn.node.id)"
          >
            <div class="lineage-graph__thumb">
              <img
                v-if="thumbUrl(pn.node)"
                :src="thumbUrl(pn.node)"
                alt=""
                class="lineage-graph__thumb-img"
              />
              <PictureFilled v-else :size="20" class="lineage-graph__thumb-fallback" />
            </div>
            <div class="lineage-graph__info">
              <div v-if="pn.node.width && pn.node.height" class="lineage-graph__dim">
                {{ pn.node.width }}x{{ pn.node.height }}
              </div>
              <div class="lineage-graph__time">{{ fmtTime(pn.node.created_at) }}</div>
              <span v-if="relText(pn.node.relation_type)" class="lineage-graph__rel-tag">
                {{ relText(pn.node.relation_type) }}
              </span>
              <span v-if="isOnCanvas(pn.node.id)" class="lineage-graph__canvas-tag">
                {{ $t('canvas.onCanvasBadge') }}
              </span>
            </div>
          </button>
        </div>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue';
import { Aim, DqIcon, PictureFilled } from '@danqing/dq-shell';
import type { LineageNode } from '@/types';
import { lineageRelationLabel } from '@/utils/lineageRelationLabel';

const props = defineProps<{
  data: LineageNode | null;
  currentNodeId?: string;
  /** Asset ids currently placed on the active canvas session. */
  onCanvasIds?: string[];
}>();

const emit = defineEmits<{
  (e: 'focus-asset', assetId: string): void;
  (e: 'request-close'): void;
}>();

const onCanvasSet = computed(() => new Set(props.onCanvasIds || []));

/* ------------------------------------------------------------------ */
/*  Layout constants                                                    */
/* ------------------------------------------------------------------ */
const NODE_W = 170;
const NODE_H = 68;
const NODE_GAP_X = 90;
const NODE_GAP_Y = 30;
const PAD = 24;

/* ------------------------------------------------------------------ */
/*  Tree layout computation                                             */
/* ------------------------------------------------------------------ */

interface PosNode {
  node: LineageNode;
  x: number;
  y: number;
  depth: number;
}

function buildTree(root: LineageNode): LineageNode[] {
  let current: LineageNode = root;
  while (current.parent) {
    current = current.parent;
  }
  populateChainChildren(current);
  return flattenTree(current);
}

function populateChainChildren(node: LineageNode) {
  let n: LineageNode | null = node;
  while (n) {
    if (n.children && n.children.length > 0) {
      const first = n.children[0];
      const hasFullChildren = n.children.length > 1 || (first && !isChainNode(first));
      if (hasFullChildren) break;
      n = n.children[0];
    } else {
      break;
    }
  }
}

function isChainNode(node: LineageNode): boolean {
  return node.parent !== null && node.parent.children.length === 1;
}

function flattenTree(root: LineageNode): LineageNode[] {
  const result: { node: LineageNode; depth: number }[] = [];
  function walk(node: LineageNode, depth: number) {
    result.push({ node, depth });
    if (node.children) {
      for (const child of node.children) {
        walk(child, depth + 1);
      }
    }
  }
  walk(root, 0);
  return result.map(({ node }) => node);
}

/* ------------------------------------------------------------------ */
/*  Computed tree + layout                                              */
/* ------------------------------------------------------------------ */

const positionedNodes = computed<PosNode[]>(() => {
  if (!props.data) return [];
  const root = JSON.parse(JSON.stringify(props.data)) as LineageNode;
  const all = buildTree(root);

  const cols: Map<number, LineageNode[]> = new Map();
  let maxDepth = 0;
  for (const n of all) {
    let d = 0;
    let p: LineageNode | null = n.parent;
    while (p) {
      d++;
      p = p.parent;
    }
    if (!cols.has(d)) cols.set(d, []);
    cols.get(d)!.push(n);
    if (d > maxDepth) maxDepth = d;
  }

  const xForDepth = (d: number) => PAD + d * (NODE_W + NODE_GAP_X);

  const positioned: PosNode[] = [];
  const colHeights: number[] = [];
  for (let d = 0; d <= maxDepth; d++) {
    const nodes = cols.get(d) || [];
    const h = nodes.length * NODE_H + Math.max(0, nodes.length - 1) * NODE_GAP_Y;
    colHeights.push(h);
  }
  const totalH = Math.max(...colHeights) + PAD * 2;
  const centerY = totalH / 2;

  for (let d = 0; d <= maxDepth; d++) {
    const nodes = cols.get(d) || [];
    const colH = nodes.length * NODE_H + Math.max(0, nodes.length - 1) * NODE_GAP_Y;
    const startY = centerY - colH / 2;
    for (let i = 0; i < nodes.length; i++) {
      positioned.push({
        node: nodes[i],
        x: xForDepth(d),
        y: startY + i * (NODE_H + NODE_GAP_Y),
        depth: d,
      });
    }
  }

  return positioned;
});

/* ------------------------------------------------------------------ */
/*  Graph dimensions                                                    */
/* ------------------------------------------------------------------ */

const maxDepth = computed(() => {
  let max = 0;
  for (const pn of positionedNodes.value) {
    if (pn.depth > max) max = pn.depth;
  }
  return max;
});

const graphWidth = computed(() => {
  if (positionedNodes.value.length === 0) return 0;
  return PAD + (maxDepth.value + 1) * (NODE_W + NODE_GAP_X) - NODE_GAP_X + PAD;
});

const graphHeight = computed(() => {
  if (positionedNodes.value.length === 0) return 0;
  let maxY = 0;
  for (const pn of positionedNodes.value) {
    if (pn.y + NODE_H > maxY) maxY = pn.y + NODE_H;
  }
  return maxY + PAD;
});

const hasNodes = computed(() => positionedNodes.value.length > 0);

/* ------------------------------------------------------------------ */
/*  Edges                                                               */
/* ------------------------------------------------------------------ */

interface Edge {
  key: string;
  path: string;
}

interface EdgeLabel {
  key: string;
  x: number;
  y: number;
  text: string;
}

const edges = computed<Edge[]>(() => {
  const result: Edge[] = [];
  const nodeMap = new Map<string, { x: number; y: number }>();
  for (const pn of positionedNodes.value) {
    nodeMap.set(pn.node.id, { x: pn.x, y: pn.y });
  }

  for (const pn of positionedNodes.value) {
    if (!pn.node.children || pn.node.children.length === 0) continue;
    const px = pn.x + NODE_W;
    const py = pn.y + NODE_H / 2;
    for (const child of pn.node.children) {
      const cp = nodeMap.get(child.id);
      if (!cp) continue;
      const cx = cp.x;
      const cy = cp.y + NODE_H / 2;
      const midX = (px + cx) / 2;
      const path = `M ${px} ${py} C ${midX} ${py}, ${midX} ${cy}, ${cx} ${cy}`;
      result.push({ key: `${pn.node.id}->${child.id}`, path });
    }
  }
  return result;
});

const edgeLabels = computed<EdgeLabel[]>(() => {
  const result: EdgeLabel[] = [];
  const nodeMap = new Map<string, { x: number; y: number }>();
  for (const pn of positionedNodes.value) {
    nodeMap.set(pn.node.id, { x: pn.x, y: pn.y });
  }

  for (const pn of positionedNodes.value) {
    if (!pn.node.children || pn.node.children.length === 0) continue;
    const px = pn.x + NODE_W;
    const py = pn.y + NODE_H / 2;
    for (const child of pn.node.children) {
      const cp = nodeMap.get(child.id);
      if (!cp) continue;
      const cy = cp.y + NODE_H / 2;
      const rel = child.relation_type;
      if (!rel) continue;
      const text = relText(rel);
      if (!text) continue;
      const mx = (px + cp.x) / 2;
      const my = (py + cy) / 2;
      result.push({ key: `el-${pn.node.id}->${child.id}`, x: mx, y: my - 4, text });
    }
  }
  return result;
});

/* ------------------------------------------------------------------ */
/*  Pan / Zoom                                                          */
/* ------------------------------------------------------------------ */

const containerRef = ref<HTMLElement>();
const scale = ref(1);
const panX = ref(0);
const panY = ref(0);

let isPanning = false;
let panStart = { x: 0, y: 0 };

const viewportStyle = computed(() => ({
  transform: `translate(${panX.value}px, ${panY.value}px) scale(${scale.value})`,
  transformOrigin: '0 0',
}));

function onWheel(e: WheelEvent) {
  const delta = e.deltaY > 0 ? 0.92 : 1.08;
  const newScale = Math.min(2.5, Math.max(0.3, scale.value * delta));

  if (containerRef.value) {
    const rect = containerRef.value.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const scaleRatio = newScale / scale.value;
    panX.value = mx - scaleRatio * (mx - panX.value);
    panY.value = my - scaleRatio * (my - panY.value);
  }

  scale.value = newScale;
}

function onPanStart(e: MouseEvent) {
  isPanning = true;
  panStart = { x: e.clientX - panX.value, y: e.clientY - panY.value };
}

function onPanMove(e: MouseEvent) {
  if (!isPanning) return;
  panX.value = e.clientX - panStart.x;
  panY.value = e.clientY - panStart.y;
}

function onPanEnd() {
  isPanning = false;
}

function resetView() {
  scale.value = 1;
  panX.value = 0;
  panY.value = 0;
}

const isPanned = computed(() => scale.value !== 1 || panX.value !== 0 || panY.value !== 0);

/* ------------------------------------------------------------------ */
/*  Helpers                                                             */
/* ------------------------------------------------------------------ */

function isOnCanvas(assetId: string): boolean {
  return onCanvasSet.value.has(assetId);
}

function onNodeClick(assetId: string) {
  emit('focus-asset', assetId);
}

function onNodeDblClick(assetId: string) {
  emit('focus-asset', assetId);
  emit('request-close');
}

function thumbUrl(n: LineageNode): string {
  if (n.thumbnail_path) return `/api/assets/${n.id}/thumbnail`;
  if (n.file_path) return `/api/assets/${n.id}/file`;
  return '';
}

function relText(rt: string | null): string {
  return lineageRelationLabel(rt);
}

function fmtTime(ts: string): string {
  try {
    const d = new Date(ts);
    const m = d.getMonth() + 1;
    const day = d.getDate();
    const h = String(d.getHours()).padStart(2, '0');
    const min = String(d.getMinutes()).padStart(2, '0');
    return `${m}/${day} ${h}:${min}`;
  } catch {
    return '';
  }
}
</script>

<style scoped>
.lineage-graph {
  position: relative;
  width: 100%;
  height: 100%;
  overflow: hidden;
  background: var(--dq-bg-page);
  cursor: grab;
  user-select: none;
}

.lineage-graph:active {
  cursor: grabbing;
}

.lineage-graph__empty {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: var(--dq-label-tertiary);
  font-size: var(--dq-font-size-body);
}

.lineage-graph__viewport {
  position: absolute;
  top: 0;
  left: 0;
  min-width: 100%;
  min-height: 100%;
  transition: none;
}

/* ---- Reset button ---- */

.lineage-graph__reset-btn {
  position: absolute;
  top: 8px;
  right: 8px;
  z-index: 10;
  width: 28px;
  height: 28px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--dq-bg-base);
  border: 0.5px solid var(--dq-border);
  color: var(--dq-label-secondary);
  cursor: pointer;
  opacity: 0.8;
  transition: opacity 0.15s, border-color 0.15s, color 0.15s;
}

.lineage-graph__reset-btn:hover {
  opacity: 1;
  border-color: var(--dq-accent);
  color: var(--dq-accent);
}

.lineage-graph__reset-btn > :deep(svg) {
  width: 14px;
  height: 14px;
}

/* ---- SVG edges ---- */

.lineage-graph__edges {
  position: absolute;
  top: 0;
  left: 0;
  overflow: visible;
  pointer-events: none;
}

.lineage-graph__edge {
  fill: none;
  stroke: var(--dq-border);
  stroke-width: 1.5;
  stroke-linecap: round;
}

.lineage-graph__edge-label {
  fill: var(--dq-label-tertiary);
  font-size: var(--dq-font-size-caption);
  text-anchor: middle;
  pointer-events: none;
}

/* ---- Node cards ---- */

.lineage-graph__node {
  position: absolute;
}

.lineage-graph__node-card {
  display: flex;
  gap: 10px;
  align-items: center;
  width: 100%;
  padding: 10px;
  border-radius: 8px;
  background: var(--dq-bg-base);
  border: 0.5px solid var(--dq-border);
  transition: border-color 0.15s, box-shadow 0.15s;
  height: 100%;
  box-sizing: border-box;
  cursor: pointer;
  text-align: left;
  font: inherit;
  color: inherit;
}

.lineage-graph__node--on-canvas .lineage-graph__node-card {
  border-color: color-mix(in srgb, var(--dq-accent) 35%, var(--dq-border));
}

.lineage-graph__node-card:hover {
  border-color: var(--dq-accent-border-hover);
  box-shadow: 0 0 0 1px var(--dq-accent-ring-subtle);
}

.lineage-graph__node--current .lineage-graph__node-card {
  border-color: var(--dq-accent);
  box-shadow: 0 0 0 2px var(--dq-accent-ring-subtle);
}

.lineage-graph__node--root .lineage-graph__node-card {
  border-style: dashed;
}

.lineage-graph__thumb {
  width: 44px;
  height: 44px;
  border-radius: 6px;
  overflow: hidden;
  flex-shrink: 0;
  background: var(--dq-fill-tertiary);
  display: flex;
  align-items: center;
  justify-content: center;
}

.lineage-graph__thumb-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.lineage-graph__thumb-fallback {
  color: var(--dq-label-tertiary);
}

.lineage-graph__info {
  display: flex;
  flex-direction: column;
  gap: 3px;
  min-width: 0;
  overflow: hidden;
}

.lineage-graph__dim {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-primary);
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.lineage-graph__time {
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-tertiary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.lineage-graph__rel-tag {
  display: inline-block;
  font-size: var(--dq-font-size-caption);
  padding: 1px 6px;
  border-radius: 4px;
  background: var(--dq-accent-tint);
  color: var(--dq-accent);
  line-height: 16px;
  width: fit-content;
}

.lineage-graph__canvas-tag {
  display: inline-block;
  font-size: var(--dq-font-size-caption);
  padding: 1px 5px;
  border-radius: 4px;
  background: color-mix(in srgb, var(--dq-accent) 14%, transparent);
  color: var(--dq-accent);
  line-height: 14px;
  width: fit-content;
}
</style>
