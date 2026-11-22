<template>
  <div class="task-trace">
    <div v-if="errorMessage" class="task-trace__notice" role="alert">
      {{ errorMessage }}
    </div>

    <template v-else-if="rows.length === 0">
      <div class="task-trace__empty">
        <p class="task-trace__empty-title">{{ $t('studio.traceEmptyTitle') }}</p>
        <p class="task-trace__empty-hint">
          {{ reason || $t('studio.traceEmptyHint') }}
        </p>
      </div>
    </template>

    <template v-else>
      <header class="task-trace__meta">
        <span class="task-trace__meta-item">
          {{ $t('studio.traceSpans', { count: rows.length }) }}
        </span>
        <span v-if="totalLabel" class="task-trace__meta-item">{{ totalLabel }}</span>
        <button
          v-if="taskId"
          type="button"
          class="task-trace__meta-item task-trace__id-btn"
          :title="$t('studio.traceCopyId')"
          @click="copyText(taskId)"
        >
          {{ taskId }}
        </button>
        <span
          v-if="errorSpanCount > 0"
          class="task-trace__meta-item task-trace__meta-item--error"
        >
          {{ $t('studio.traceErrors', { count: errorSpanCount }) }}
        </span>
      </header>

      <div class="task-trace__split">
        <ol class="task-trace__rows" role="listbox" :aria-label="$t('studio.traceTitle')">
          <li
            v-for="row in rows"
            :key="row.span.span_id"
            class="task-trace__row"
            :class="{
              [`task-trace__row--${row.span.status || 'unset'}`]: true,
              'is-selected': row.span.span_id === selectedId,
            }"
            role="option"
            :aria-selected="row.span.span_id === selectedId"
          >
            <button
              type="button"
              class="task-trace__row-btn"
              @click="selectedId = row.span.span_id"
            >
              <span
                class="task-trace__label"
                :style="{ paddingLeft: `${row.depth * 14}px` }"
              >
                <span
                  class="task-trace__status-dot"
                  :class="`task-trace__status-dot--${row.span.status || 'unset'}`"
                  aria-hidden="true"
                />
                <span class="task-trace__name">{{ spanLabel(row.span) }}</span>
                <span v-if="progressLabel(row.span)" class="task-trace__chip">
                  {{ progressLabel(row.span) }}
                </span>
                <span v-else-if="eventCount(row.span) > 0" class="task-trace__chip">
                  {{ $t('studio.traceEvents', { count: eventCount(row.span) }) }}
                </span>
              </span>
              <span class="task-trace__track" aria-hidden="true">
                <span
                  class="task-trace__bar"
                  :class="`task-trace__bar--${row.span.status || 'unset'}`"
                  :style="{ marginLeft: `${row.offsetPercent}%`, width: `${row.widthPercent}%` }"
                />
              </span>
              <span class="task-trace__duration">{{ formatDuration(row.span.duration_ms) }}</span>
            </button>
          </li>
        </ol>

        <aside
          v-if="selected"
          class="task-trace__detail"
          :aria-label="$t('studio.traceSpanDetail')"
        >
          <header class="task-trace__detail-head">
            <div class="task-trace__detail-title">
              <span
                class="task-trace__status-pill"
                :class="`task-trace__status-pill--${selected.status || 'unset'}`"
              >
                {{ statusLabel(selected.status) }}
              </span>
              <h5>{{ spanLabel(selected) }}</h5>
            </div>
            <p class="task-trace__detail-sub">
              {{ formatDuration(selected.duration_ms) }}
              <template v-if="relativeStart(selected)">
                · +{{ formatDuration(relativeStart(selected)!) }}
              </template>
            </p>
          </header>

          <section
            v-if="selected.status === 'error' && selected.status_message"
            class="task-trace__error"
            role="alert"
          >
            <h6 class="task-trace__section-label">{{ $t('studio.traceError') }}</h6>
            <pre class="task-trace__error-body">{{ selected.status_message }}</pre>
          </section>

          <section class="task-trace__section">
            <h6 class="task-trace__section-label">{{ $t('studio.traceIds') }}</h6>
            <dl class="task-trace__kv">
              <div v-if="taskId" class="task-trace__kv-row">
                <dt>task.id</dt>
                <dd>
                  <button type="button" class="task-trace__id-btn" @click="copyText(taskId)">
                    {{ taskId }}
                  </button>
                </dd>
              </div>
              <div class="task-trace__kv-row">
                <dt>span.id</dt>
                <dd>
                  <button type="button" class="task-trace__id-btn" @click="copyText(selected.span_id)">
                    {{ selected.span_id }}
                  </button>
                </dd>
              </div>
              <div v-if="selected.parent_span_id && selected.parent_span_id !== '0000000000000000'" class="task-trace__kv-row">
                <dt>parent.id</dt>
                <dd>
                  <button
                    type="button"
                    class="task-trace__id-btn"
                    @click="selectOrCopyParent(selected.parent_span_id)"
                  >
                    {{ selected.parent_span_id }}
                  </button>
                </dd>
              </div>
            </dl>
          </section>

          <section class="task-trace__section">
            <h6 class="task-trace__section-label">
              {{ $t('studio.traceTags') }}
              <span v-if="tagEntries.length" class="task-trace__count">{{ tagEntries.length }}</span>
            </h6>
            <p v-if="tagEntries.length === 0" class="task-trace__muted">
              {{ $t('studio.traceTagsEmpty') }}
            </p>
            <dl v-else class="task-trace__kv">
              <div
                v-for="[key, value] in tagEntries"
                :key="key"
                class="task-trace__kv-row"
              >
                <dt :title="key">{{ key }}</dt>
                <dd>
                  <button
                    type="button"
                    class="task-trace__tag-value"
                    :title="$t('studio.traceCopyValue')"
                    @click="copyText(value)"
                  >
                    {{ value }}
                  </button>
                </dd>
              </div>
            </dl>
          </section>

          <section class="task-trace__section">
            <h6 class="task-trace__section-label">
              {{ $t('studio.traceEventList') }}
              <span v-if="selectedEvents.length" class="task-trace__count">
                {{ selectedEvents.length }}
              </span>
            </h6>
            <p v-if="selectedEvents.length === 0" class="task-trace__muted">
              {{ $t('studio.traceEventsEmpty') }}
            </p>
            <ol v-else class="task-trace__events">
              <li
                v-for="(event, index) in selectedEvents"
                :key="`${event.name}-${event.time_unix_nanos}-${index}`"
                class="task-trace__event"
              >
                <div class="task-trace__event-head">
                  <span class="task-trace__event-name">{{ eventDisplayName(event) }}</span>
                  <time class="task-trace__event-time">
                    +{{ formatDuration(eventOffsetMs(selected, event)) }}
                  </time>
                </div>
                <dl v-if="eventAttrEntries(event).length" class="task-trace__kv task-trace__kv--nested">
                  <div
                    v-for="[key, value] in eventAttrEntries(event)"
                    :key="key"
                    class="task-trace__kv-row"
                  >
                    <dt>{{ key }}</dt>
                    <dd>{{ value }}</dd>
                  </div>
                </dl>
              </li>
            </ol>
          </section>
        </aside>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import type { TraceSpan, TraceSpanEvent } from '@/types';
import { formatStepProgressTitle, stepProgressFromSpans } from '@/utils/traceProgress';
import { $tt } from '@/utils/i18n';
import { toast } from '@/utils/feedback';
import { copyTextToClipboard } from '@/utils/clipboard';

const props = defineProps<{
  spans: TraceSpan[];
  /** Product task id — also the OTel TraceId source (UUID octets). */
  taskId?: string | null;
  /** Why the list is empty (from the API), shown verbatim instead of a guess. */
  reason?: string | null;
  /** Load/query failure from the store. */
  errorMessage?: string | null;
}>();

interface WaterfallRow {
  span: TraceSpan;
  depth: number;
  offsetPercent: number;
  widthPercent: number;
}

const selectedId = ref<string | null>(null);

const spanById = computed(() => {
  const map = new Map<string, TraceSpan>();
  for (const span of props.spans) map.set(span.span_id, span);
  return map;
});

const rows = computed<WaterfallRow[]>(() => {
  const spans = [...props.spans].sort((left, right) => {
    if (left.start_unix_nanos !== right.start_unix_nanos) {
      return left.start_unix_nanos - right.start_unix_nanos;
    }
    return right.duration_ms - left.duration_ms;
  });
  if (spans.length === 0) return [];

  const origin = spans[0].start_unix_nanos;
  const end = spans.reduce(
    (latest, span) => Math.max(latest, span.start_unix_nanos + span.duration_ms * 1_000_000),
    origin,
  );
  const total = Math.max(end - origin, 1);
  const depthOf = (span: TraceSpan, guard = 0): number => {
    if (guard > 32) return 0;
    const parent = span.parent_span_id;
    if (!parent || parent === '0000000000000000' || !spanById.value.has(parent)) return 0;
    const parentSpan = spanById.value.get(parent);
    return parentSpan ? depthOf(parentSpan, guard + 1) + 1 : 0;
  };

  return spans.map((span) => {
    const offsetPercent = ((span.start_unix_nanos - origin) / total) * 100;
    const widthPercent = Math.max((span.duration_ms * 1_000_000 / total) * 100, 0.6);
    return {
      span,
      depth: depthOf(span),
      offsetPercent: Math.min(offsetPercent, 99.4),
      widthPercent: Math.min(widthPercent, 100 - Math.min(offsetPercent, 99.4)),
    };
  });
});

const selected = computed(() =>
  selectedId.value ? spanById.value.get(selectedId.value) ?? null : null,
);

const taskId = computed(() => props.taskId || null);

const totalLabel = computed(() => {
  const root = props.spans.find((span) =>
    !span.parent_span_id || span.parent_span_id === '0000000000000000',
  ) ?? props.spans[0];
  if (!root) return null;
  return formatDuration(root.duration_ms);
});

const errorSpanCount = computed(
  () => props.spans.filter((span) => span.status === 'error').length,
);

const tagEntries = computed(() => attributeEntries(selected.value?.attributes));

const selectedEvents = computed(() => {
  const events = selected.value?.events;
  if (!Array.isArray(events)) return [] as TraceSpanEvent[];
  return [...events].sort((a, b) => a.time_unix_nanos - b.time_unix_nanos);
});

const originNanos = computed(() => {
  if (props.spans.length === 0) return 0;
  return Math.min(...props.spans.map((span) => span.start_unix_nanos));
});

watch(
  () => props.spans.map((span) => span.span_id).join(','),
  () => {
    if (selectedId.value && spanById.value.has(selectedId.value)) return;
    const failed = props.spans.find((span) => span.status === 'error');
    const root = props.spans.find((span) =>
      !span.parent_span_id || span.parent_span_id === '0000000000000000',
    );
    selectedId.value = failed?.span_id ?? root?.span_id ?? props.spans[0]?.span_id ?? null;
  },
  { immediate: true },
);

function spanLabel(span: TraceSpan): string {
  if (span.name.startsWith('phase.')) return span.name.slice('phase.'.length);
  return span.name;
}

function eventCount(span: TraceSpan): number {
  return Array.isArray(span.events) ? span.events.length : 0;
}

function progressLabel(span: TraceSpan): string | null {
  return formatStepProgressTitle(stepProgressFromSpans([span]));
}

function statusLabel(status: string | undefined): string {
  switch (status) {
    case 'error':
      return $tt('studio.traceStatusError');
    case 'ok':
      return $tt('studio.traceStatusOk');
    case 'unset':
      return $tt('studio.traceStatusUnset');
    default:
      return status || $tt('studio.traceStatusUnset');
  }
}

function relativeStart(span: TraceSpan): number | null {
  if (!originNanos.value) return null;
  return Math.max(0, (span.start_unix_nanos - originNanos.value) / 1_000_000);
}

function eventOffsetMs(span: TraceSpan, event: TraceSpanEvent): number {
  return Math.max(0, (event.time_unix_nanos - span.start_unix_nanos) / 1_000_000);
}

function attributeEntries(attrs: Record<string, unknown> | null | undefined): Array<[string, string]> {
  if (!attrs || typeof attrs !== 'object') return [];
  return Object.entries(attrs)
    .map(([key, value]) => [key, formatAttrValue(value)] as [string, string])
    .sort(([a], [b]) => a.localeCompare(b));
}

function eventAttrEntries(event: TraceSpanEvent): Array<[string, string]> {
  return attributeEntries(event.attributes);
}

function eventDisplayName(event: TraceSpanEvent): string {
  if (event.name === 'progress') {
    const tip = formatStepProgressTitle({
      phase: String(event.attributes?.phase ?? ''),
      step: Number(event.attributes?.['progress.step'] ?? 0),
      total: Number(event.attributes?.['progress.total'] ?? 0),
    });
    if (tip) return tip;
  }
  if (event.name === 'log') {
    const message = event.attributes?.['log.message'];
    if (message != null) return String(message);
  }
  return event.name;
}

function formatAttrValue(value: unknown): string {
  if (value == null) return '—';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function formatDuration(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return '—';
  if (ms < 1) return '<1ms';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(2)}s`;
  const minutes = Math.floor(ms / 60_000);
  const seconds = Math.round((ms % 60_000) / 1000);
  return `${minutes}m ${seconds}s`;
}

function selectOrCopyParent(parentId: string) {
  if (spanById.value.has(parentId)) {
    selectedId.value = parentId;
    return;
  }
  void copyText(parentId);
}

async function copyText(text: string) {
  const ok = await copyTextToClipboard(text);
  if (ok) toast.success($tt('gallery.copied'));
  else toast.error($tt('gallery.copyFailed'));
}
</script>

<style scoped>
.task-trace {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-width: 0;
}

.task-trace__notice {
  padding: 10px 12px;
  border: 1px solid color-mix(in srgb, var(--dq-danger, #ff453a) 40%, transparent);
  border-radius: var(--dq-radius-control-sm);
  color: var(--dq-label-primary);
  background: color-mix(in srgb, var(--dq-danger, #ff453a) 10%, transparent);
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
}

.task-trace__empty {
  padding: 24px 12px;
  text-align: center;
}

.task-trace__empty-title {
  margin: 0 0 4px;
  color: var(--dq-label-secondary);
  font-size: var(--dq-font-size-body);
  font-weight: 500;
}

.task-trace__empty-hint {
  margin: 0;
  color: var(--dq-label-tertiary);
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
}

.task-trace__meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px 12px;
  color: var(--dq-label-tertiary);
  font-size: var(--dq-font-size-caption);
  font-variant-numeric: tabular-nums;
}

.task-trace__meta-item--error {
  color: var(--dq-danger, #ff453a);
  font-weight: 500;
}

.task-trace__split {
  display: grid;
  grid-template-columns: minmax(0, 1.15fr) minmax(240px, 0.95fr);
  gap: 12px;
  min-height: 0;
  align-items: start;
}

.task-trace__rows {
  display: flex;
  flex-direction: column;
  gap: 1px;
  margin: 0;
  padding: 4px;
  list-style: none;
  border: 1px solid var(--dq-border-subtle);
  border-radius: var(--dq-radius-control-sm);
  background: var(--dq-surface-inset, var(--dq-fill-tertiary));
  max-height: min(52vh, 520px);
  overflow: auto;
}

.task-trace__row-btn {
  display: grid;
  grid-template-columns: minmax(88px, 1.35fr) minmax(64px, 1.8fr) 56px;
  align-items: center;
  gap: 8px;
  width: 100%;
  min-height: 28px;
  padding: 4px 8px;
  border: 0;
  border-radius: 6px;
  background: transparent;
  color: inherit;
  cursor: pointer;
  font: inherit;
  text-align: left;
}

.task-trace__row.is-selected .task-trace__row-btn,
.task-trace__row-btn:hover {
  background: var(--dq-fill-tertiary);
}

.task-trace__row.is-selected .task-trace__row-btn {
  outline: 1px solid color-mix(in srgb, var(--dq-accent) 45%, transparent);
}

.task-trace__label {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
}

.task-trace__status-dot {
  flex: 0 0 auto;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--dq-label-tertiary);
}

.task-trace__status-dot--ok {
  background: var(--dq-success, #30d158);
}

.task-trace__status-dot--error {
  background: var(--dq-danger, #ff453a);
}

.task-trace__status-dot--unset {
  background: var(--dq-accent);
}

.task-trace__name {
  overflow: hidden;
  color: var(--dq-label-primary);
  font-size: var(--dq-font-size-caption);
  font-weight: 500;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.task-trace__row--error .task-trace__name {
  color: var(--dq-danger, #ff453a);
}

.task-trace__chip {
  flex-shrink: 0;
  color: var(--dq-label-tertiary);
  font-size: 10px;
  font-variant-numeric: tabular-nums;
}

.task-trace__track {
  position: relative;
  height: 8px;
  border-radius: 4px;
  background: color-mix(in srgb, var(--dq-label-tertiary) 18%, transparent);
  overflow: hidden;
}

.task-trace__bar {
  display: block;
  height: 100%;
  border-radius: 4px;
  background: var(--dq-accent);
}

.task-trace__bar--error {
  background: var(--dq-danger, #ff453a);
}

.task-trace__bar--ok {
  background: color-mix(in srgb, var(--dq-accent) 78%, var(--dq-success, #30d158));
}

.task-trace__bar--unset {
  background: var(--dq-label-tertiary);
}

.task-trace__duration {
  color: var(--dq-label-secondary);
  font-size: var(--dq-font-size-caption);
  font-variant-numeric: tabular-nums;
  text-align: right;
}

.task-trace__detail {
  display: flex;
  flex-direction: column;
  gap: 14px;
  min-width: 0;
  max-height: min(52vh, 520px);
  overflow: auto;
  padding: 12px 14px;
  border: 1px solid var(--dq-border-subtle);
  border-radius: var(--dq-radius-control-sm);
  background: var(--dq-bg-base);
}

.task-trace__detail-head {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--dq-border-subtle);
}

.task-trace__detail-title {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.task-trace__detail-title h5 {
  margin: 0;
  color: var(--dq-label-primary);
  font-size: var(--dq-font-size-body);
  font-weight: 600;
  line-height: 1.3;
  word-break: break-word;
}

.task-trace__detail-sub {
  margin: 0;
  color: var(--dq-label-tertiary);
  font-size: var(--dq-font-size-caption);
  font-variant-numeric: tabular-nums;
}

.task-trace__status-pill {
  display: inline-flex;
  align-items: center;
  padding: 2px 7px;
  border-radius: 999px;
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.02em;
  text-transform: uppercase;
  color: var(--dq-label-secondary);
  background: var(--dq-fill-tertiary);
}

.task-trace__status-pill--ok {
  color: var(--dq-success, #30d158);
  background: color-mix(in srgb, var(--dq-success, #30d158) 14%, transparent);
}

.task-trace__status-pill--error {
  color: var(--dq-danger, #ff453a);
  background: color-mix(in srgb, var(--dq-danger, #ff453a) 14%, transparent);
}

.task-trace__status-pill--unset {
  color: var(--dq-accent);
  background: color-mix(in srgb, var(--dq-accent) 14%, transparent);
}

.task-trace__section,
.task-trace__error {
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-width: 0;
}

.task-trace__section-label {
  display: flex;
  align-items: center;
  gap: 6px;
  margin: 0;
  color: var(--dq-label-tertiary);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: var(--dq-tracking-wider, 0.04em);
  text-transform: uppercase;
}

.task-trace__count {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 16px;
  padding: 0 5px;
  border-radius: 999px;
  background: var(--dq-fill-on-glass, var(--dq-fill-tertiary));
  color: var(--dq-label-secondary);
  font-size: 10px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  text-transform: none;
  letter-spacing: 0;
}

.task-trace__muted {
  margin: 0;
  color: var(--dq-label-tertiary);
  font-size: var(--dq-font-size-caption);
}

.task-trace__error {
  padding: 10px 12px;
  border-radius: var(--dq-radius-control-sm);
  background: color-mix(in srgb, var(--dq-danger, #ff453a) 10%, transparent);
  border: 1px solid color-mix(in srgb, var(--dq-danger, #ff453a) 28%, transparent);
}

.task-trace__error-body {
  margin: 0;
  color: var(--dq-danger, #ff453a);
  font-family: var(--dq-font-mono, ui-monospace, SFMono-Regular, Menlo, monospace);
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
  white-space: pre-wrap;
  word-break: break-word;
}

.task-trace__kv {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin: 0;
}

.task-trace__kv-row {
  display: grid;
  grid-template-columns: minmax(72px, 0.85fr) minmax(0, 1.4fr);
  gap: 8px;
  align-items: start;
}

.task-trace__kv-row dt {
  overflow: hidden;
  color: var(--dq-label-tertiary);
  font-family: var(--dq-font-mono, ui-monospace, SFMono-Regular, Menlo, monospace);
  font-size: 11px;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.task-trace__kv-row dd {
  margin: 0;
  min-width: 0;
  color: var(--dq-label-primary);
  font-size: var(--dq-font-size-caption);
  line-height: 1.4;
  word-break: break-word;
}

.task-trace__kv--nested {
  margin-top: 4px;
  padding-left: 2px;
}

.task-trace__id-btn,
.task-trace__tag-value {
  display: inline;
  padding: 0;
  border: 0;
  background: transparent;
  color: inherit;
  cursor: pointer;
  font: inherit;
  text-align: left;
  word-break: break-all;
}

.task-trace__id-btn {
  font-family: var(--dq-font-mono, ui-monospace, SFMono-Regular, Menlo, monospace);
  font-size: 11px;
  color: var(--dq-label-secondary);
}

.task-trace__id-btn:hover,
.task-trace__tag-value:hover {
  color: var(--dq-accent);
}

.task-trace__events {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.task-trace__event {
  padding: 8px 10px;
  border-radius: var(--dq-radius-control-sm);
  background: var(--dq-fill-tertiary);
}

.task-trace__event-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
}

.task-trace__event-name {
  min-width: 0;
  color: var(--dq-label-primary);
  font-size: var(--dq-font-size-caption);
  font-weight: 500;
  line-height: 1.4;
  word-break: break-word;
}

.task-trace__event-time {
  flex-shrink: 0;
  color: var(--dq-label-tertiary);
  font-size: 10px;
  font-variant-numeric: tabular-nums;
}

@media (max-width: 1080px) {
  .task-trace__split {
    grid-template-columns: 1fr;
  }

  .task-trace__rows,
  .task-trace__detail {
    max-height: none;
  }
}
</style>
