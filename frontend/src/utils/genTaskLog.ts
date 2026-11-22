import { $tt } from '@/utils/i18n';

export type LogDisplayKind = 'milestone' | 'progress' | 'technical' | 'error' | 'warning';

export interface LogDisplayChip {
  key: string;
  label: string;
  value: string;
}

export interface LogDisplayItem {
  index: number;
  time: string;
  kind: LogDisplayKind;
  title: string;
  detail?: string;
  chips?: LogDisplayChip[];
}

/** 从日志行或进度文案解析去噪步 key，如 "3/10"。 */
export function parseDenoiseStepKey(msg: string): string | null {
  const text = String(msg || '').trim();
  const en = text.match(/^Step\s+(\d+)\s*\/\s*(\d+)/i);
  if (en) return `${en[1]}/${en[2]}`;
  const zh = text.match(/去噪\s*(\d+)\s*\/\s*(\d+)/);
  if (zh) return `${zh[1]}/${zh[2]}`;
  return null;
}

/** 将后端英文 Step 行统一为 i18n 去噪进度文案。 */
export function formatGenLogMessage(raw: string): string {
  const key = parseDenoiseStepKey(raw);
  if (!key) return raw;
  const [current, total] = key.split('/');
  return $tt('studio.queueDenoiseProgress', { current, total });
}

export function isDuplicateDenoiseStepLog(
  recentLogs: { message: string }[],
  rawMessage: string,
): boolean {
  const key = parseDenoiseStepKey(rawMessage);
  if (!key) return false;
  const last = recentLogs[recentLogs.length - 1];
  if (!last) return false;
  return parseDenoiseStepKey(last.message) === key;
}

export function progressPhaseLabel(phase: string, message: string): string {
  const key =
    phase ||
    (message === 'denoise' ? 'denoising' : message === 'post' ? 'decoding' : message);
  if (!key || key === 'denoise' || key === 'post') {
    if (key === 'denoise') return $tt('studio.phase.denoising');
    if (key === 'post') return $tt('studio.phase.decoding');
    return '';
  }
  const i18nKey = `studio.phase.${key}`;
  const out = $tt(i18nKey);
  return out !== i18nKey ? out : '';
}

export function formatLogTimestamp(): string {
  const now = new Date();
  return (
    String(now.getHours()).padStart(2, '0') +
    ':' +
    String(now.getMinutes()).padStart(2, '0') +
    ':' +
    String(now.getSeconds()).padStart(2, '0')
  );
}

const GRAPH_STEP_RE = /^\[([^\]]+)\]\s*(.*)$/;

const TECHNICAL_HINT_RE =
  /^(infer |vae_|preview |batch |sigma_|t_embed|scheduler_|cfg_|supports_|image_seq_len|use_empirical|requires_sigma)/i;

function parseGraphStep(message: string): { node: string; detail: string } | null {
  const m = String(message || '').trim().match(GRAPH_STEP_RE);
  if (!m) return null;
  return { node: m[1], detail: (m[2] || '').trim() };
}

function graphStepTitle(node: string): string {
  const key = `studio.logGraph.${node}`;
  const out = $tt(key);
  return out !== key ? out : node;
}

function isPhaseMilestoneMessage(message: string): boolean {
  const phases = ['encoding', 'loading_model', 'denoising', 'decoding', 'saving'];
  return phases.some((p) => {
    const label = $tt(`studio.phase.${p}`);
    return label && message === label;
  });
}

export function parseInferParams(message: string): Record<string, string> | null {
  const text = String(message || '').trim();
  if (!text.startsWith('infer ')) return null;
  const out: Record<string, string> = {};
  for (const token of text.split(/\s+/)) {
    const eq = token.indexOf('=');
    if (eq > 0) {
      out[token.slice(0, eq)] = token.slice(eq + 1);
    }
  }
  return Object.keys(out).length > 0 ? out : null;
}

const INFERENCE_PLAN_RE = /^\[inference\]\s*(.+)$/;

/** Parse ``[inference] family=z_image steps=28 teacache=quality …`` task log lines. */
export function parseInferencePlanLog(message: string): Record<string, string> | null {
  const m = String(message || '').trim().match(INFERENCE_PLAN_RE);
  if (!m) return null;
  const out: Record<string, string> = {};
  for (const token of m[1].split(/\s+/)) {
    const eq = token.indexOf('=');
    if (eq > 0) {
      out[token.slice(0, eq)] = token.slice(eq + 1);
    }
  }
  return Object.keys(out).length > 0 ? out : null;
}

const TEACACHE_SUMMARY_RE =
  /^TeaCache skipped (\d+)\/(\d+) steps \((\d+)%\), thresh=([\d.]+)$/;

/** Parse TeaCache skip summary emitted after denoise. */
export function parseTeacacheSummaryLog(message: string): Record<string, string> | null {
  const m = String(message || '').trim().match(TEACACHE_SUMMARY_RE);
  if (!m) return null;
  return {
    skipped: m[1],
    total: m[2],
    skip_pct: m[3],
    thresh: m[4],
  };
}

function formatTeacacheModeLabel(mode: string): string {
  const aliases: Record<string, string> = {
    none: 'studio.teacacheOff',
    auto: 'studio.teacacheAuto',
    quality: 'studio.teacacheQuality',
    medium: 'studio.teacacheMedium',
    fast: 'studio.teacacheFast',
  };
  const i18nKey = aliases[mode] || `studio.teacache${mode.charAt(0).toUpperCase()}${mode.slice(1)}`;
  const out = $tt(i18nKey);
  return out !== i18nKey ? out : mode;
}

function inferencePlanChips(params: Record<string, string>): LogDisplayChip[] {
  const order: { key: string; labelKey: string; format?: (v: string) => string }[] = [
    { key: 'teacache', labelKey: 'studio.logChip.teacacheMode', format: formatTeacacheModeLabel },
    { key: 'steps', labelKey: 'studio.logChip.steps' },
    { key: 'batched_cfg', labelKey: 'studio.logChip.batchedCfg', format: (v) => (v === 'on' ? 'on' : v) },
    { key: 'mlx_compile', labelKey: 'studio.logChip.mlxCompile', format: (v) => (v === 'on' ? 'on' : v) },
    { key: 'lemica', labelKey: 'studio.logChip.lemicaMode' },
    { key: 'preview', labelKey: 'studio.logChip.previewMode' },
    { key: 'attn', labelKey: 'studio.logChip.attentionBackend' },
  ];
  const chips: LogDisplayChip[] = [];
  for (const { key, labelKey, format } of order) {
    const value = params[key];
    if (value == null || value === '') continue;
    const label = $tt(labelKey);
    chips.push({
      key,
      label: label !== labelKey ? label : key,
      value: format ? format(value) : value,
    });
  }
  return chips;
}

function teacacheSummaryChips(params: Record<string, string>): LogDisplayChip[] {
  const order: { key: string; labelKey: string }[] = [
    { key: 'skipped', labelKey: 'studio.logChip.teacacheSkipped' },
    { key: 'skip_pct', labelKey: 'studio.logChip.teacacheSkipRate' },
    { key: 'thresh', labelKey: 'studio.logChip.teacacheThresh' },
  ];
  const chips: LogDisplayChip[] = [];
  for (const { key, labelKey } of order) {
    const raw = params[key];
    if (raw == null || raw === '') continue;
    const label = $tt(labelKey);
    let value = raw;
    if (key === 'skipped' && params.total) {
      value = `${raw}/${params.total}`;
    } else if (key === 'skip_pct') {
      value = `${raw}%`;
    }
    chips.push({
      key,
      label: label !== labelKey ? label : key,
      value,
    });
  }
  return chips;
}

/** Build TeaCache summary log line from task/asset result metadata (SSE ``result`` event). */
export function buildInferenceResultLogMessage(
  metadata: Record<string, unknown> | null | undefined,
): string | null {
  if (!metadata || typeof metadata !== 'object') return null;
  const skippedRaw = metadata.teacache_skipped;
  const computedRaw = metadata.teacache_computed;
  if (skippedRaw == null && computedRaw == null) return null;
  const skipped = Number(skippedRaw) || 0;
  const computed = Number(computedRaw) || 0;
  const total = skipped + computed;
  if (total <= 0) return null;
  const rateRaw = metadata.teacache_skip_rate;
  const pct =
    rateRaw != null && Number.isFinite(Number(rateRaw))
      ? Math.round(Number(rateRaw) * 100)
      : Math.round((skipped / total) * 100);
  const thresh =
    metadata.teacache_thresh != null ? Number(metadata.teacache_thresh).toFixed(3) : '0.200';
  return `TeaCache skipped ${skipped}/${total} steps (${pct}%), thresh=${thresh}`;
}

function inferParamChips(params: Record<string, string>): LogDisplayChip[] {
  const order: { key: string; labelKey: string }[] = [
    { key: 'model', labelKey: 'studio.logChip.model' },
    { key: 'size', labelKey: 'studio.logChip.size' },
    { key: 'steps', labelKey: 'studio.logChip.steps' },
    { key: 'guidance', labelKey: 'studio.logChip.guidance' },
    { key: 'base_seed', labelKey: 'studio.logChip.seed' },
  ];
  const chips: LogDisplayChip[] = [];
  for (const { key, labelKey } of order) {
    const value = params[key];
    if (value == null || value === '') continue;
    const labelKeyFull = labelKey;
    const label = $tt(labelKeyFull);
    chips.push({
      key,
      label: label !== labelKeyFull ? label : key,
      value,
    });
  }
  return chips;
}

export function classifyLogEntry(message: string, level: string): LogDisplayKind {
  const lvl = String(level || 'info').toLowerCase();
  if (lvl === 'error') return 'error';
  if (lvl === 'warning') return 'warning';
  if (lvl === 'success') return 'milestone';
  if (parseDenoiseStepKey(message)) return 'progress';
  if (message === $tt('studio.queuePostProcessHint')) return 'progress';
  if (isPhaseMilestoneMessage(message)) return 'milestone';
  if (message === $tt('studio.startingGen') || message === $tt('studio.genComplete')) {
    return 'milestone';
  }
  if (parseGraphStep(message)) return 'milestone';
  if (parseInferParams(message)) return 'milestone';
  if (parseInferencePlanLog(message)) return 'milestone';
  if (parseTeacacheSummaryLog(message)) return 'milestone';
  if (/^preview step /i.test(message)) return 'progress';
  if (TECHNICAL_HINT_RE.test(message)) return 'technical';
  return 'info' === lvl ? 'technical' : 'milestone';
}

function buildDisplayTitle(message: string, kind: LogDisplayKind): string {
  const graph = parseGraphStep(message);
  if (graph) return graphStepTitle(graph.node);
  const infer = parseInferParams(message);
  if (infer) return $tt('studio.logInferTitle');
  const plan = parseInferencePlanLog(message);
  if (plan) return $tt('studio.logInferencePlanTitle');
  const teacache = parseTeacacheSummaryLog(message);
  if (teacache) return $tt('studio.logTeacacheSummaryTitle');
  if (kind === 'progress' && parseDenoiseStepKey(message)) {
    return formatGenLogMessage(message);
  }
  if (/^preview step /i.test(message)) {
    const m = message.match(/preview step (\d+)\/(\d+)/i);
    if (m) {
      return $tt('studio.livingCanvasStep', { step: m[1], total: m[2] });
    }
  }
  return message;
}

function buildDisplayDetail(
  message: string,
  kind: LogDisplayKind,
  showTechnical: boolean,
  title: string,
): string | undefined {
  if (!showTechnical && kind === 'technical') return undefined;
  let detail: string | undefined;
  const graph = parseGraphStep(message);
  if (graph) {
    const d = graph.detail;
    detail = !d || d === 'start' ? undefined : d;
  } else if (parseInferParams(message)) {
    detail = showTechnical ? message : undefined;
  } else if (parseInferencePlanLog(message) || parseTeacacheSummaryLog(message)) {
    detail = showTechnical ? message : undefined;
  } else if (kind === 'technical') {
    detail = message;
  } else if (kind === 'milestone' || kind === 'progress') {
    detail = showTechnical ? message : undefined;
  } else {
    detail = showTechnical ? message : undefined;
  }
  if (detail && detail.trim() === title.trim()) return undefined;
  return detail;
}

export function buildLogDisplayItems(
  entries: { time: string; message: string; level: string }[],
  showTechnical: boolean,
): LogDisplayItem[] {
  const items: LogDisplayItem[] = [];
  entries.forEach((entry, index) => {
    const kind = classifyLogEntry(entry.message, entry.level);
    if (kind === 'technical' && !showTechnical) return;

    const infer = parseInferParams(entry.message);
    const plan = parseInferencePlanLog(entry.message);
    const teacache = parseTeacacheSummaryLog(entry.message);
    const title = buildDisplayTitle(entry.message, kind);
    items.push({
      index,
      time: entry.time,
      kind,
      title,
      detail: buildDisplayDetail(entry.message, kind, showTechnical, title),
      chips: infer
        ? inferParamChips(infer)
        : plan
          ? inferencePlanChips(plan)
          : teacache
            ? teacacheSummaryChips(teacache)
            : undefined,
    });
  });
  return items;
}

export function latestProgressItem(items: LogDisplayItem[]): LogDisplayItem | null {
  for (let i = items.length - 1; i >= 0; i -= 1) {
    if (items[i].kind === 'progress') return items[i];
  }
  return null;
}

/** 简洁模式下隐藏逐步进度行（顶部状态条已展示最新进度）。 */
export function filterLogTimelineItems(
  items: LogDisplayItem[],
  showTechnical: boolean,
  excludeIndex?: number | null,
  options?: { hidePipelineMilestones?: boolean },
): LogDisplayItem[] {
  let filtered = items;
  if (excludeIndex != null) {
    filtered = filtered.filter((item) => item.index !== excludeIndex);
  }
  if (!showTechnical) {
    filtered = filtered.filter((item) => item.kind !== 'progress');
  }
  if (options?.hidePipelineMilestones) {
    filtered = filtered.filter((item) => {
      if (item.kind !== 'milestone') return true;
      return Boolean(item.chips?.length);
    });
  }
  return filtered;
}

export function parseProgressFraction(
  title: string,
): { current: number; total: number } | null {
  const key = parseDenoiseStepKey(title);
  if (!key) return null;
  const [currentRaw, totalRaw] = key.split('/');
  const current = Number(currentRaw);
  const total = Number(totalRaw);
  if (!Number.isFinite(current) || !Number.isFinite(total) || total <= 0) return null;
  return { current, total };
}

export function resolveDisplayProgressPercent(
  graphProgress: number | null | undefined,
  live: { progress?: number; step?: number; total?: number } | null | undefined,
  activeTitle: string | null | undefined,
): number | null {
  if (live?.step != null && live.total != null && live.total > 0) {
    return Math.min(100, Math.max(0, (live.step / live.total) * 100));
  }
  if (typeof live?.progress === 'number' && live.progress > 0) {
    return Math.min(100, Math.max(0, live.progress * 100));
  }
  if (activeTitle) {
    const frac = parseProgressFraction(activeTitle);
    if (frac) {
      return Math.min(100, Math.max(0, (frac.current / frac.total) * 100));
    }
  }
  if (typeof graphProgress === 'number' && graphProgress > 0) {
    return Math.min(100, Math.max(0, graphProgress * 100));
  }
  return null;
}
