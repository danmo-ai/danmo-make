import type { TraceSpan, TraceStepProgress } from '@/types';
import { $tt } from '@/utils/i18n';

/** Latest `progress` event across exported spans (historical; open-phase events are not exported yet). */
export function stepProgressFromSpans(spans: TraceSpan[] | null | undefined): TraceStepProgress | null {
  if (!spans?.length) return null;
  let best: { time: number; tip: TraceStepProgress } | null = null;
  for (const span of spans) {
    for (const event of span.events ?? []) {
      if (event.name !== 'progress') continue;
      const attrs = (event.attributes ?? {}) as Record<string, unknown>;
      const tip: TraceStepProgress = {
        phase: String(attrs.phase ?? ''),
        step: Number(attrs['progress.step'] ?? 0),
        total: Number(attrs['progress.total'] ?? 0),
      };
      const time = Number(event.time_unix_nanos ?? span.start_unix_nanos ?? 0);
      if (!best || time >= best.time) {
        best = { time, tip };
      }
    }
  }
  return best?.tip ?? null;
}

export function formatStepProgressTitle(tip: TraceStepProgress | null | undefined): string | null {
  if (!tip) return null;
  const phase = String(tip.phase || '').trim();
  const step = Number(tip.step) || 0;
  const total = Number(tip.total) || 0;
  if (total > 0 && step > 0) {
    if (phase === 'denoise' || phase === 'denoising') {
      return $tt('studio.queueDenoiseProgress', { current: step, total });
    }
    if (phase === 'train') {
      return $tt('studio.queueTrainProgress', { current: step, total });
    }
    if (phase === 'lm') {
      return $tt('studio.queueLmProgress', { current: step, total });
    }
    if (phase) {
      return `${phase} ${step} / ${total}`;
    }
    return `${step} / ${total}`;
  }
  if (phase) return phase;
  return null;
}

export function stepProgressPercent(tip: TraceStepProgress | null | undefined): number | null {
  if (!tip) return null;
  const total = Number(tip.total) || 0;
  const step = Number(tip.step) || 0;
  if (total <= 0) return null;
  return Math.min(100, Math.max(0, (step / total) * 100));
}
