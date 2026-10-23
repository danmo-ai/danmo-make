/** Timed lyric lines for synced (scrolling) playback — backend ``lyrics_segments`` or LRC. */
export interface LyricLine {
  start: number;
  end: number;
  text: string;
  label?: string;
}

export type LyricsTimelineMode = 'dit_cross_attention' | 'structure_estimate' | 'lrc' | string;

const LRC_TIME_RE = /\[(\d{1,3}):(\d{1,2}(?:\.\d{1,3})?)\]/g;
const SECTION_ONLY_RE = /^\s*\[[^\]]+\]\s*$/;

function finiteOr(v: unknown, fallback: number): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

/** Sort by start and close open ends at the next line start (or ``duration``). */
function normalize(lines: LyricLine[], duration = 0): LyricLine[] {
  const sorted = lines
    .filter((l) => l.text.trim() && Number.isFinite(l.start))
    .sort((a, b) => a.start - b.start);
  return sorted.map((l, i) => {
    const next = sorted[i + 1]?.start;
    const cap = next ?? (duration > 0 ? duration : l.start + 5);
    const end = l.end > l.start ? Math.min(l.end, cap) : cap;
    return { ...l, end: Math.max(end, l.start) };
  });
}

export function linesFromSegments(segments: unknown, duration = 0): LyricLine[] {
  if (!Array.isArray(segments)) return [];
  const lines: LyricLine[] = [];
  for (const raw of segments) {
    if (!raw || typeof raw !== 'object') continue;
    const s = raw as Record<string, unknown>;
    const text = String(s.text ?? '').trim();
    if (!text || SECTION_ONLY_RE.test(text)) continue;
    const start = finiteOr(s.start_sec, NaN);
    lines.push({
      start,
      end: finiteOr(s.end_sec, start),
      text,
      label: s.label != null ? String(s.label) : undefined,
    });
  }
  return normalize(lines, duration);
}

export function parseLrc(lrc: string, duration = 0): LyricLine[] {
  const lines: LyricLine[] = [];
  for (const row of String(lrc || '').split(/\r?\n/)) {
    const stamps: number[] = [];
    let m: RegExpExecArray | null;
    LRC_TIME_RE.lastIndex = 0;
    let lastIdx = 0;
    while ((m = LRC_TIME_RE.exec(row))) {
      stamps.push(Number(m[1]) * 60 + Number(m[2]));
      lastIdx = LRC_TIME_RE.lastIndex;
    }
    if (!stamps.length) continue;
    const text = row.slice(lastIdx).trim();
    if (!text) continue;
    for (const start of stamps) lines.push({ start, end: start, text });
  }
  return normalize(lines, duration);
}

/** Index of the line playing at ``t`` (last line with ``start <= t``); -1 before the first. */
export function activeLineIndex(lines: LyricLine[], t: number): number {
  let lo = 0;
  let hi = lines.length - 1;
  let ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (lines[mid].start <= t) {
      ans = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return ans;
}

/** 0..1 progress through ``line`` at ``t`` (karaoke-style fill). */
export function lineProgress(line: LyricLine | undefined, t: number): number {
  if (!line) return 0;
  const span = line.end - line.start;
  if (span <= 0) return t >= line.start ? 1 : 0;
  return Math.min(1, Math.max(0, (t - line.start) / span));
}

function lrcStamp(sec: number): string {
  const s = Math.max(0, sec);
  const mm = Math.floor(s / 60);
  const ss = s - mm * 60;
  return `[${String(mm).padStart(2, '0')}:${ss.toFixed(2).padStart(5, '0')}]`;
}

export function linesToLrc(lines: LyricLine[]): string {
  return lines.map((l) => `${lrcStamp(l.start)}${l.text}`).join('\n') + (lines.length ? '\n' : '');
}
