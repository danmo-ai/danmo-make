import type { GalleryItem } from '@/types';
import { assetIdFromGalleryPath } from '@/utils/copilotHandoff';
import { $tt } from '@/utils/i18n';

const SOURCE_ACTION_I18N: Record<string, string> = {
  create: 'gallery.actionCreate',
  rewrite: 'gallery.actionRewrite',
  retouch: 'action.image.retouch',
  extend: 'action.image.extend',
  upscale: 'gallery.actionUpscale',
  animate: 'action.video.animate',
  cover: 'action.audio.cover',
  upload: 'gallery.actionUpload',
  avatar: 'gallery.actionAvatar',
  tool: 'gallery.actionTool',
  preview: 'gallery.actionPreview',
};

export function galleryAssetId(item: GalleryItem | null | undefined): string {
  if (!item?.path) return '';
  return assetIdFromGalleryPath(item.path) || '';
}

export function galleryTaskId(item: GalleryItem | null | undefined): string {
  if (!item) return '';
  const direct = String(item.source_task_id || '').trim();
  if (direct) return direct;
  const meta = item.metadata || {};
  return String(meta.source_task_id || meta.task_id || '').trim();
}

export function gallerySourceAction(item: GalleryItem | null | undefined): string {
  if (!item) return '';
  const direct = String(item.source_action || '').trim();
  if (direct) return direct;
  return String(item.metadata?.source_action || '').trim();
}

export function sourceActionLabel(action: string | null | undefined): string {
  const key = String(action || '').trim();
  if (!key) return '';
  const i18nKey = SOURCE_ACTION_I18N[key];
  if (i18nKey) {
    const translated = $tt(i18nKey);
    if (translated !== i18nKey) return translated;
  }
  return key;
}

export function metaNumber(
  item: GalleryItem | null | undefined,
  ...keys: string[]
): number | null {
  if (!item) return null;
  const meta = item.metadata || {};
  for (const key of keys) {
    const raw = meta[key];
    if (raw == null || raw === '') continue;
    const n = Number(raw);
    if (Number.isFinite(n)) return n;
  }
  return null;
}

export function metaText(
  item: GalleryItem | null | undefined,
  ...keys: string[]
): string {
  if (!item) return '';
  const meta = item.metadata || {};
  for (const key of keys) {
    const raw = meta[key];
    if (raw == null) continue;
    const text = String(raw).trim();
    if (text) return text;
  }
  return '';
}

export type GalleryAdapterMeta = {
  id: string;
  weight: number | null;
  /** Optional filesystem path from task params (for display name). */
  path?: string;
};

/** Prefer LoRA folder name from path when id is opaque (user-lora-…). */
function adapterDisplayName(id: string, path?: string): string {
  const p = String(path || '').trim();
  if (p) {
    const parts = p.split(/[/\\]/).filter(Boolean);
    // …/Lora/<name>/final_adapters.safetensors → <name>
    if (parts.length >= 2) {
      const file = parts[parts.length - 1] || '';
      const parent = parts[parts.length - 2] || '';
      if (/adapters|safetensors|lora/i.test(file) && parent && !/^user-lora-/i.test(parent)) {
        return parent;
      }
      if (file && !/^user-lora-/i.test(file)) return file.replace(/\.safetensors$/i, '');
    }
  }
  if (id.includes('/') || id.includes('\\')) {
    return id.split(/[/\\]/).filter(Boolean).pop() || id;
  }
  return id;
}

/** LoRA / adapter rows stored on asset metadata (or task overlay). */
export function galleryAdapters(
  item: GalleryItem | null | undefined,
  overlay?: Record<string, unknown> | null,
): GalleryAdapterMeta[] {
  const raw =
    overlay?.adapters ??
    item?.metadata?.adapters ??
    item?.metadata?.loras ??
    null;
  if (!Array.isArray(raw)) return [];
  const out: GalleryAdapterMeta[] = [];
  for (const row of raw) {
    if (!row || typeof row !== 'object') continue;
    const rec = row as Record<string, unknown>;
    const id = String(rec.id || rec.name || '').trim();
    if (!id) continue;
    const path = String(rec.path || '').trim() || undefined;
    const weightRaw = rec.weight ?? rec.scale;
    const weight =
      weightRaw == null || weightRaw === ''
        ? null
        : Number.isFinite(Number(weightRaw))
          ? Number(weightRaw)
          : null;
    out.push({ id, weight, path });
  }
  return out;
}

export function formatAdapterLabel(adapter: GalleryAdapterMeta): string {
  const short = adapterDisplayName(adapter.id, adapter.path);
  if (adapter.weight == null) return short;
  const w = Number(adapter.weight);
  const text = Number.isInteger(w) ? String(w) : w.toFixed(2).replace(/\.?0+$/, '');
  return `${short} × ${text}`;
}

export function formatGalleryResolution(item: GalleryItem | null | undefined): string {
  if (!item) return '';
  const w = Number(item.width || item.metadata?.width || 0);
  const h = Number(item.height || item.metadata?.height || 0);
  if (w > 0 && h > 0) return `${w}×${h}`;
  return '';
}

/** SQLite `datetime('now')` is UTC, written as `YYYY-MM-DD HH:MM:SS`. */
const SQLITE_UTC_STAMP = /^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})(\.\d+)?$/;

/**
 * Gallery timestamps come from two writers.
 * Older rows are local ISO (`2026-11-18T16:22:24`). Rust rows from
 * `datetime('now')` are UTC with a space (`2026-11-18 09:03:58`).
 * Parsing the UTC form as local time drops those assets behind older ones.
 */
export function parseGalleryTimestamp(dateStr: string | null | undefined): number {
  if (!dateStr) return NaN;
  const raw = dateStr.trim();
  const sqlite = raw.match(SQLITE_UTC_STAMP);
  if (sqlite) {
    return Date.parse(`${sqlite[1]}T${sqlite[2]}${sqlite[3] || ''}Z`);
  }
  return Date.parse(raw);
}

export function formatGalleryDate(dateStr: string | null | undefined): string {
  const ms = parseGalleryTimestamp(dateStr);
  if (!Number.isFinite(ms)) return dateStr ? String(dateStr) : '';
  return new Date(ms).toLocaleString();
}

export function formatGalleryClock(sec: number | null | undefined): string {
  if (sec == null || !Number.isFinite(Number(sec))) return '';
  const s = Math.max(0, Math.floor(Number(sec)));
  const m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, '0')}`;
}
