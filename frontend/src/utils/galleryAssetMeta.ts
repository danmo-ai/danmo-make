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

export function formatGalleryResolution(item: GalleryItem | null | undefined): string {
  if (!item) return '';
  const w = Number(item.width || item.metadata?.width || 0);
  const h = Number(item.height || item.metadata?.height || 0);
  if (w > 0 && h > 0) return `${w}×${h}`;
  return '';
}

export function formatGalleryDate(dateStr: string | null | undefined): string {
  if (!dateStr) return '';
  try {
    return new Date(dateStr).toLocaleString();
  } catch {
    return String(dateStr);
  }
}

export function formatGalleryClock(sec: number | null | undefined): string {
  if (sec == null || !Number.isFinite(Number(sec))) return '';
  const s = Math.max(0, Math.floor(Number(sec)));
  const m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, '0')}`;
}
