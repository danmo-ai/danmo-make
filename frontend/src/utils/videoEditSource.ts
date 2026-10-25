import type { GalleryItem } from '@/types';
import { isVideoGalleryItem } from '@/utils/canvasAssets';

export type VideoEditSourceMode = 'first_frame' | 'image_only' | 'source_video';

export type AnimateReferenceAcceptKind = 'image' | 'video' | 'image_or_video';

const BERNINI_MAX_REFERENCE_IMAGES = 5;

export function berniniMaxReferenceImages(): number {
  return BERNINI_MAX_REFERENCE_IMAGES;
}

export function resolveVideoEditSourceMode(
  parameters: Record<string, unknown> | undefined | null,
): VideoEditSourceMode {
  const raw = String(parameters?.video_edit_source_mode || 'image_only');
  if (raw === 'source_video') return 'source_video';
  if (raw === 'first_frame') return 'first_frame';
  return 'image_only';
}

export function isBerniniRenderer(
  parameters: Record<string, unknown> | undefined | null,
): boolean {
  return Boolean(parameters?.bernini_renderer);
}

export function supportsBerniniReferenceImages(
  parameters: Record<string, unknown> | undefined | null,
): boolean {
  return isBerniniRenderer(parameters);
}

export type Ref2VAKind = 'image' | 'video' | 'audio';

export type Ref2VASpec = {
  enabled: boolean;
  maxTotal: number;
  maxImages: number;
  maxVideos: number;
  maxAudios: number;
  kinds: Ref2VAKind[];
  pairedAudio: boolean;
  backgroundIsLastImage: boolean;
  hintKey: string;
};

const REF2VA_KINDS: ReadonlySet<string> = new Set(['image', 'video', 'audio']);

function asNonNegInt(value: unknown, fallback: number): number {
  const n = Number(value);
  if (!Number.isFinite(n) || n < 0) return fallback;
  return Math.floor(n);
}

function parseRef2VAKinds(raw: unknown): Ref2VAKind[] {
  if (!Array.isArray(raw)) return [];
  const out: Ref2VAKind[] = [];
  for (const item of raw) {
    const kind = String(item || '').trim().toLowerCase();
    if (REF2VA_KINDS.has(kind) && !out.includes(kind as Ref2VAKind)) {
      out.push(kind as Ref2VAKind);
    }
  }
  return out;
}

/** Registry-driven Ref2VA capability. Missing / disabled → ``enabled: false``. */
export function ref2vaSpec(
  parameters: Record<string, unknown> | undefined | null,
): Ref2VASpec {
  const raw = parameters?.ref2va;
  const empty: Ref2VASpec = {
    enabled: false,
    maxTotal: 0,
    maxImages: 0,
    maxVideos: 0,
    maxAudios: 0,
    kinds: [],
    pairedAudio: false,
    backgroundIsLastImage: false,
    hintKey: 'create.ref2vaHintPacked',
  };
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return empty;
  const cfg = raw as Record<string, unknown>;
  if (cfg.enabled === false) return empty;
  const kinds = parseRef2VAKinds(cfg.kinds);
  if (!kinds.length) return empty;
  const maxImages = kinds.includes('image') ? asNonNegInt(cfg.max_images, 9) : 0;
  const maxVideos = kinds.includes('video') ? asNonNegInt(cfg.max_videos, 3) : 0;
  const maxAudios = kinds.includes('audio') ? asNonNegInt(cfg.max_audios, 3) : 0;
  const sum = maxImages + maxVideos + maxAudios;
  const maxTotal = Math.max(1, asNonNegInt(cfg.max_total, sum || 1));
  const hintKey = String(cfg.hint_key || '').trim() || 'create.ref2vaHintPacked';
  return {
    enabled: true,
    maxTotal,
    maxImages,
    maxVideos,
    maxAudios,
    kinds,
    pairedAudio: Boolean(cfg.paired_audio),
    backgroundIsLastImage: Boolean(cfg.background_is_last_image),
    hintKey,
  };
}

export function supportsRef2VA(
  parameters: Record<string, unknown> | undefined | null,
): boolean {
  return ref2vaSpec(parameters).enabled;
}

/** Subject slots available for paired audio (excludes optional background still). */
export function ref2vaSubjectCount(imageCount: number, spec: Ref2VASpec): number {
  if (imageCount < 1) return 0;
  if (spec.backgroundIsLastImage && imageCount >= spec.maxImages && spec.maxImages > 0) {
    return Math.max(0, spec.maxImages - 1);
  }
  return imageCount;
}

export function videoRequiresSourceVideo(
  parameters: Record<string, unknown> | undefined | null,
): boolean {
  return resolveVideoEditSourceMode(parameters) === 'source_video';
}

export function animateReferenceAcceptKind(
  parameters: Record<string, unknown> | undefined | null,
): AnimateReferenceAcceptKind {
  if (videoRequiresSourceVideo(parameters)) {
    return 'image';
  }
  return 'image';
}

export function videoSupportsVideoEdit(
  actions: Record<string, unknown> | undefined | null,
  parameters: Record<string, unknown> | undefined | null,
): boolean {
  if (!actions || typeof actions !== 'object') return false;
  if (!Object.prototype.hasOwnProperty.call(actions, 'animate') || actions.animate == null) {
    return false;
  }
  const mode = resolveVideoEditSourceMode(parameters);
  return mode === 'first_frame' || mode === 'source_video';
}

export function galleryPathIsVideo(path: string, item?: GalleryItem | null): boolean {
  if (item) return isVideoGalleryItem(item);
  if (!path.startsWith('asset:')) {
    const base = path.split(/[/\\]/).pop()?.toLowerCase() || '';
    return /\.(mp4|mov|webm|mkv|avi)$/.test(base);
  }
  return false;
}
