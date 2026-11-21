import type { CanvasOverlayKind } from '@/types';
import type { CanvasMedia } from '@/composables/useCanvasStore';

export const CANVAS_OVERLAY_KINDS = [
  'reference',
  'control',
  'start_frame',
  'tail_frame',
  'video_source',
  'cover_source',
] as const satisfies readonly CanvasOverlayKind[];

export const OVERLAY_KINDS_BY_MEDIA: Record<CanvasMedia, CanvasOverlayKind[]> = {
  image: ['control'],
  video: ['start_frame', 'tail_frame', 'video_source'],
  // Cover source binds in the composer only — no canvas overlay layer for audio.
  audio: [],
};

export const OVERLAY_LABEL_KEYS: Record<CanvasOverlayKind, string> = {
  reference: 'canvas.overlayReference',
  control: 'canvas.overlayControl',
  start_frame: 'canvas.overlayStartFrame',
  tail_frame: 'canvas.overlayTailFrame',
  video_source: 'canvas.overlayVideoSource',
  cover_source: 'canvas.overlayCoverSource',
};

export const OVERLAY_BADGES: Record<CanvasOverlayKind, string> = {
  reference: 'REF',
  control: 'CTRL',
  start_frame: 'START',
  tail_frame: 'TAIL',
  video_source: 'VID',
  cover_source: 'COVER',
};

export const OVERLAY_DEFAULT_OPACITY: Record<CanvasOverlayKind, number> = {
  reference: 0.42,
  control: 0.55,
  start_frame: 0.45,
  tail_frame: 0.45,
  video_source: 0.5,
  cover_source: 0.55,
};
