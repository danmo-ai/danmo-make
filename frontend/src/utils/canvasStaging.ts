import type { CanvasItemState, CanvasStagingState, GalleryItem } from '@/types';

/** Long-edge cap so the landing box stays a guide, not a near-fullscreen frame. */
export const STAGING_MAX = 420;
export const STAGING_MIN = 240;
const STAGING_EDGE_MIN = 160;

/** Clamp persisted / resized staging so legacy 768-tall boxes shrink on load. */
export function clampStagingSize(width: number, height: number): {
  width: number;
  height: number;
} {
  let w = Math.max(STAGING_EDGE_MIN, Number(width) || STAGING_MIN);
  let h = Math.max(STAGING_EDGE_MIN, Number(height) || STAGING_MIN);
  const long = Math.max(w, h);
  if (long > STAGING_MAX) {
    const s = STAGING_MAX / long;
    w = Math.max(STAGING_EDGE_MIN, Math.round(w * s));
    h = Math.max(STAGING_EDGE_MIN, Math.round(h * s));
  }
  return { width: w, height: h };
}

/** Staging box size preserving source aspect ratio. */
export function computeStagingDimensions(gi?: GalleryItem | null): {
  width: number;
  height: number;
} {
  const refW = Math.max(64, gi?.width || 512);
  const refH = Math.max(64, gi?.height || 512);
  const aspect = refW / refH;

  if (aspect >= 1) {
    const width = Math.min(STAGING_MAX, Math.max(STAGING_MIN, refW));
    const height = Math.round(width / aspect);
    return clampStagingSize(width, Math.max(STAGING_EDGE_MIN, height));
  }

  const height = Math.min(STAGING_MAX, Math.max(STAGING_MIN, refH));
  const width = Math.round(height * aspect);
  return clampStagingSize(Math.max(STAGING_EDGE_MIN, width), height);
}

export function layoutInStaging(
  index: number,
  item: GalleryItem,
  staging: CanvasStagingState
): { x: number; y: number; scale: number } {
  const isAudio =
    item.metadata?.asset_kind === 'audio' || item.mime_type?.startsWith('audio/');
  const baseW = isAudio ? 280 : item.width || 512;
  const baseH = isAudio ? 120 : item.height || 512;

  const pad = 20;
  const gap = 12;
  const innerW = Math.max(80, staging.width - pad * 2);
  const cellW = Math.min(240, innerW);
  const cols = Math.max(1, Math.floor((innerW + gap) / (cellW + gap)));
  const col = index % cols;
  const row = Math.floor(index / cols);

  const targetW = Math.min(cellW, innerW / cols - gap);
  const scale = Math.min(1, targetW / Math.max(baseW, baseH, 1));
  const displayW = baseW * scale;
  const displayH = baseH * scale;

  const rowStride = displayH + gap;
  const x =
    staging.x +
    pad +
    col * (innerW / cols) +
    (innerW / cols - displayW) / 2;
  const y = staging.y + pad + row * rowStride;

  return { x, y, scale: Math.max(0.1, Math.min(3, scale)) };
}

/** Parent asset path (`asset:…`) from gallery lineage metadata, if any. */
export function parentAssetPath(item: GalleryItem | null | undefined): string | null {
  const parentId = String(item?.metadata?.parent_asset_id || '').trim();
  if (!parentId) return null;
  return `asset:${parentId}`;
}

/** Place a child node to the right of its parent when the parent is already on canvas. */
export function lineageSlotBesideParent(
  item: GalleryItem,
  galleryItems: GalleryItem[],
  canvasItems: Record<string, CanvasItemState>,
  extraSiblingOffset = 0
): { x: number; y: number } | null {
  const parentPath = parentAssetPath(item);
  if (!parentPath) return null;
  const parentState = canvasItems[parentPath];
  const parentGi = galleryItems.find((g) => g.path === parentPath);
  if (!parentState || !parentGi) return null;

  const pw = (parentGi.width || 512) * parentState.scale;
  const ph = (parentGi.height || 512) * parentState.scale;
  const gap = 48;

  let siblingOffset = extraSiblingOffset;
  for (const path of Object.keys(canvasItems)) {
    if (path === item.path) continue;
    const gi = galleryItems.find((g) => g.path === path);
    if (gi && parentAssetPath(gi) === parentPath) siblingOffset += 1;
  }

  const rowStride = Math.max(80, Math.min(ph, item.height || 512) * 0.35);
  return {
    x: parentState.x + pw + gap,
    y: parentState.y + siblingOffset * rowStride,
  };
}

export function countItemsInStaging(
  items: Record<string, { x: number; y: number }>,
  staging: CanvasStagingState
): number {
  const x0 = staging.x;
  const y0 = staging.y;
  const x1 = staging.x + staging.width;
  const y1 = staging.y + staging.height;
  return Object.values(items).filter(
    (it) => it.x >= x0 - 8 && it.y >= y0 - 8 && it.x <= x1 && it.y <= y1
  ).length;
}
