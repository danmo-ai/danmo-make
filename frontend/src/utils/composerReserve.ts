export const COMPOSER_RESERVE_CSS_DRAWER_MODE = '72px';
export const COMPOSER_SCRIM_CSS_DRAWER_MODE = '0px';

/** CSS length for floating canvas controls above the composer bar. */
export const COMPOSER_RESERVE_CSS_EXPANDED = 'min(200px, 36vh)';
export const COMPOSER_RESERVE_CSS_COLLAPSED = '56px';
export const COMPOSER_SCRIM_CSS_EXPANDED = 'min(220px, 38vh)';
export const COMPOSER_SCRIM_CSS_COLLAPSED = '80px';

export function composerReservePx(
  viewportHeight: number,
  collapsed: boolean,
  drawerMode = false,
): number {
  if (drawerMode) return 72 + 16;
  if (collapsed) return 56 + 24;
  return Math.min(viewportHeight * 0.38, 220) + 88;
}

export function readComposerReservePx(el: HTMLElement | null, viewportHeight: number): number {
  if (!el) return composerReservePx(viewportHeight, false);
  const raw = getComputedStyle(el).getPropertyValue('--dq-composer-reserve-px').trim();
  const n = parseFloat(raw);
  return Number.isFinite(n) && n > 0 ? n : composerReservePx(viewportHeight, false);
}
