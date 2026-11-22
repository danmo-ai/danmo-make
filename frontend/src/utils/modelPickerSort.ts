/** Create-page model picker: recommended models first, then display name. */
export function compareModelPickerRows(
  a: { recommended?: unknown; name?: unknown },
  b: { recommended?: unknown; name?: unknown },
  locale = 'zh',
): number {
  const ar = a.recommended ? 1 : 0;
  const br = b.recommended ? 1 : 0;
  if (ar !== br) return br - ar;
  const an = String(a.name || '');
  const bn = String(b.name || '');
  try {
    return an.localeCompare(bn, locale);
  } catch {
    return an < bn ? -1 : an > bn ? 1 : 0;
  }
}
