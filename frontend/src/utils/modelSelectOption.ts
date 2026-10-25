/** Option row for the shared create-page model select. */
export type ModelSelectOption = {
  label: string;
  value: string;
  disabled?: boolean;
  commercialUseAllowed?: boolean;
  description?: string;
};

type ModelSelectSource = {
  name?: unknown;
  modelKey?: unknown;
  versionKey?: unknown;
  ready?: unknown;
  commercialUseAllowed?: unknown;
  description?: unknown;
};

/** Name, version value, ready state, commercial badge, and description tip. */
export function modelSelectOption(
  row: ModelSelectSource,
  opts?: { unreadySuffix?: string },
): ModelSelectOption {
  const ready = Boolean(row.ready);
  const suffix = !ready && opts?.unreadySuffix ? ` (${opts.unreadySuffix})` : '';
  const description = String(row.description || '').trim();
  return {
    label: `${String(row.name || '')}${suffix}`,
    value: `${String(row.modelKey || '')}|${String(row.versionKey || '')}`,
    disabled: !ready,
    commercialUseAllowed: row.commercialUseAllowed === true,
    ...(description ? { description } : {}),
  };
}
