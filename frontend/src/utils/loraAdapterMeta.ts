/** Registry-driven compatible LoRA row from ``GET /api/settings/loras/compatible/{model}``. */

export type CompatibleLoraRow = {
  id?: string;
  name?: string;
  base_model?: string;
  source?: string;
  tags?: string[];
  compose_overrides?: Record<string, number | string>;
  hint_key?: string;
  /** `[min, max]` inference weight the trainer recommends (e.g. Turbo-trained LoRAs: 0.8–1.0). */
  recommended_lora_scale?: [number, number];
  /** False when the registry card has no installed weight. */
  ready?: boolean;
  /** Companion adapters declared by a trained LoRA. Inference appends them. */
  extra_adapters?: Array<string | { id?: string; weight?: number }>;
};

export function stackedAdapterToken(item: string | { id?: string; weight?: number } | null | undefined): string {
  if (!item) return '';
  if (typeof item === 'string') return item.trim();
  const id = String(item.id || '').trim();
  if (!id) return '';
  const weight = Number(item.weight);
  if (!Number.isFinite(weight) || weight === 1) return id;
  return `${id}@${weight}`;
}

export function loraRecommendedScale(
  lora: CompatibleLoraRow | null | undefined,
): [number, number] | null {
  const rng = lora?.recommended_lora_scale;
  if (!Array.isArray(rng) || rng.length !== 2) return null;
  const lo = Number(rng[0]);
  const hi = Number(rng[1]);
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo > hi) return null;
  return [lo, hi];
}

export function findCompatibleLora(
  loras: CompatibleLoraRow[],
  loraId: string,
): CompatibleLoraRow | undefined {
  const key = String(loraId || '').trim();
  if (!key) return undefined;
  return loras.find((row) => String(row.id) === key);
}

/** MiniMax-H3 Turbo distill — driven by composer memory-mode, not the generic LoRA picker. */
export function isH3TurboLora(lora: CompatibleLoraRow | null | undefined): boolean {
  const id = String(lora?.id || '').trim();
  if (!id) return false;
  const mid = id.includes(':') ? id.slice(0, id.indexOf(':')) : id;
  return mid === 'minimax-h3-turbo-lora';
}

export function filterOutH3TurboLoras(loras: CompatibleLoraRow[]): CompatibleLoraRow[] {
  return loras.filter((row) => !isH3TurboLora(row));
}

function scalarOverrides(
  raw: unknown,
): Record<string, number | string> | null {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null;
  const out: Record<string, number | string> = {};
  for (const [key, value] of Object.entries(raw as Record<string, unknown>)) {
    if (typeof value === 'number' || typeof value === 'string') out[key] = value;
  }
  return Object.keys(out).length > 0 ? out : null;
}

/** Card `parameters.compose_overrides`, then `versions[version]` when the id is `card:version`. */
export function registryComposeOverrides(
  registryModel: Record<string, unknown> | null | undefined,
  loraId?: string,
): Record<string, number | string> | null {
  const parameters = registryModel?.parameters;
  const card = scalarOverrides(
    parameters && typeof parameters === 'object'
      ? (parameters as Record<string, unknown>).compose_overrides
      : null,
  );
  const version = String(loraId || '').includes(':')
    ? String(loraId).slice(String(loraId).indexOf(':') + 1)
    : '';
  const versions = registryModel?.versions;
  const spec = version && versions && typeof versions === 'object'
    ? (versions as Record<string, unknown>)[version]
    : null;
  const pinned = scalarOverrides(
    spec && typeof spec === 'object'
      ? (spec as Record<string, unknown>).compose_overrides
      : null,
  );
  if (!card && !pinned) return null;
  return { ...(card || {}), ...(pinned || {}) };
}

export function applyLoraComposeOverrides(
  params: Record<string, unknown>,
  lora: CompatibleLoraRow | null | undefined,
  registryModel?: Record<string, unknown> | null,
): void {
  const fromRow = scalarOverrides(lora?.compose_overrides);
  const overrides = fromRow || registryComposeOverrides(registryModel, lora?.id);
  if (!overrides) return;
  for (const [key, value] of Object.entries(overrides)) {
    params[key] = value;
  }
}

export function loraOptionLabel(lora: CompatibleLoraRow, myLoraTag?: string): string {
  const base = String(lora.name || lora.id || '');
  if (lora.source === 'user_trained' && myLoraTag) {
    return `${base} (${myLoraTag})`;
  }
  const tags = (lora.tags || []).filter(Boolean);
  if (tags.length > 0) {
    return `${base} · ${tags.join(' · ')}`;
  }
  return base;
}

/** Fill hint, tags, and compose overrides from the public registry card when the API row omits them. */
export function withRegistryLoraMeta(
  lora: CompatibleLoraRow,
  registryModel: Record<string, unknown> | null | undefined,
): CompatibleLoraRow {
  const meta = registryModel?.metadata;
  const loraMeta = meta && typeof meta === 'object'
    ? (meta as Record<string, unknown>).lora
    : null;
  const record = loraMeta && typeof loraMeta === 'object'
    ? (loraMeta as Record<string, unknown>)
    : null;
  const hint = String(lora.hint_key || record?.hint_key || '').trim();
  const tags = lora.tags && lora.tags.length > 0
    ? lora.tags
    : Array.isArray(record?.tags)
      ? record.tags.filter((tag): tag is string => typeof tag === 'string' && tag.length > 0)
      : lora.tags;
  let recommended = lora.recommended_lora_scale;
  const rawRange = record?.recommended_lora_scale;
  if (
    !recommended
    && Array.isArray(rawRange)
    && rawRange.length === 2
    && rawRange.every((item) => typeof item === 'number')
  ) {
    recommended = [Number(rawRange[0]), Number(rawRange[1])];
  }
  return {
    ...lora,
    hint_key: hint || undefined,
    tags,
    recommended_lora_scale: recommended,
    compose_overrides: scalarOverrides(lora.compose_overrides)
      || registryComposeOverrides(registryModel, lora.id)
      || undefined,
  };
}

export function loraHintKey(lora: CompatibleLoraRow | null | undefined): string | null {
  const key = String(lora?.hint_key || '').trim();
  return key || null;
}

export function loraHasComposeOverrides(lora: CompatibleLoraRow | null | undefined): boolean {
  const overrides = lora?.compose_overrides;
  return Boolean(overrides && typeof overrides === 'object' && Object.keys(overrides).length > 0);
}
