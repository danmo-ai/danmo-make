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
};

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

export function applyLoraComposeOverrides(
  params: Record<string, unknown>,
  lora: CompatibleLoraRow | null | undefined,
): void {
  const overrides = lora?.compose_overrides;
  if (!overrides || typeof overrides !== 'object') return;
  for (const [key, value] of Object.entries(overrides)) {
    if (value === undefined || value === null) continue;
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

export function loraHintKey(lora: CompatibleLoraRow | null | undefined): string | null {
  const key = String(lora?.hint_key || '').trim();
  return key || null;
}

export function loraHasComposeOverrides(lora: CompatibleLoraRow | null | undefined): boolean {
  const overrides = lora?.compose_overrides;
  return Boolean(overrides && typeof overrides === 'object' && Object.keys(overrides).length > 0);
}
