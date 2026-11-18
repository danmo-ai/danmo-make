const SEED_RANGE = 2_147_483_647;

export function randomSeed(): number {
  return Math.floor(Math.random() * SEED_RANGE);
}

/** A seed the user typed. An empty field is not a seed. */
export function enteredSeed(raw: unknown): number | null {
  if (raw == null) return null;
  const text = String(raw).trim();
  if (!text) return null;
  const n = Number.parseInt(text, 10);
  return Number.isFinite(n) ? n : null;
}

/** The typed seed. An empty field gets a fresh random seed; a non-numeric entry stays unset. */
export function seedForSubmit(raw: unknown): number | null {
  if (raw == null || String(raw).trim() === '') return randomSeed();
  return enteredSeed(raw);
}
