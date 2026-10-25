/**
 * models_registry.json parameters — inference and utilities aligned with
 * backend/core/registry_format.typed_parameters.
 */

export type InferredParamType = 'int' | 'float' | 'enum' | 'bool' | 'object' | 'skip';

export type RawParamSpec = Record<string, unknown>;

export type NormalizedParamSpec = RawParamSpec & { type: InferredParamType };

export function inferType(spec: unknown, key?: string): InferredParamType {
  if (!spec || typeof spec !== 'object') return 'skip';
  if (key === 'resolution_presets' || key === 'ref2va') return 'skip';
  const s = spec as RawParamSpec;
  if (typeof s.type === 'string') return s.type as InferredParamType;
  if (typeof s.default === 'boolean' || (key && String(key).endsWith('_support'))) {
    return 'bool';
  }
  if (Array.isArray(s.options)) return 'enum';
  if (typeof s.min === 'number' && typeof s.max === 'number') {
    const d = s.default;
    return Number.isInteger(d) && typeof d !== 'boolean' ? 'int' : 'float';
  }
  return 'object';
}

export function normalizeParamsDef(parameters: Record<string, unknown> | undefined | null): Record<string, NormalizedParamSpec> {
  const out: Record<string, NormalizedParamSpec> = {};
  for (const [key, spec] of Object.entries(parameters || {})) {
    const t = inferType(spec, key);
    if (t === 'skip') continue;
    out[key] = { ...(spec as RawParamSpec), type: t };
  }
  return out;
}

export function isCapabilityOnly(key: string, spec: NormalizedParamSpec): boolean {
  return spec.type === 'bool' && String(key).endsWith('_support');
}

export function isRenderableScalar(key: string, spec: NormalizedParamSpec): boolean {
  if (isCapabilityOnly(key, spec)) return false;
  if (spec.type === 'object') return false;
  return spec.type === 'int' || spec.type === 'float' || spec.type === 'enum';
}

const PREFERRED_ORDER = [
  'steps',
  'guidance',
  'scheduler',
  'scheduler_sigma_schedule',
  'enable_cfg_renorm',
  'cfg_renorm_min',
  'strength',
  'controlnet_strength',
  'redux_strength',
];

export function sortParamKeys(keys: string[]): string[] {
  return [...keys].sort((a, b) => {
    const ia = PREFERRED_ORDER.indexOf(a);
    const ib = PREFERRED_ORDER.indexOf(b);
    if (ia === -1 && ib === -1) return a.localeCompare(b);
    if (ia === -1) return 1;
    if (ib === -1) return -1;
    return ia - ib;
  });
}

export function resolutionPair(
  normalized: Record<string, NormalizedParamSpec>
): { width: NormalizedParamSpec; height: NormalizedParamSpec } | null {
  const w = normalized.width;
  const h = normalized.height;
  if (w && h && w.type === 'enum' && h.type === 'enum') {
    return { width: w, height: h };
  }
  return null;
}

export type ResolutionSizeOption = {
  label: string;
  value: string;
  pixelLabel?: string;
  /** Registry ``resolution_presets.default`` or width×height default. */
  isDefault?: boolean;
};

const CURATED_RESOLUTION_PAIRS: Array<[number, number]> = [
  [512, 512],
  [768, 768],
  [1024, 1024],
  [1280, 1280],
  [1536, 1536],
  [1920, 1920],
  [512, 768],
  [768, 512],
  [768, 1024],
  [1024, 768],
  [768, 1280],
  [1280, 768],
  [1024, 1280],
  [1280, 1024],
  [1024, 1536],
  [1536, 1024],
  [1280, 1536],
  [1536, 1280],
  [1920, 1280],
  [1280, 1920],
  [1920, 1536],
  [1536, 1920],
  [1920, 1024],
  [1024, 1920],
];

function gcd(a: number, b: number): number {
  let x = Math.abs(a);
  let y = Math.abs(b);
  while (y) {
    [x, y] = [y, x % y];
  }
  return x || 1;
}

function aspectRatioLabel(width: number, height: number): string {
  if (width === height) return '1:1';
  const g = gcd(width, height);
  return `${width / g}:${height / g}`;
}

export function parseSizeValue(value: string): { width: number; height: number } | null {
  const [w, h] = value.split('x').map(Number);
  if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) return null;
  return { width: w, height: h };
}

export function formatSizeValue(width: number, height: number): string {
  return `${width}x${height}`;
}

function resolutionPresetStrings(raw: Record<string, unknown>): string[] | null {
  const presets = raw.resolution_presets;
  if (!presets || typeof presets !== 'object') return null;
  const spec = presets as RawParamSpec;
  if (!Array.isArray(spec.options)) return null;
  return spec.options.map(String);
}

function resolutionPresetDefault(raw: Record<string, unknown>): string | null {
  const presets = raw.resolution_presets;
  if (!presets || typeof presets !== 'object') return null;
  const def = (presets as RawParamSpec).default;
  return typeof def === 'string' ? def : null;
}

function toResolutionSizeOption(value: string, isDefault = false): ResolutionSizeOption {
  const parsed = parseSizeValue(value);
  if (!parsed) return { label: value, value, isDefault };
  const pixelLabel = `${parsed.width}×${parsed.height}`;
  const label =
    parsed.width === parsed.height ? '1:1' : aspectRatioLabel(parsed.width, parsed.height);
  return { label, value, pixelLabel, isDefault };
}

/** Single-line label for Composer ``DqOption`` (avoids duplicate value + slot text). */
export function formatResolutionOptionLabel(opt: ResolutionSizeOption): string {
  const px = opt.pixelLabel || opt.value.replace('x', '×');
  if (opt.label === '1:1') return px;
  return `${px} (${opt.label})`;
}

function orderResolutionOptions(options: ResolutionSizeOption[], preferred?: string | null): ResolutionSizeOption[] {
  if (!preferred) return options;
  const idx = options.findIndex((o) => o.value === preferred);
  if (idx <= 0) return options;
  return [options[idx]!, ...options.slice(0, idx), ...options.slice(idx + 1)];
}

/** Build Composer resolution dropdown from registry ``parameters`` (presets or width/height enums). */
export function buildResolutionSizeOptions(
  parameters: Record<string, unknown> | undefined | null,
): ResolutionSizeOption[] {
  const raw = parameters || {};
  const presetValues = resolutionPresetStrings(raw);
  if (presetValues?.length) {
    const unique = [...new Set(presetValues)];
    const presetDefault = resolutionPresetDefault(raw);
    const options = unique.map((value) =>
      toResolutionSizeOption(value, value === presetDefault),
    );
    return orderResolutionOptions(options, presetDefault);
  }

  const normalized = normalizeParamsDef(raw);
  const pair = resolutionPair(normalized);
  if (!pair) return [];

  const wOpts = (pair.width.options as number[] | undefined) ?? [Number(pair.width.default) || 1024];
  const hOpts = (pair.height.options as number[] | undefined) ?? [Number(pair.height.default) || 1024];
  const wSet = new Set(wOpts);
  const hSet = new Set(hOpts);
  const wDef = Number(pair.width.default) || wOpts[0] || 1024;
  const hDef = Number(pair.height.default) || hOpts[0] || 1024;

  const seen = new Map<string, [number, number]>();
  const addPair = (w: number, h: number) => {
    if (!wSet.has(w) || !hSet.has(h)) return;
    seen.set(formatSizeValue(w, h), [w, h]);
  };

  addPair(wDef, hDef);
  for (const [w, h] of CURATED_RESOLUTION_PAIRS) addPair(w, h);

  if (seen.size < 4) {
    for (const w of wOpts) {
      for (const h of hOpts) addPair(w, h);
    }
  }

  const defaultValue = formatSizeValue(wDef, hDef);
  const entries = [...seen.entries()].sort(([a], [b]) => {
    if (a === defaultValue) return -1;
    if (b === defaultValue) return 1;
    const pa = parseSizeValue(a);
    const pb = parseSizeValue(b);
    if (!pa || !pb) return a.localeCompare(b);
    return pa.width * pa.height - pb.width * pb.height;
  });

  const options = entries.map(([value, [w, h]]) => ({
    label: w === h ? '1:1' : aspectRatioLabel(w, h),
    value,
    pixelLabel: `${w}×${h}`,
    isDefault: value === defaultValue,
  }));
  return orderResolutionOptions(options, defaultValue);
}

/** Pick saved size if valid, else registry default preset / width×height. */
export function pickResolutionForModel(
  parameters: Record<string, unknown> | undefined | null,
  savedSize: string | null | undefined,
): string | null {
  const options = buildResolutionSizeOptions(parameters);
  if (!options.length) return null;
  if (savedSize && options.some((o) => o.value === savedSize)) return savedSize;

  const presetDefault = resolutionPresetDefault(parameters || {});
  if (presetDefault && options.some((o) => o.value === presetDefault)) return presetDefault;

  const normalized = normalizeParamsDef(parameters || {});
  const pair = resolutionPair(normalized);
  if (pair) {
    const def = formatSizeValue(Number(pair.width.default), Number(pair.height.default));
    if (options.some((o) => o.value === def)) return def;
  }
  return options[0]?.value ?? null;
}

/** Match animate start-frame aspect ratio to the closest registry preset (by AR, then scale). */
export function pickClosestResolutionPreset(
  options: ResolutionSizeOption[],
  sourceWidth: number,
  sourceHeight: number,
): string | null {
  if (!options.length || sourceWidth <= 0 || sourceHeight <= 0) return null;
  const targetAr = sourceWidth / sourceHeight;
  let bestValue: string | null = null;
  let bestScore = Infinity;
  for (const opt of options) {
    const parsed = parseSizeValue(opt.value);
    if (!parsed) continue;
    const ar = parsed.width / parsed.height;
    const arDiff = Math.abs(Math.log(ar) - Math.log(targetAr));
    const scale = Math.max(
      parsed.width / sourceWidth,
      parsed.height / sourceHeight,
      sourceWidth / parsed.width,
      sourceHeight / parsed.height,
    );
    const score = arDiff * 2 + Math.log(Math.max(scale, 1));
    if (score < bestScore) {
      bestScore = score;
      bestValue = opt.value;
    }
  }
  return bestValue;
}

export function loadImageNaturalSize(url: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      resolve({ width: img.naturalWidth, height: img.naturalHeight });
    };
    img.onerror = () => reject(new Error('failed to load image dimensions'));
    img.src = url;
  });
}

export function scalarKeysForForm(normalized: Record<string, NormalizedParamSpec>): string[] {
  const pair = resolutionPair(normalized);
  const skip = new Set<string>();
  if (pair) {
    skip.add('width');
    skip.add('height');
  }
  const keys = Object.keys(normalized).filter((k) => {
    if (skip.has(k)) return false;
    return isRenderableScalar(k, normalized[k]!);
  });
  return sortParamKeys(keys);
}

export function applyDefaults(parameters: Record<string, unknown> | undefined | null, target: Record<string, unknown>): void {
  const n = normalizeParamsDef(parameters || {});
  for (const [key, spec] of Object.entries(n)) {
    if (isCapabilityOnly(key, spec) || spec.type === 'object') continue;
    if ('default' in spec) {
      target[key] = spec.default;
    }
  }
  if (typeof target.seed !== 'undefined') target.seed = '';
  if (typeof target.lora !== 'undefined') target.lora = '';
  if (typeof target.controlnet !== 'undefined') {
    target.controlnet = '';
    target.controlnet_strength =
      n.controlnet_strength && 'default' in n.controlnet_strength
        ? (n.controlnet_strength as NormalizedParamSpec).default
        : 0.8;
  }
  const ls = n.lora_scale;
  if (typeof target.lora_scale !== 'undefined') {
    target.lora_scale = ls && 'default' in ls ? (ls as NormalizedParamSpec).default : 0.8;
  }
}

export function strengthDefaultFromRegistry(
  parameters: Record<string, unknown> | undefined | null,
): number {
  const spec = normalizeParamsDef(parameters || {}).strength;
  if (spec && typeof spec.default === 'number') return spec.default;
  return 0.4;
}

/** 图生图是否使用 ``source_fidelity``（标准 img2img 混合）；VL/concat 编辑模型忽略强度。 */
export function img2imgUsesStrength(parameters: Record<string, unknown> | undefined | null): boolean {
  const raw = parameters || {};
  if (raw.edit_use_vl_vision || raw.edit_conditioning_concat) return false;
  return 'strength' in normalizeParamsDef(raw);
}

/**
 * How many reference images the composer may hold, including picture 1.
 * Multi-image edit is ``edit_plus_multi_image`` (Qwen / FireRed) or
 * ``reference_images_support`` (HiDream). Otherwise only the primary image.
 */
export function referenceImageSlotCap(
  parameters: Record<string, unknown> | undefined | null,
): number {
  const raw = parameters || {};
  const multi = Boolean(raw.edit_plus_multi_image || raw.reference_images_support);
  if (!multi) return 1;
  const max = Number(raw.edit_max_reference_images ?? raw.max_reference_images ?? 1);
  if (!Number.isFinite(max)) return 1;
  return Math.max(1, Math.floor(max));
}

/** Extra slots after picture 1. */
export function extraReferenceSlotCap(
  parameters: Record<string, unknown> | undefined | null,
): number {
  return Math.max(0, referenceImageSlotCap(parameters) - 1);
}

/** UI 去噪强度 (strength) → edits API ``source_fidelity``（高 strength = 更多改动 = 低 fidelity）。 */
export function strengthToSourceFidelity(strength: unknown, fallback = 0.4): number {
  const s = Number(strength);
  const v = Number.isFinite(s) ? s : fallback;
  return Math.min(0.95, Math.max(0.05, 1 - v));
}

/** Enum inference options declared in models_registry (Composer advanced panel). */
export const COMPOSER_INFERENCE_ENUM_KEYS = ['teacache_mode', 'lemica_mode'] as const;

/** Video composer generic enum loop (TeaCache). H3 Veda is a dedicated mutex group. */
export const VIDEO_INFERENCE_ENUM_KEYS = ['teacache_mode'] as const;

/** Still submitted on create/edit when the model declares them. */
export const VIDEO_H3_INFERENCE_ENUM_KEYS = ['h3_veda'] as const;

export function hasRegistryParam(
  parameters: Record<string, unknown> | undefined | null,
  key: string,
): boolean {
  return Object.prototype.hasOwnProperty.call(normalizeParamsDef(parameters || {}), key);
}

export function enumParamOptions(spec: NormalizedParamSpec | undefined): string[] {
  if (!spec || !Array.isArray(spec.options)) return [];
  return spec.options.map(String);
}

/** Attach registry enum fields to create/edit API bodies (including explicit ``none``). */
export function appendActiveEnumFields(
  body: Record<string, unknown>,
  params: Record<string, unknown>,
  keys: readonly string[],
  parameters?: Record<string, unknown> | undefined | null,
): void {
  const schema = normalizeParamsDef(parameters || {});
  for (const key of keys) {
    const spec = schema[key];
    if (!spec) continue;
    const raw = params[key] ?? spec.default;
    if (raw === undefined || raw === null || raw === '') continue;
    const val = String(raw).trim().toLowerCase();
    body[key] = val === 'off' ? 'none' : val;
  }
}

/** Latent refine upscaler block when registry declares ``latent_refine_scale``. */
export function appendLatentRefineField(
  body: Record<string, unknown>,
  params: Record<string, unknown>,
  parameters?: Record<string, unknown> | undefined | null,
): void {
  if (!hasRegistryParam(parameters, 'latent_refine_scale')) return;
  const schema = normalizeParamsDef(parameters || {});
  const scaleSpec = schema.latent_refine_scale;
  const fallbackScale = typeof scaleSpec?.default === 'number' ? scaleSpec.default : 1.0;
  const refineScale = Number(params.latent_refine_scale ?? fallbackScale);
  if (!Number.isFinite(refineScale) || refineScale <= 1.0) return;
  const denoiseSpec = schema.latent_refine_denoise;
  const fallbackDenoise = typeof denoiseSpec?.default === 'number' ? denoiseSpec.default : 0.35;
  body.latent_refine = {
    scale: refineScale,
    denoise_strength: Number(params.latent_refine_denoise ?? fallbackDenoise) || fallbackDenoise,
    hires_steps: 0,
    interpolation: 'linear',
  };
}

/** Legacy i18n keys for enum option labels under ``create.*``. */
export function enumOptionI18nKey(paramKey: string, value: string): string | null {
  const map: Record<string, Record<string, string>> = {
    teacache_mode: {
      none: 'teacacheOff',
      auto: 'teacacheAuto',
      quality: 'teacacheQuality',
      medium: 'teacacheMedium',
      fast: 'teacacheFast',
    },
    lemica_mode: {
      none: 'lemicaOff',
      slow: 'lemicaSlow',
      medium: 'lemicaMedium',
      fast: 'lemicaFast',
    },
    h3_veda: {
      off: 'h3VedaOff',
      on: 'h3VedaOn',
    },
  };
  return map[paramKey]?.[value] ?? null;
}

export function composerParamLabelI18nKey(paramKey: string): string | null {
  const map: Record<string, string> = {
    teacache_mode: 'teacacheMode',
    lemica_mode: 'lemicaMode',
    latent_refine_scale: 'latentRefineScale',
    latent_refine_denoise: 'latentRefineDenoise',
  };
  return map[paramKey] ?? null;
}

export function composerParamHintI18nKey(paramKey: string): string | null {
  const map: Record<string, string> = {
    teacache_mode: 'teacacheHint',
    lemica_mode: 'lemicaHint',
  };
  return map[paramKey] ?? null;
}

export function hasDeviation(
  parameters: Record<string, unknown> | undefined | null,
  target: Record<string, unknown>,
  opts?: { ignoreKeys?: string[] }
): boolean {
  const ignore = new Set(opts?.ignoreKeys || ['strength', 'controlnet_strength']);
  const n = normalizeParamsDef(parameters || {});
  for (const [key, spec] of Object.entries(n)) {
    if (ignore.has(key)) continue;
    if (!isRenderableScalar(key, spec)) continue;
    const def = spec.default;
    if (def !== undefined && target[key] !== def) return true;
  }
  const pair = resolutionPair(n);
  if (pair) {
    if (target.width !== pair.width.default || target.height !== pair.height.default) {
      return true;
    }
  }
  if (target.lora) return true;
  if (target.seed != null && String(target.seed).trim() !== '') return true;
  return false;
}
