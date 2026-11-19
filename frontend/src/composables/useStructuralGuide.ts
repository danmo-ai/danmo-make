import { ref, type Ref } from 'vue';
import { api } from '@/utils/api';
import { $mn } from '@/utils/i18n';
import {
  appendActiveEnumFields,
  appendLatentRefineField,
  COMPOSER_INFERENCE_ENUM_KEYS,
} from '@/utils/registryParamSchema';

export type StructuralGuideType =
  | 'canny'
  | 'depth'
  | 'pose'
  | 'hed'
  | 'mlsd'
  | 'scribble'
  | 'gray'
  | 'auto'
  | 'redux'
  | 'fill';
export type ControlNetScope = 'create' | 'retouch' | 'extend';
export type ControlBinding = 'structural' | 'standalone_edit';

/** Mirrors backend ``CONTROLNET_DECLARED_BACKENDS`` until CUDA batch lands. */
export const CONTROLNET_DECLARED_BACKENDS = ['mlx'] as const;

/** Prefer explicit API flag; infer MLX on Apple Silicon while system info is loading. */
export function resolveControlNetHostRuntimeAvailable(
  hostRuntimeAvailable?: boolean | null,
  compatibleControlNets?: Record<string, unknown>[],
  systemInfo?: { platform?: string; architecture?: string; dependencies?: Record<string, string> } | null,
): boolean {
  if (typeof hostRuntimeAvailable === 'boolean') return hostRuntimeAvailable;
  if (
    systemInfo?.platform === 'Darwin' &&
    systemInfo?.architecture === 'arm64'
  ) {
    const mlx = systemInfo.dependencies?.mlx;
    if (mlx && mlx !== 'not installed') return true;
  }
  const nets = compatibleControlNets;
  if (!nets?.length) return false;
  return nets.some((n) => n.runtime_available === true);
}

/** Host-level ControlNet runtime (prefer system info; fall back to compatible-net rows). */
export function isControlNetHostRuntimeAvailable(
  hostRuntimeAvailable?: boolean | null,
  compatibleControlNets?: Record<string, unknown>[],
  systemInfo?: { platform?: string; architecture?: string; dependencies?: Record<string, string> } | null,
): boolean {
  return resolveControlNetHostRuntimeAvailable(
    hostRuntimeAvailable,
    compatibleControlNets,
    systemInfo,
  );
}

export type ControlImageRef = {
  path: string;
  previewUrl: string;
  assetId?: string;
} | null;

function findControlNetRow(
  key: string,
  rows?: Record<string, unknown>[],
): Record<string, unknown> | undefined {
  if (!key || !rows?.length) return undefined;
  return rows.find((n) => String(n.key) === key);
}

/** Registry / compatible-row binding (`structural` | `standalone_edit`). */
export function controlNetBinding(
  row: Record<string, unknown> | undefined | null,
): ControlBinding | '' {
  const b = String(row?.binding || '').trim();
  if (b === 'structural' || b === 'standalone_edit') return b;
  return '';
}

export function controlNetGuideType(
  row: Record<string, unknown> | undefined | null,
  fallback: StructuralGuideType = 'canny',
): StructuralGuideType {
  const t = String(row?.guide_type || '').trim();
  if (
    t === 'canny' ||
    t === 'depth' ||
    t === 'pose' ||
    t === 'hed' ||
    t === 'mlsd' ||
    t === 'scribble' ||
    t === 'gray' ||
    t === 'auto' ||
    t === 'redux' ||
    t === 'fill'
  ) {
    return t;
  }
  return fallback;
}

export function controlNetSupportsInpaint(row: Record<string, unknown> | undefined | null): boolean {
  return row?.supports_inpaint === true;
}

export function controlNetAllowsWithReference(
  row: Record<string, unknown> | undefined | null,
): boolean {
  return row?.allows_with_reference === true;
}

export function isStandaloneEditControlNet(
  keyOrRow: string | Record<string, unknown> | undefined | null,
  rows?: Record<string, unknown>[],
  registry?: Record<string, Record<string, unknown>>,
): boolean {
  if (keyOrRow && typeof keyOrRow === 'object') {
    return controlNetBinding(keyOrRow) === 'standalone_edit';
  }
  const key = String(keyOrRow || '');
  const fromRows = findControlNetRow(key, rows);
  if (fromRows) return controlNetBinding(fromRows) === 'standalone_edit';
  if (registry?.[key]) return controlNetBinding(registry[key]) === 'standalone_edit';
  return false;
}

export function isStructuralRetouchControlNet(
  row: Record<string, unknown> | undefined | null,
): boolean {
  return controlNetBinding(row) === 'structural' && controlNetSupportsInpaint(row);
}

/** @deprecated Prefer controlNetGuideType from registry/compatible rows. */
export function inferStructuralGuideType(
  controlnetKey: string,
  compatibleControlNets?: Record<string, unknown>[],
  registry?: Record<string, Record<string, unknown>>,
): StructuralGuideType {
  const row =
    findControlNetRow(controlnetKey, compatibleControlNets) ||
    (registry?.[controlnetKey] as Record<string, unknown> | undefined);
  return controlNetGuideType(row, 'canny');
}

export function controlNetDisplayName(n: Record<string, unknown>): string {
  return $mn(
    n as { name?: string | { zh?: string; en?: string }; name_en?: string },
    String(n.key || ''),
  );
}

export function controlNetReady(n: Record<string, unknown>): boolean {
  if (n.ready === true) return true;
  const vr = n.versions_ready as Record<string, boolean> | undefined;
  if (vr && Object.values(vr).some(Boolean)) return true;
  return false;
}

/** Prefer `key:readyVersion` so safetensors ControlNets resolve (e.g. `:8steps`). */
export function resolveControlNetModelId(
  controlnetKey: string,
  opts?: {
    compatibleControlNets?: Record<string, unknown>[];
    registry?: Record<string, Record<string, unknown>>;
    versionKey?: string;
  },
): string {
  const key = String(controlnetKey || '').trim();
  if (!key) return '';
  if (key.includes(':')) return key;
  if (opts?.versionKey) return `${key}:${opts.versionKey}`;
  const row =
    findControlNetRow(key, opts?.compatibleControlNets) ||
    (opts?.registry?.[key] as Record<string, unknown> | undefined);
  const vr = row?.versions_ready as Record<string, boolean> | undefined;
  if (vr) {
    const ready = Object.entries(vr).find(([, ok]) => ok);
    if (ready?.[0]) return `${key}:${ready[0]}`;
  }
  const versions = (row?.versions || opts?.registry?.[key]?.versions) as
    | Record<string, { default?: boolean }>
    | undefined;
  if (versions) {
    const def = Object.entries(versions).find(([, cfg]) => cfg?.default === true);
    if (def?.[0]) return `${key}:${def[0]}`;
    const first = Object.keys(versions)[0];
    if (first) return `${key}:${first}`;
  }
  return key;
}

export function buildStructuralGuidePayload(
  controlnetKey: string,
  assetId: string,
  strength: number,
  inpaint?: { sourceAssetId?: string; maskAssetId?: string },
  opts?: {
    compatibleControlNets?: Record<string, unknown>[];
    registry?: Record<string, Record<string, unknown>>;
    versionKey?: string;
  },
): Record<string, unknown> {
  const bare = String(controlnetKey || '').split(':')[0] || '';
  const row =
    findControlNetRow(bare, opts?.compatibleControlNets) ||
    (opts?.registry?.[bare] as Record<string, unknown> | undefined);
  const payload: Record<string, unknown> = {
    asset_id: assetId,
    model_id: resolveControlNetModelId(controlnetKey, opts),
    type: controlNetGuideType(row, 'canny'),
    weight: Number(strength) || 0.8,
  };
  const src = String(inpaint?.sourceAssetId || '').trim();
  const msk = String(inpaint?.maskAssetId || '').trim();
  if (src && msk) {
    payload.inpaint_source_asset_id = src;
    payload.inpaint_mask_asset_id = msk;
  }
  return payload;
}

/** Attach structural guide + registry inference options to create/rewrite bodies. */
export function appendZImageEnhancementFields(
  body: Record<string, unknown>,
  opts: {
    parameters?: Record<string, unknown> | null;
    params?: Record<string, unknown>;
    controlnet?: string;
    controlAssetId?: string | null;
    controlnetStrength?: number;
    inpaintSourceId?: string;
    inpaintMaskId?: string;
    compatibleControlNets?: Record<string, unknown>[];
    registry?: Record<string, Record<string, unknown>>;
  },
): { ok: true } | { ok: false; code: 'inpaint_pair_required' } {
  const controlnet = String(opts.controlnet || '');
  const controlAssetId = String(opts.controlAssetId || '').trim();
  if (controlAssetId && controlnet) {
    const inpSrc = String(opts.inpaintSourceId || '').trim();
    const inpMsk = String(opts.inpaintMaskId || '').trim();
    if (Boolean(inpSrc) !== Boolean(inpMsk)) {
      return { ok: false, code: 'inpaint_pair_required' };
    }
    body.structural_guide = buildStructuralGuidePayload(
      controlnet,
      controlAssetId,
      Number(opts.controlnetStrength) || 0.8,
      inpSrc && inpMsk ? { sourceAssetId: inpSrc, maskAssetId: inpMsk } : undefined,
      {
        compatibleControlNets: opts.compatibleControlNets,
        registry: opts.registry,
      },
    );
  }
  const composerParams = opts.params || {};
  appendActiveEnumFields(body, composerParams, COMPOSER_INFERENCE_ENUM_KEYS, opts.parameters);
  appendLatentRefineField(body, composerParams, opts.parameters);
  return { ok: true };
}

/** Attach registry inference enums (TeaCache / LeMiCa) + latent refine without structural guide. */
export function appendImageInferenceFields(
  body: Record<string, unknown>,
  composerParams: Record<string, unknown>,
  parameters?: Record<string, unknown> | null,
): void {
  appendActiveEnumFields(body, composerParams, COMPOSER_INFERENCE_ENUM_KEYS, parameters);
  appendLatentRefineField(body, composerParams, parameters);
}

export function fillModelRegistryDefaultsPatch(
  modelKey: string,
  registry: Record<string, Record<string, unknown>>,
): Partial<{ guidance: number; steps: number }> {
  const row = registry[modelKey];
  if (!row || controlNetBinding(row) !== 'standalone_edit') return {};
  const raw = row.parameters as Record<string, { default?: number }> | undefined;
  if (!raw) return {};
  const patch: Partial<{ guidance: number; steps: number }> = {};
  if (typeof raw.guidance?.default === 'number') patch.guidance = raw.guidance.default;
  if (typeof raw.steps?.default === 'number') patch.steps = raw.steps.default;
  return patch;
}

export function controlNetRegistryDefaultsPatch(
  key: string,
  compatibleControlNets: Record<string, unknown>[] | undefined,
): Partial<{ guidance: number; steps: number; controlnet_strength: number }> {
  const net = compatibleControlNets?.find((n) => String(n.key) === key);
  const raw = net?.parameters as Record<string, { default?: number }> | undefined;
  if (!raw) return {};
  const patch: Partial<{ guidance: number; steps: number; controlnet_strength: number }> = {};
  if (typeof raw.guidance?.default === 'number') patch.guidance = raw.guidance.default;
  if (typeof raw.steps?.default === 'number') patch.steps = raw.steps.default;
  const strength =
    typeof raw.controlnet_strength?.default === 'number'
      ? raw.controlnet_strength.default
      : typeof raw.redux_strength?.default === 'number'
        ? raw.redux_strength.default
        : undefined;
  if (typeof strength === 'number') patch.controlnet_strength = strength;
  return patch;
}

export function resolveControlAssetId(
  controlImage: ControlImageRef,
  messages: { assetRequired: string; required: string },
): string {
  if (controlImage?.assetId) return controlImage.assetId;
  if (controlImage?.path?.startsWith('asset:')) {
    return controlImage.path.slice('asset:'.length);
  }
  if (controlImage) throw new Error(messages.assetRequired);
  throw new Error(messages.required);
}

export type StructuralGuideValidationCode =
  | 'fill_edit_only'
  | 'flux_only'
  | 'no_img2img'
  | 'missing_control_image'
  | 'controlnet_not_ready'
  | 'runtime_unavailable';

/** First ready structural ControlNet (excludes standalone_edit). */
export function pickDefaultStructuralControlNet(
  nets: Record<string, unknown>[] | undefined,
): string {
  if (!nets?.length) return '';
  const ready = nets.find(
    (n) => controlNetReady(n) && controlNetBinding(n) !== 'standalone_edit',
  );
  if (ready) return String(ready.key || '');
  const any = nets.find((n) => controlNetBinding(n) !== 'standalone_edit');
  return any ? String(any.key || '') : '';
}

export function applyControlNetRegistryDefaults(
  controlnetKey: string,
  compatibleControlNets: Record<string, unknown>[] | undefined,
  params: {
    guidance?: number;
    steps?: number;
    controlnet_strength?: number;
  },
): void {
  const patch = controlNetRegistryDefaultsPatch(controlnetKey, compatibleControlNets);
  if (patch.guidance != null) params.guidance = patch.guidance;
  if (patch.steps != null) params.steps = patch.steps;
  if (patch.controlnet_strength != null) params.controlnet_strength = patch.controlnet_strength;
}

export function validateFillEditPrompt(prompt: string): boolean {
  return Boolean(String(prompt || '').trim());
}

export function validateStructuralGuideForCreate(opts: {
  controlnet: string;
  baseModel: string;
  hasReferenceImage: boolean;
  hasControlImage?: boolean;
  hostRuntimeAvailable?: boolean | null;
  compatibleControlNets?: Record<string, unknown>[];
  systemInfo?: { platform?: string; architecture?: string; dependencies?: Record<string, string> } | null;
}): { ok: true } | { ok: false; code: StructuralGuideValidationCode } {
  const key = String(opts.controlnet || '');
  if (!key) return { ok: true };
  if (
    !isControlNetHostRuntimeAvailable(
      opts.hostRuntimeAvailable,
      opts.compatibleControlNets,
      opts.systemInfo,
    )
  ) {
    return { ok: false, code: 'runtime_unavailable' };
  }
  const entry = opts.compatibleControlNets?.find((n) => String(n.key) === key);
  if (!entry) return { ok: false, code: 'controlnet_not_ready' };
  if (controlNetBinding(entry) === 'standalone_edit') {
    return { ok: false, code: 'fill_edit_only' };
  }
  if (opts.hasReferenceImage && !controlNetAllowsWithReference(entry)) {
    return { ok: false, code: 'no_img2img' };
  }
  if (!opts.hasControlImage) return { ok: false, code: 'missing_control_image' };
  if (!controlNetReady(entry)) return { ok: false, code: 'controlnet_not_ready' };
  return { ok: true };
}

export function useCompatibleControlNets(scope: ControlNetScope = 'create') {
  const compatibleControlNets = ref<Record<string, unknown>[]>([]);
  const loading = ref(false);

  async function loadCompatibleControlNets(modelKey: string): Promise<void> {
    if (!modelKey) {
      compatibleControlNets.value = [];
      return;
    }
    loading.value = true;
    try {
      const rows = await api.settings.getCompatibleControlNets(modelKey, scope);
      compatibleControlNets.value = (rows as Record<string, unknown>[]) || [];
    } finally {
      loading.value = false;
    }
  }

  function clearIfIncompatible(
    params: { controlnet?: unknown },
    onClear: () => void,
  ): void {
    const key = String(params.controlnet || '');
    if (!key) return;
    const listed = compatibleControlNets.value.some((n) => String(n.key) === key);
    if (!listed) {
      params.controlnet = '';
      onClear();
    }
  }

  return {
    compatibleControlNets: compatibleControlNets as Ref<Record<string, unknown>[]>,
    loading,
    loadCompatibleControlNets,
    clearIfIncompatible,
  };
}
