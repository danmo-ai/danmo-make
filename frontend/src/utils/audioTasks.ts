export const AUDIO_WORK_MODES = ['create', 'cover', 'repaint', 'lego', 'extract', 'complete'] as const;
export type AudioWorkMode = (typeof AUDIO_WORK_MODES)[number];
export type AudioEditMode = Exclude<AudioWorkMode, 'create'>;

export const AUDIO_TRACKS = [
  'vocals',
  'backing_vocals',
  'drums',
  'bass',
  'guitar',
  'keyboard',
  'percussion',
  'strings',
  'synth',
  'fx',
  'brass',
  'woodwinds',
] as const;

export function isAudioWorkMode(value: unknown): value is AudioWorkMode {
  return typeof value === 'string' && AUDIO_WORK_MODES.includes(value as AudioWorkMode);
}

export function supportsAudioAction(config: Record<string, any> | null | undefined, mode: AudioWorkMode): boolean {
  const actions = config?.actions;
  if (!actions || typeof actions !== 'object') return false;
  if (Array.isArray(actions)) return actions.includes(mode);
  return Object.prototype.hasOwnProperty.call(actions, mode) && actions[mode] != null;
}

export function audioModeNeedsRegion(mode: AudioWorkMode): boolean {
  return mode === 'repaint' || mode === 'lego';
}

export function audioModeTrackArity(mode: AudioWorkMode): 'none' | 'single' | 'multi' {
  if (mode === 'lego' || mode === 'extract') return 'single';
  if (mode === 'complete') return 'multi';
  return 'none';
}

export function serializeAudioTracks(mode: AudioWorkMode, tracks: unknown): string {
  const values = Array.isArray(tracks) ? tracks.map(String).map((v) => v.trim()).filter(Boolean) : [];
  const arity = audioModeTrackArity(mode);
  if (arity === 'single') return values[0] || '';
  if (arity === 'multi') return values.join(' | ');
  return '';
}

export function parseAudioTracks(value: unknown): string[] {
  return String(value || '')
    .split('|')
    .map((track) => track.trim().toLowerCase())
    .filter((track) => (AUDIO_TRACKS as readonly string[]).includes(track));
}

export function normalizeAudioRegion(
  start: unknown,
  end: unknown,
  duration: unknown,
): { start: number; end: number } | null {
  const total = Number(duration);
  let from = Number(start);
  let to = Number(end);
  if (!Number.isFinite(total) || total <= 0 || !Number.isFinite(from) || !Number.isFinite(to)) return null;
  from = Math.min(Math.max(0, from), total);
  to = Math.min(Math.max(0, to), total);
  if (to <= from) return null;
  return { start: from, end: to };
}

export function dcwEnabledForSubmit(value: unknown): boolean | null {
  if (value === true || value === 'on') return true;
  if (value === false || value === 'off') return false;
  return null;
}

export interface AudioPayloadInput {
  model: string;
  title?: string;
  prompt?: string;
  lyrics?: string;
  instrumental?: boolean;
  vocal_language?: string;
  vocal_type?: string;
  bpm?: number | null;
  key_scale?: string;
  time_signature?: string;
  duration?: number | null;
  steps?: number | null;
  guidance?: number | null;
  seed?: number | null;
  n?: number;
  audio_format?: string;
  denoise?: string;
  lm_expansion?: string;
  audio_code_indices?: number[] | null;
  adapters?: Array<{ id: string; weight: number }>;
  metadata?: Record<string, unknown>;
  cover_variant?: string;
  source_fidelity?: number;
  cover_noise_strength?: number;
  timbre_from_source?: boolean;
  region_start_sec?: number | null;
  region_end_sec?: number | null;
  repaint_injection_ratio?: number;
  repaint_crossfade_frames?: number;
  tracks?: string[];
  dcw_enabled?: unknown;
  dcw_mode?: string;
  dcw_scaler?: number;
  dcw_high_scaler?: number;
}

function audioCommonPayload(input: AudioPayloadInput): Record<string, unknown> {
  return {
    model: input.model,
    title: String(input.title || '').trim(),
    prompt: String(input.prompt || '').trim(),
    lyrics: String(input.lyrics || '').trim(),
    instrumental: Boolean(input.instrumental),
    vocal_language: String(input.vocal_language || '').trim(),
    vocal_type: String(input.vocal_type || '').trim(),
    bpm: input.bpm ?? null,
    key_scale: String(input.key_scale || '').trim(),
    time_signature: String(input.time_signature || '').trim(),
    duration: input.duration ?? null,
    steps: input.steps ?? null,
    guidance: input.guidance ?? null,
    seed: input.seed ?? null,
    n: input.n ?? 1,
    audio_format: input.audio_format || 'wav',
    denoise: input.denoise || 'off',
    adapters: input.adapters || [],
    metadata: input.metadata || {},
    dcw_enabled: dcwEnabledForSubmit(input.dcw_enabled),
    dcw_mode: input.dcw_mode || 'double',
    dcw_wavelet: 'haar',
    dcw_scaler: input.dcw_scaler ?? 0.05,
    dcw_high_scaler: input.dcw_high_scaler ?? 0.02,
  };
}

export function buildAudioGenerationPayload(input: AudioPayloadInput): Record<string, unknown> {
  return {
    ...audioCommonPayload(input),
    lm_expansion: input.lm_expansion || 'auto',
    audio_code_indices: input.audio_code_indices?.length ? input.audio_code_indices : null,
  };
}

export function buildAudioEditPayload(
  mode: AudioEditMode,
  sourceAssetId: string,
  input: AudioPayloadInput,
): Record<string, unknown> {
  const task = mode === 'cover' ? input.cover_variant || 'cover' : mode;
  const body: Record<string, unknown> = {
    ...audioCommonPayload(input),
    operation: mode,
    task,
    source_asset_id: sourceAssetId,
    source_fidelity: input.source_fidelity ?? 1,
    audio_cover_strength: input.source_fidelity ?? 1,
    cover_noise_strength: input.cover_noise_strength ?? 0,
    timbre_from_source: Boolean(input.timbre_from_source),
    track: serializeAudioTracks(mode, input.tracks),
  };
  if (audioModeNeedsRegion(mode)) {
    body.region_start_sec = input.region_start_sec;
    body.region_end_sec = input.region_end_sec;
    body.repaint_injection_ratio = input.repaint_injection_ratio ?? 1;
    body.repaint_crossfade_frames = input.repaint_crossfade_frames ?? 8;
  }
  return body;
}

export function validateAudioEditInput(
  mode: AudioEditMode,
  input: AudioPayloadInput & { sourceAssetId?: string; sourceDuration?: number },
): string | null {
  if (!String(input.sourceAssetId || '').trim()) return 'source';
  const arity = audioModeTrackArity(mode);
  if (arity !== 'none' && !serializeAudioTracks(mode, input.tracks)) return 'track';
  if (audioModeNeedsRegion(mode)) {
    const region = normalizeAudioRegion(
      input.region_start_sec,
      input.region_end_sec,
      input.sourceDuration,
    );
    if (!region) return 'region';
  }
  return null;
}
