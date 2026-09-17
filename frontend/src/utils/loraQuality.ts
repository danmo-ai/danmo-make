export type LoraQualityLevel = 'good' | 'fair' | 'poor';

export type LoraQualityHint = {
  code: string;
  severity: 'info' | 'warning' | 'error';
  params?: Record<string, unknown>;
  source?: string;
};

export type LoraVlmSummary = {
  audit_kind?: 'concept' | 'style';
  avg_score?: number;
  audited_count?: number;
  samples?: Array<VlmImageSample>;
};

export type VlmImageSample = {
  file: string;
  score?: number | null;
  vlm_score?: number | null;
  heuristic_score?: number | null;
  reason?: string;
  issues?: string[];
  source?: string;
  suitable_for_training?: boolean;
};

export function vlmScoreLevel(score: number | null | undefined): 'good' | 'fair' | 'poor' | null {
  if (score == null || Number.isNaN(Number(score))) return null;
  const s = Number(score);
  if (s >= 4) return 'good';
  if (s >= 3) return 'fair';
  return 'poor';
}

export function buildVlmSampleMap(samples: VlmImageSample[] | undefined): Map<string, VlmImageSample> {
  const map = new Map<string, VlmImageSample>();
  if (!samples?.length) return map;
  for (const sample of samples) {
    const file = String(sample.file || '').trim();
    if (!file) continue;
    map.set(file, sample);
    const base = file.split('/').pop();
    if (base && base !== file) map.set(base, sample);
  }
  return map;
}

export function lookupVlmSample(
  map: Map<string, VlmImageSample>,
  datasetFile: string
): VlmImageSample | undefined {
  const key = (datasetFile || '').trim();
  if (!key) return undefined;
  return map.get(key) ?? map.get(key.split('/').pop() || '');
}

export type FaceAuditAction = 'crop' | 'keep' | 'tiny' | 'none' | 'error';

export type FaceAuditRow = {
  file: string;
  width?: number;
  height?: number;
  faces: number;
  face_px?: number;
  face_frac?: number;
  multi?: boolean;
  action: FaceAuditAction;
  /** Training crop window `[left, top, width, height]` in source pixels (when re-framed). */
  window?: number[] | null;
  error?: string;
};

export type FaceAuditReport = {
  available: boolean;
  reason?: string;
  resolution?: number[];
  rows: FaceAuditRow[];
};

export type LoraDatasetHealthReport = {
  level: LoraQualityLevel;
  score: number;
  kind?: 'concept' | 'style';
  caption_mode_auto?: 'unified' | 'per_image';
  stats: Record<string, number>;
  hints: LoraQualityHint[];
  faces?: FaceAuditReport;
  vision_available?: boolean;
  vlm_audited?: boolean;
  audit_kind?: 'concept' | 'style';
  vlm?: LoraVlmSummary;
};

export function buildFaceRowMap(report: FaceAuditReport | undefined | null): Map<string, FaceAuditRow> {
  const map = new Map<string, FaceAuditRow>();
  if (!report?.available || !report.rows?.length) return map;
  for (const row of report.rows) {
    const file = String(row.file || '').trim();
    if (!file) continue;
    map.set(file, row);
    const base = file.split('/').pop();
    if (base && base !== file) map.set(base, row);
  }
  return map;
}

export type FaceCropOverlay = { left: number; top: number; width: number; height: number };

/**
 * Map a source-pixel crop window onto a square `object-fit: cover` thumbnail (centre crop of
 * the short edge). Values are percentages of the thumbnail box; may extend beyond 0–100 when
 * the training window reaches into the part of a tall/wide image the thumbnail does not show.
 */
export function faceCropOverlay(row: FaceAuditRow | undefined): FaceCropOverlay | null {
  if (!row?.window || row.window.length !== 4 || !row.width || !row.height) return null;
  const [l, t, w, h] = row.window.map(Number);
  if (!(w > 0) || !(h > 0)) return null;
  const side = Math.min(row.width, row.height);
  const ox = (row.width - side) / 2;
  const oy = (row.height - side) / 2;
  return {
    left: ((l - ox) / side) * 100,
    top: ((t - oy) / side) * 100,
    width: (w / side) * 100,
    height: (h / side) * 100,
  };
}

export type LoraTrainingQualityReport = {
  level: LoraQualityLevel;
  score: number;
  metrics: Record<string, unknown>;
  hints: LoraQualityHint[];
  dataset_health?: LoraDatasetHealthReport | null;
  vision_available?: boolean;
  vlm_audited?: boolean;
  audit_kind?: 'concept' | 'style';
  vlm?: LoraVlmSummary;
};

export function qualityAlertType(level: LoraQualityLevel | undefined): 'success' | 'warning' | 'error' | 'info' {
  if (level === 'poor') return 'error';
  if (level === 'fair') return 'warning';
  if (level === 'good') return 'success';
  return 'info';
}
