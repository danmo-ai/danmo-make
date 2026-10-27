export type BilingualVersionName = string | { zh?: string; en?: string };

export interface ModelVersion {
  name: BilingualVersionName;
  size?: string;
  source?: string;
  source_type?: string;
  default?: boolean;
}

/** Flatten registry version name for sort / regex (import $vn at call sites in Vue). */
export function versionNamePlain(name: BilingualVersionName | undefined, fallback = ''): string {
  if (name == null || name === '') return fallback;
  if (typeof name === 'string') return name;
  return name.zh || name.en || fallback;
}

export interface ModelVersionLayoutInput {
  id: string;
  source?: string;
  versions?: Record<string, ModelVersion>;
}

export interface VersionRow {
  verKey: string;
  ver: ModelVersion;
  vstatus: string;
}

export interface FullVersionBlock extends VersionRow {}

export interface ModelVersionLayout {
  fullBlocks: FullVersionBlock[];
  lightweightRows: VersionRow[];
  fullBlockSource: string | null;
  lightweightBlockSource: string | null;
}

export function resolveVersionSource(
  model: ModelVersionLayoutInput,
  ver: ModelVersion,
): string {
  return ver.source || model.source || '';
}

export function uniformDownloadSource(model: ModelVersionLayoutInput): string {
  if (!model.versions) return model.source || '';
  const downloadable = Object.values(model.versions);
  if (downloadable.length === 0) return model.source || '';
  const sources = downloadable.map((ver) => resolveVersionSource(model, ver));
  const unique = [...new Set(sources.filter(Boolean))];
  return unique.length === 1 ? unique[0] : '';
}

function uniformBlockSource(
  model: ModelVersionLayoutInput,
  rows: VersionRow[],
): string | null {
  if (rows.length === 0) return null;
  const sources = rows.map((row) => resolveVersionSource(model, row.ver));
  const unique = [...new Set(sources.filter(Boolean))];
  return unique.length === 1 ? unique[0] : null;
}

export function buildModelVersionLayout(
  model: ModelVersionLayoutInput,
  getVersionStatus: (modelId: string, verKey: string) => string,
): ModelVersionLayout {
  if (!model.versions) {
    return {
      fullBlocks: [],
      lightweightRows: [],
      fullBlockSource: null,
      lightweightBlockSource: null,
    };
  }

  const fullRows: VersionRow[] = [];
  const lightweightRows: VersionRow[] = [];

  for (const [verKey, ver] of Object.entries(model.versions)) {
    const row: VersionRow = {
      verKey,
      ver,
      vstatus: getVersionStatus(model.id, verKey),
    };
    if (ver.source_type === 'prequantized') {
      lightweightRows.push(row);
    } else {
      fullRows.push(row);
    }
  }

  fullRows.sort((a, b) => {
    if (a.ver.default === true && b.ver.default !== true) return -1;
    if (b.ver.default === true && a.ver.default !== true) return 1;
    return versionNamePlain(a.ver.name).localeCompare(versionNamePlain(b.ver.name));
  });

  lightweightRows.sort((a, b) =>
    versionNamePlain(a.ver.name).localeCompare(versionNamePlain(b.ver.name))
  );

  const fullBlocks: FullVersionBlock[] = fullRows;

  return {
    fullBlocks,
    lightweightRows,
    fullBlockSource: uniformBlockSource(model, fullRows),
    lightweightBlockSource: uniformBlockSource(model, lightweightRows),
  };
}

export function simplifyPrequantizedName(name: string, source: string): string {
  let text = name.trim();
  if (source === 'modelscope') {
    text = text.replace(/^魔搭\s*[·•]\s*/u, '');
  }
  text = text.replace(/预量化\s*/gu, '');
  return text.replace(/\s+/g, ' ').trim();
}
