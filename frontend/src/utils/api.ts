import axios from 'axios';
import type {
  AssetRow,
  CanvasSessionDetail,
  CanvasSessionState,
  CanvasSessionSummary,
  CharacterVisibility,
  FirstFrameStrategy,
  GalleryGroup,
  GalleryItem,
  LongVideoChainMode,
  LongVideoFlfMode,
  LongVideoProjectDetail,
  LongVideoProjectState,
  LongVideoProjectSummary,
  LongVideoSegmentRole,
  LongVideoShotCastLook,
  LongVideoShotSceneLook,
  LongVideoStartFrameMode,
  QueueState,
  RegistryData,
  SettingsData,
  SystemInfo,
} from '@/types';
import { DQ_STORAGE, getItem } from '@/utils/storage';
import { ScriptParseError, type ScriptParseQualityIssue } from '@/utils/scriptParseError';

const API_BASE = '';

/** Default REST timeout; generation/training use task SSE — not this client. */
const DEFAULT_REQUEST_TIMEOUT_MS = 30_000;
/** Dataset auto-caption (vision LLM per image) and bulk uploads can run many minutes. */
const LONG_REQUEST_TIMEOUT_MS = 600_000;
/** Single-round local LLM (prompt enhance, lyrics, vision caption) — includes cold model load. */
const LLM_REQUEST_TIMEOUT_MS = 180_000;
/** Multi-round LLM (Plan → Expand batches → Continuity), e.g. long-video storyboard. */
const LLM_MULTI_ROUND_TIMEOUT_MS = 300_000;

/** OpenAI chat.completion response. */
export type ChatCompletionResult = {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{ message: { role: string; content: string } }>;
  usage?: Record<string, unknown>;
};

export function completionText(res: ChatCompletionResult): string {
  return (res.choices?.[0]?.message?.content ?? '').trim();
}

export type LongVideoChapterAnalyzeShot = {
  id?: string;
  order?: number;
  visual_prompt?: string;
  motion_prompt?: string;
  video_prompt?: string;
  start_visual_prompt?: string;
  end_visual_prompt?: string;
  anchor_visual_prompt?: string;
  segment_role?: LongVideoSegmentRole;
  start_frame_mode?: LongVideoStartFrameMode;
  segment_group_id?: string;
  segment_group_index?: number;
  face_anchor_shot_id?: string;
  flf_mode?: LongVideoFlfMode;
  end_frame_sync_anchor?: boolean;
  chain_mode?: LongVideoChainMode;
  scene_prompt?: string;
  cast_looks?: LongVideoShotCastLook[];
  scene_look?: LongVideoShotSceneLook;
  duration_sec?: number;
  first_frame_visibility?: CharacterVisibility;
  end_visibility?: CharacterVisibility;
  characters_on_screen?: string[];
  clip_start_state?: string;
  clip_end_state?: string;
  first_frame_requirement?: string;
  camera_zone_id?: string;
  first_frame_strategy?: FirstFrameStrategy;
  location?: string;
  narrative_beat_index?: number;
  shot_size?: string;
};

export type ScriptParseDecomposeResult = {
  chapter_title: string;
  synopsis: string;
  mood?: string;
  character_anchor: string;
  style_anchor?: string;
  characters?: Array<{
    id: string;
    name: string;
    default_look_id: string;
    looks: Array<{ id: string; label: string; body: string }>;
  }>;
  scenes?: Array<{
    id: string;
    name: string;
    default_look_id: string;
    looks: Array<{ id: string; label: string; body: string }>;
    spatial_layout_json?: Record<string, unknown>;
  }>;
  scene_beats: Array<{ order: number; title?: string; beat: string }>;
  scene_count: number;
  script_artifact: Record<string, unknown>;
  parse_phases?: Array<{ phase: string; message?: string }>;
  llm_calls: number;
  parse_run_id?: string;
  long_video_project_id?: string;
};

export type ScriptParseExpandResult = ScriptParseDecomposeResult & {
  shots?: LongVideoChapterAnalyzeShot[];
  quality_warnings?: string[];
  quality_issues?: Array<{
    code: string;
    message: string;
    severity?: 'warning' | 'critical';
    shot_index?: number | null;
    beat_index?: number | null;
  }>;
};

/** @deprecated use ScriptParseExpandResult */
export type LongVideoChapterAnalyzeResult = ScriptParseExpandResult;

export type LongVideoProjectActivityItem = {
  id: string;
  project_id: string;
  category: string;
  event_type: string;
  phase?: string;
  status?: string;
  summary?: string;
  task_id?: string | null;
  parse_run_id?: string | null;
  shot_id?: string;
  detail?: Record<string, unknown>;
  created_at: string;
};

export type LongVideoParseRunDetail = {
  parse_run_id: string;
  project_id: string;
  status: string;
  started_at?: string;
  completed_at?: string;
  summary?: string;
  detail?: Record<string, unknown>;
  phases?: Array<{ phase?: string; message?: string; at?: string }>;
  events?: LongVideoProjectActivityItem[];
};

const client = axios.create({
  baseURL: API_BASE,
  timeout: DEFAULT_REQUEST_TIMEOUT_MS,
  paramsSerializer: (params) => {
    const searchParams = new URLSearchParams();
    for (const [key, value] of Object.entries(params)) {
      if (value === undefined || value === null) continue;
      if (Array.isArray(value)) {
        value.forEach((v) => searchParams.append(key, String(v)));
      } else {
        searchParams.append(key, String(value));
      }
    }
    return searchParams.toString();
  },
});

client.interceptors.request.use((config) => {
  const lang = getItem(DQ_STORAGE.LANG) || 'zh';
  config.headers = config.headers ?? {};
  config.headers['Accept-Language'] = lang;
  const apiKey = (getItem(DQ_STORAGE.HTTP_API_KEY) || '').trim();
  if (apiKey && !config.headers.Authorization && !config.headers['X-API-Key']) {
    config.headers.Authorization = `Bearer ${apiKey}`;
  }
  return config;
});

/** Append ``api_key`` for EventSource (cannot set Authorization headers). */
export function withAccessApiKey(url: string): string {
  const apiKey = (getItem(DQ_STORAGE.HTTP_API_KEY) || '').trim();
  if (!apiKey) return url;
  const sep = url.includes('?') ? '&' : '?';
  return `${url}${sep}api_key=${encodeURIComponent(apiKey)}`;
}

/** Task id from media submit 202 body `{ task: { id, ... } }` (legacy flat `{ id }` tolerated). */
export function taskIdFromSubmitResponse(res: unknown): string {
  if (res == null || typeof res !== 'object') return '';
  const top = res as Record<string, unknown>;
  const task = top.task;
  if (task != null && typeof task === 'object') {
    const nested = (task as Record<string, unknown>).id;
    if (nested != null && String(nested).trim() !== '') {
      return String(nested);
    }
  }
  const direct = top.id;
  if (direct != null && String(direct).trim() !== '') {
    return String(direct);
  }
  return '';
}


async function parseScriptParseStream<T>(
  path: string,
  body: Record<string, unknown>,
  onProgress?: (phase: string, message: string) => void,
): Promise<T> {
  const lang = getItem(DQ_STORAGE.LANG) || 'zh';
  const res = await fetch(path, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept-Language': lang,
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const errBody = (await res.json()) as { detail?: string };
      if (errBody.detail) detail = String(errBody.detail);
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  if (!res.body) {
    throw new Error('empty stream body');
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let finalResult: T | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split('\n\n');
    buffer = chunks.pop() ?? '';
    for (const chunk of chunks) {
      const line = chunk.split('\n').find((l) => l.startsWith('data: '));
      if (!line) continue;
      const payload = JSON.parse(line.slice(6)) as {
        event?: string;
        phase?: string;
        message?: string;
        pass?: string;
        detail?: string;
        quality_issues?: ScriptParseQualityIssue[];
        data?: T;
      };
      if (payload.event === 'progress' && payload.phase) {
        onProgress?.(payload.phase, payload.message ?? payload.phase);
      } else if (payload.event === 'review_retry') {
        onProgress?.('review_retry', payload.pass ?? payload.message ?? 'review_retry');
      } else if (payload.event === 'result' && payload.data) {
        finalResult = payload.data;
      } else if (payload.event === 'error') {
        throw new ScriptParseError(
          payload.detail || 'script parse failed',
          payload.quality_issues ?? [],
        );
      }
    }
  }
  if (!finalResult) {
    throw new Error('script parse stream ended without result');
  }
  return finalResult;
}

function assetRowToGalleryItem(a: AssetRow): GalleryItem {
  const aid = a.id;
  const meta = { ...(a.metadata || {}) };
  const rawPath = String(a.path || '');
  const base = rawPath.split(/[/\\]/).filter(Boolean).pop() || aid;
  const thumb = a.thumbnail_url || `${API_BASE}/api/assets/${aid}/thumbnail`;
  const durRaw = a.duration_seconds != null ? a.duration_seconds : (meta.duration_seconds as number | undefined);
  const duration_seconds =
    durRaw != null && Number.isFinite(Number(durRaw)) ? Number(durRaw) : null;
  return {
    path: `asset:${aid}`,
    name: base,
    width: Number(a.width || meta.width || 0),
    height: Number(a.height || meta.height || 0),
    duration_seconds,
    created_at: String(a.created_at || ''),
    title: String(meta.title || ''),
    prompt: String(meta.prompt || ''),
    model: String(meta.model || ''),
    thumbnail: thumb,
    source_task_id: String(a.source_task_id || meta.source_task_id || ''),
    source_action: a.source_action ?? (meta.source_action as string | null | undefined) ?? null,
    metadata: {
      ...meta,
      ...(duration_seconds != null ? { duration_seconds } : {}),
      asset_kind:
        a.kind != null && String(a.kind).trim() !== '' ? String(a.kind) : 'image',
      ...(a.parent_asset_id ? { parent_asset_id: a.parent_asset_id } : {}),
      ...(a.relation_type ? { relation_type: a.relation_type } : {}),
    },
  };
}

export const api = {
  gallery: {
    async listImages(limit = 40, offset = 0, options: Record<string, unknown> = {}): Promise<GalleryItem[]> {
      const lim = Number(limit) || 40;
      const off = Number(offset) || 0;
      const params = { limit: lim, offset: off, exclude_upload_refs: true, ...options };
      const data = await api.gen.listAssets(null, lim, off, params);
      const rows = (data.items || []).map(assetRowToGalleryItem);
      return rows;
    },

    getImageUrl(path: string): string {
      if (!path.startsWith('asset:')) {
        throw new Error('expected asset:id');
      }
      const id = path.slice('asset:'.length);
      return `${API_BASE}/api/assets/${id}/file`;
    },

    async deleteImage(path: string): Promise<unknown> {
      if (!path.startsWith('asset:')) {
        throw new Error('expected asset:id');
      }
      const id = path.slice('asset:'.length);
      return api.gen.deleteAsset(id);
    },

    async batchDeleteImages(paths: string[]): Promise<unknown> {
      const ids = paths
        .filter((p) => p.startsWith('asset:'))
        .map((p) => p.slice('asset:'.length));
      return api.gen.batchDeleteAssets(ids);
    },

    async uploadImage(file: File): Promise<unknown> {
      const formData = new FormData();
      formData.append('file', file);
      const response = await client.post('/api/gallery/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      return response.data;
    },

    async listGroups(
      kind: 'image' | 'video' | 'audio' | null = null,
      limit = 100,
      offset = 0,
    ): Promise<GalleryGroup[]> {
      const params: Record<string, unknown> = { limit, offset };
      if (kind) params.kind = kind;
      const response = await client.get('/api/assets/groups', { params });
      const rows = (response.data.items || []).map((g: Record<string, unknown>) => ({
        id: String(g.id || ''),
        title: String(g.title || ''),
        kind: String(g.kind || 'mixed'),
        asset_count: Number(g.asset_count || 0),
        preview_assets: ((g.preview_assets as unknown[]) || []).map((a) => assetRowToGalleryItem(a as AssetRow)),
        created_at: String(g.created_at || ''),
        updated_at: String(g.updated_at || ''),
        metadata: (g.metadata as Record<string, unknown>) || {},
      }));
      return rows;
    },

    groupFileUrl(groupId: string): string {
      return `${API_BASE}/api/assets/groups/${encodeURIComponent(groupId)}`;
    },
  },

  adapters: {
    async list(forModel?: string): Promise<unknown> {
      const params = forModel ? { for_model: forModel } : {};
      const response = await client.get('/api/adapters', { params });
      return response.data;
    },
  },

  models: {
    async list(params: Record<string, unknown> = {}): Promise<unknown> {
      const response = await client.get('/api/models', { params });
      return response.data;
    },

    async install(modelId: string, body: Record<string, unknown> = {}): Promise<unknown> {
      const response = await client.post(
        `/api/models/${encodeURIComponent(modelId)}/install`,
        body
      );
      return response.data;
    },

    async deleteVersion(modelId: string, versionKey: string): Promise<unknown> {
      const response = await client.delete(
        `/api/models/${encodeURIComponent(modelId)}/versions/${encodeURIComponent(versionKey)}`
      );
      return response.data;
    },

    async installBatch(
      modelIds: string[],
      items?: Array<{ model_id: string; version_key?: string }>,
    ): Promise<unknown> {
      const body = items && items.length > 0
        ? { items: items.map((row) => ({ model_id: row.model_id, version: row.version_key })) }
        : { model_ids: modelIds };
      const response = await client.post('/api/models/install-batch', body);
      return response.data;
    },
  },

  setup: {
    async getRecommendations(): Promise<unknown> {
      const response = await client.get('/api/setup/recommendations');
      return response.data;
    },
  },

  download: {
    async startDownload(url: string, targetName: string, type = 'model'): Promise<unknown> {
      const response = await client.post('/api/download/start', {
        url,
        target_name: targetName,
        type,
      });
      return response.data;
    },

    async listDownloads(): Promise<unknown> {
      const response = await client.get('/api/download/tasks');
      return response.data;
    },

    async cancel(taskId: string): Promise<void> {
      await client.post(`/api/download/cancel/${encodeURIComponent(taskId)}`);
    },

    async delete(taskId: string): Promise<void> {
      await client.delete(`/api/download/tasks/${encodeURIComponent(taskId)}`);
    },

    async resume(taskId: string): Promise<unknown> {
      const response = await client.post(
        `/api/download/resume/${encodeURIComponent(taskId)}`
      );
      return response.data;
    },

    async searchLoras(params: Record<string, unknown>): Promise<unknown> {
      const response = await client.get('/api/download/lora/search', { params });
      return response.data;
    },

    async listLoraBaseModels(): Promise<unknown> {
      const response = await client.get('/api/download/lora/base-models');
      return response.data;
    },

    async startLoraHubDownload(body: Record<string, unknown>): Promise<unknown> {
      const response = await client.post('/api/download/lora/hub', body);
      return response.data;
    },

    async startLoraDownload(url: string, filename: string): Promise<unknown> {
      const response = await client.post('/api/download/lora', { url, filename });
      return response.data;
    },

    installProgressStreamUrl(taskId: string): string {
      return withAccessApiKey(
        `${API_BASE}/api/download/progress/${encodeURIComponent(taskId)}/stream`,
      );
    },

    async startConvert(body: Record<string, unknown>): Promise<unknown> {
      const response = await client.post('/api/download/convert', body);
      return response.data;
    },

    convertProgressStreamUrl(taskId: string): string {
      return withAccessApiKey(
        `${API_BASE}/api/download/convert/${encodeURIComponent(taskId)}/stream`,
      );
    },

    async cancelConversion(taskId: string): Promise<void> {
      await client.post(`/api/download/convert/${encodeURIComponent(taskId)}/cancel`);
    },
  },

  tasks: {
    logStreamUrl(taskId: string): string {
      return withAccessApiKey(
        `${API_BASE}/api/tasks/${encodeURIComponent(taskId)}/stream`,
      );
    },

    /** Denoise step preview (work_dir PNG); polled by create view — not SSE. */
    previewUrl(taskId: string): string {
      return withAccessApiKey(
        `${API_BASE}/api/tasks/${encodeURIComponent(taskId)}/preview`,
      );
    },

    async fetchGraph(taskId: string): Promise<Record<string, unknown>> {
      const response = await client.get(`/api/tasks/${encodeURIComponent(taskId)}/graph`);
      return response.data;
    },

    async diagnose(taskId: string, body?: { locale?: string }): Promise<{ summary: string }> {
      const response = await client.post(
        `/api/tasks/${encodeURIComponent(taskId)}/diagnose`,
        body ?? {},
      );
      return response.data;
    },
  },

  settings: {
    async getSettings(): Promise<SettingsData> {
      const response = await client.get('/api/settings');
      return response.data;
    },

    async updateSettings(settings: SettingsData): Promise<{ success?: boolean; restart_required?: boolean }> {
      const response = await client.put('/api/settings', settings);
      return response.data;
    },

    async createAccessKey(kind: 'http' | 'mcp'): Promise<{
      success: boolean;
      kind: string;
      key: string;
      hint: string;
      message?: string;
    }> {
      const response = await client.post('/api/settings/access-keys', { kind });
      return response.data;
    },

    async revokeAccessKey(kind: 'http' | 'mcp'): Promise<{ success: boolean; kind: string }> {
      const response = await client.delete(`/api/settings/access-keys/${kind}`);
      return response.data;
    },

    async getWorkspacePaths(): Promise<Record<string, string>> {
      const response = await client.get('/api/settings/workspace-paths');
      return response.data;
    },

    async getWorkspaceStatus(): Promise<{
      configured: boolean;
      effective_root: string;
      bootstrap_root: string;
      control_plane?: string;
    }> {
      const response = await client.get('/api/settings/workspace-status');
      return response.data;
    },

    async applyWorkspace(path: string): Promise<{
      success: boolean;
      restart_required: boolean;
      workspace?: string;
    }> {
      const response = await client.post('/api/settings/apply-workspace', { path });
      return response.data;
    },

    async pickWorkspaceDirectory(): Promise<{ path: string }> {
      const response = await client.post('/api/settings/pick-workspace-directory');
      return response.data;
    },

    async restoreConfigDefaults(files?: string[]): Promise<{
      success: boolean;
      restored: string[];
      restart_required: boolean;
    }> {
      const body = files && files.length > 0 ? { files } : {};
      const response = await client.post('/api/settings/restore-config-defaults', body);
      return response.data;
    },

    async listModels(): Promise<unknown> {
      const response = await client.get('/api/settings/models');
      return response.data;
    },

    async listLoras(): Promise<unknown> {
      const response = await client.get('/api/settings/loras');
      return response.data;
    },

    async refreshModels(): Promise<unknown> {
      const response = await client.post('/api/settings/refresh');
      return response.data;
    },

    async getSystemInfo(): Promise<SystemInfo> {
      const response = await client.get('/api/settings/system');
      return response.data;
    },

    async installEnvironment(): Promise<unknown> {
      const response = await client.post('/api/settings/install');
      return response.data;
    },

    async getModelRegistry(): Promise<unknown> {
      const response = await client.get('/api/settings/registry');
      return response.data;
    },

    async getModelsStatus(): Promise<unknown> {
      const response = await client.get('/api/settings/models/status');
      return response.data;
    },

    async getModelsDetailedStatus(): Promise<unknown> {
      const response = await client.get('/api/settings/models/status/detailed');
      return response.data;
    },

    async getDiskSpace(): Promise<unknown> {
      const response = await client.get('/api/settings/disk-space');
      return response.data;
    },

    async getCompatibleLoras(modelName: string): Promise<unknown> {
      const response = await client.get(`/api/settings/loras/compatible/${modelName}`);
      return response.data;
    },

    async getCompatibleControlNets(
      modelName: string,
      scope?: 'create' | 'retouch' | 'extend',
    ): Promise<unknown> {
      const response = await client.get(`/api/settings/controlnets/compatible/${modelName}`, {
        params: scope ? { scope } : undefined,
      });
      return response.data;
    },

    async updateModelParameters(modelName: string, parameters: Record<string, unknown>): Promise<unknown> {
      const response = await client.post(`/api/settings/models/${modelName}/parameters`, parameters);
      return response.data;
    },

    async getSystemMonitor(): Promise<unknown> {
      const response = await client.get('/api/settings/system/monitor');
      return response.data;
    },

    async getPresets(): Promise<unknown> {
      const response = await client.get('/api/settings/presets');
      return response.data;
    },

    async savePreset(name: string, preset: Record<string, unknown>): Promise<unknown> {
      const response = await client.post('/api/settings/presets', { name, preset });
      return response.data;
    },

    async deletePreset(name: string): Promise<unknown> {
      const response = await client.delete(`/api/settings/presets/${encodeURIComponent(name)}`);
      return response.data;
    },
  },

  gen: {
    async uploadAsset(file: File): Promise<unknown> {
      const fd = new FormData();
      fd.append('file', file);
      const response = await client.post('/api/assets', fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      return response.data;
    },

    async listAssets(
      kind: string | null,
      limit = 100,
      offset = 0,
      options: Record<string, unknown> = {}
    ): Promise<{ items: AssetRow[]; total?: number }> {
      const params: Record<string, unknown> = { limit, offset, ...options };
      if (kind != null && kind !== '') params.kind = kind;
      const response = await client.get('/api/assets', { params });
      return response.data;
    },

    async deleteAsset(assetId: string): Promise<unknown> {
      const response = await client.delete(`/api/assets/${encodeURIComponent(assetId)}`);
      return response.data;
    },

    async batchDeleteAssets(assetIds: string[]): Promise<unknown> {
      const response = await client.post('/api/assets/batch-delete', {
        asset_ids: assetIds,
      });
      return response.data;
    },

    async reconcileAssets(dryRun = true): Promise<unknown> {
      const response = await client.post('/api/assets/reconcile', {
        dry_run: dryRun,
      });
      return response.data;
    },

    async urlToBlob(url: string): Promise<Blob> {
      if (!url.trim()) {
        throw new Error('urlToBlob: expected non-empty string');
      }
      const u = url.trim();
      if (u.startsWith('blob:') || u.startsWith('data:')) {
        const r = await fetch(u);
        return r.blob();
      }
      const response = await client.get(u, { responseType: 'blob' });
      return response.data as Blob;
    },

    async getAssetLineage(assetId: string): Promise<unknown> {
      const response = await client.get(`/api/assets/${encodeURIComponent(assetId)}/lineage`);
      return response.data;
    },

    async createImageGeneration(body: Record<string, unknown>): Promise<unknown> {
      const response = await client.post('/api/images/generations', body);
      return response.data;
    },

    async createImageEdit(body: Record<string, unknown>): Promise<unknown> {
      const response = await client.post('/api/images/edits', body);
      return response.data;
    },

    async createImageUpscale(body: Record<string, unknown>): Promise<unknown> {
      const response = await client.post('/api/images/upscales', body);
      return response.data;
    },

    async createVideoGeneration(body: Record<string, unknown>): Promise<unknown> {
      const response = await client.post('/api/videos/generations', body);
      return response.data;
    },

    async createVideoLongGeneration(body: Record<string, unknown>): Promise<unknown> {
      const response = await client.post('/api/videos/long-generations', body);
      return response.data;
    },

    async createVideoEdit(body: Record<string, unknown>): Promise<unknown> {
      const response = await client.post('/api/videos/edits', body);
      return response.data;
    },

    async createVideoAvatar(body: Record<string, unknown>): Promise<unknown> {
      const response = await client.post('/api/videos/avatars', body);
      return response.data;
    },

    async createVideoAvatarScript(body: Record<string, unknown>): Promise<unknown> {
      const response = await client.post('/api/videos/avatars/script', body);
      return response.data;
    },

    async createVideoUpscale(body: Record<string, unknown>): Promise<unknown> {
      const response = await client.post('/api/videos/upscales', body);
      return response.data;
    },

    async getMediaTask(taskId: string): Promise<unknown> {
      const response = await client.get(`/api/tasks/${taskId}`);
      return response.data;
    },

    async listMediaTasks({
      limit = 100,
      offset = 0,
      kind,
      status,
      since,
    }: {
      limit?: number;
      offset?: number;
      kind?: string;
      status?: string;
      since?: string;
    } = {}): Promise<unknown> {
      const params = new URLSearchParams();
      params.set('limit', String(limit));
      params.set('offset', String(offset));
      if (kind != null && kind !== '') params.set('kind', kind);
      if (status != null && status !== '') params.set('status', status);
      if (since != null && since !== '') params.set('since', since);
      const response = await client.get(`/api/tasks?${params.toString()}`);
      return response.data;
    },

    async getMediaTaskLogs(
      taskId: string,
      { offset = 0, limit = 500 }: { offset?: number; limit?: number } = {}
    ): Promise<unknown> {
      const params = new URLSearchParams();
      params.set('offset', String(offset));
      params.set('limit', String(limit));
      const response = await client.get(
        `/api/tasks/${encodeURIComponent(taskId)}/logs?${params.toString()}`
      );
      return response.data;
    },

    async patchMediaTaskPriority(taskId: string, body: { priority: string }): Promise<unknown> {
      const response = await client.patch(
        `/api/tasks/${encodeURIComponent(taskId)}`,
        body
      );
      return response.data;
    },

    async cancelMediaTask(taskId: string): Promise<{ ok?: boolean }> {
      const response = await client.delete(`/api/tasks/${encodeURIComponent(taskId)}`);
      return response.data;
    },

    async getQueue(): Promise<QueueState> {
      const response = await client.get('/api/queue');
      return response.data;
    },

    streamMediaTask(
      taskId: string,
      callbacks: {
        onLog?: (data: unknown) => void;
        onStatus?: (data: unknown) => void;
        onDone?: (data: unknown) => void;
        onError?: (event: Event) => void;
        onProgress?: (data: unknown) => void;
        onResult?: (data: unknown) => void;
        onTrace?: (data: unknown) => void;
      }
    ): EventSource {
      const url = api.tasks.logStreamUrl(taskId);
      const eventSource = new EventSource(url);

      eventSource.addEventListener('log', (event) => {
        const data = JSON.parse(event.data);
        callbacks.onLog?.(data);
      });

      eventSource.addEventListener('trace', (event) => {
        const data = JSON.parse(event.data);
        callbacks.onTrace?.(data);
      });

      eventSource.addEventListener('progress', (event) => {
        const data = JSON.parse(event.data);
        callbacks.onProgress?.(data);
        if (!callbacks.onProgress && callbacks.onStatus) {
          callbacks.onStatus?.({
            status: 'running',
            progress: data.progress,
            step: data.step,
            total: data.total,
            eta_seconds: data.eta_seconds,
          });
        }
      });

      eventSource.addEventListener('status', (event) => {
        const data = JSON.parse(event.data);
        callbacks.onStatus?.(data);
      });

      eventSource.addEventListener('result', (event) => {
        const data = JSON.parse(event.data);
        callbacks.onResult?.(data);
      });

      eventSource.addEventListener('done', (event) => {
        const data = JSON.parse(event.data);
        callbacks.onDone?.(data);
        eventSource.close();
      });

      eventSource.addEventListener('error', (event) => {
        callbacks.onError?.(event);
        eventSource.close();
      });

      return eventSource;
    },

    async enhancePrompt(body: {
      prompt: string;
      style_positive?: string;
      style_negative?: string;
      target_action?: string;
      model_id?: string;
    }): Promise<{ enhanced_prompt: string }> {
      const response = await client.post('/api/chat/enhance-prompt', body, {
        timeout: LLM_REQUEST_TIMEOUT_MS,
      });
      return response.data;
    },

    async chatCompletion(body: {
      model?: string;
      messages: Array<{
        role: string;
        content:
          | string
          | Array<
              | { type: 'text'; text: string }
              | { type: 'image_url'; image_url: { url: string } }
            >;
      }>;
      temperature?: number;
      max_tokens?: number;
      stream?: boolean;
      top_p?: number;
    }): Promise<ChatCompletionResult> {
      const response = await client.post('/v1/chat/completions', body, {
        timeout: body.messages.some(
          (m) =>
            Array.isArray(m.content) &&
            m.content.some((p) => p.type === 'image_url'),
        )
          ? LLM_REQUEST_TIMEOUT_MS
          : DEFAULT_REQUEST_TIMEOUT_MS,
      });
      return response.data;
    },

    async scriptParseDecomposeStream(
      body: {
        script_text: string;
        title?: string;
        locale?: string;
        long_video_project_id?: string;
        model?: string;
      },
      onProgress?: (phase: string, message: string) => void,
    ): Promise<ScriptParseDecomposeResult> {
      return parseScriptParseStream<ScriptParseDecomposeResult>(
        '/api/script-parse/decompose/stream',
        body,
        onProgress,
      );
    },

    async scriptParseExpandStream(
      body: {
        script_artifact: Record<string, unknown>;
        locale?: string;
        target_duration_sec?: number;
        segment_duration_sec?: number;
        max_clip_sec?: number;
        long_video_project_id?: string;
        model?: string;
        beat_indices?: number[];
      },
      onProgress?: (phase: string, message: string) => void,
    ): Promise<ScriptParseExpandResult> {
      return parseScriptParseStream<ScriptParseExpandResult>(
        '/api/script-parse/expand/stream',
        body,
        onProgress,
      );
    },

    async scriptParseExpandBeatStream(
      body: {
        script_artifact: Record<string, unknown>;
        beat_index: number;
        existing_shots?: Array<Record<string, unknown>>;
        locale?: string;
        target_duration_sec?: number;
        segment_duration_sec?: number;
        max_clip_sec?: number;
        long_video_project_id?: string;
        model?: string;
      },
      onProgress?: (phase: string, message: string) => void,
    ): Promise<ScriptParseExpandResult> {
      return parseScriptParseStream<ScriptParseExpandResult>(
        '/api/script-parse/expand/beat/stream',
        body,
        onProgress,
      );
    },

    async longVideoChapterAnalyzeStream(
      body: {
        chapter_text: string;
        chapter_title?: string;
        locale?: string;
        target_duration_sec?: number;
        segment_duration_sec?: number;
        max_clip_sec?: number;
        long_video_project_id?: string;
        model?: string;
      },
      onProgress?: (phase: string, message: string) => void,
    ): Promise<LongVideoChapterAnalyzeResult> {
      const decomposed = await this.scriptParseDecomposeStream(
        {
          script_text: body.chapter_text,
          title: body.chapter_title,
          locale: body.locale,
          long_video_project_id: body.long_video_project_id,
          model: body.model,
        },
        onProgress,
      );
      return this.scriptParseExpandStream(
        {
          script_artifact: decomposed.script_artifact,
          locale: body.locale,
          target_duration_sec: body.target_duration_sec,
          segment_duration_sec: body.segment_duration_sec,
          max_clip_sec: body.max_clip_sec,
          long_video_project_id: body.long_video_project_id,
          model: body.model,
        },
        onProgress,
      );
    },

    async getLlmModelInfo(): Promise<{
      model_id: string;
      name: string | { zh?: string; en?: string };
      available: boolean;
    }> {
      const response = await client.get('/v1/llm/model');
      return response.data;
    },

    async getVisionModelInfo(): Promise<{
      model_id: string;
      name: string | { zh?: string; en?: string };
      available: boolean;
      mlx_vlm_installed?: boolean;
    }> {
      const response = await client.get('/v1/vision/model');
      return response.data;
    },

    async *chatCompletionStream(body: {
      model?: string;
      messages: Array<{
        role: string;
        content:
          | string
          | Array<
              | { type: 'text'; text: string }
              | { type: 'image_url'; image_url: { url: string } }
            >;
      }>;
      temperature?: number;
      max_tokens?: number;
      top_p?: number;
    }): AsyncGenerator<string, void, unknown> {
      const response = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...body, stream: true }),
      });
      if (!response.body) return;
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data: ')) continue;
          const data = trimmed.slice(6);
          if (data === '[DONE]') return;
          try {
            const parsed = JSON.parse(data);
            const content = parsed.choices?.[0]?.delta?.content;
            if (content) yield content;
          } catch {
            // ignore malformed JSON
          }
        }
      }
    },
  },

  registry: {
    async getFull(): Promise<RegistryData> {
      const response = await client.get('/api/registry');
      return response.data;
    },
  },

  system: {
    async health(): Promise<unknown> {
      const response = await client.get('/api/system/health');
      return response.data;
    },

    async metrics(): Promise<unknown> {
      const response = await client.get('/api/system/metrics');
      return response.data;
    },

    async getCacheStatus(): Promise<unknown> {
      const response = await client.get('/api/system/cache');
      return response.data;
    },
  },

  canvas: {
    async listSessions(media = 'image', limit = 50): Promise<CanvasSessionSummary[]> {
      const response = await client.get('/api/canvas/sessions', { params: { media, limit } });
      return (response.data?.items || []) as CanvasSessionSummary[];
    },

    async getSession(sessionId: string): Promise<CanvasSessionDetail> {
      const response = await client.get(`/api/canvas/sessions/${encodeURIComponent(sessionId)}`);
      return response.data as CanvasSessionDetail;
    },

    async createSession(
      payload: { media?: string; title?: string; state?: CanvasSessionState } = {}
    ): Promise<CanvasSessionDetail> {
      const response = await client.post('/api/canvas/sessions', payload);
      return response.data as CanvasSessionDetail;
    },

    async updateSession(
      sessionId: string,
      payload: { title?: string; state?: CanvasSessionState }
    ): Promise<CanvasSessionDetail> {
      const response = await client.put(
        `/api/canvas/sessions/${encodeURIComponent(sessionId)}`,
        payload
      );
      return response.data as CanvasSessionDetail;
    },

    async deleteSession(sessionId: string): Promise<void> {
      await client.delete(`/api/canvas/sessions/${encodeURIComponent(sessionId)}`);
    },
  },

  longVideo: {
    async listProjects(limit = 100): Promise<LongVideoProjectSummary[]> {
      const response = await client.get('/api/long-video/projects', { params: { limit } });
      return (response.data?.items || []) as LongVideoProjectSummary[];
    },

    async getProject(projectId: string): Promise<LongVideoProjectDetail> {
      const response = await client.get(`/api/long-video/projects/${encodeURIComponent(projectId)}`);
      return response.data as LongVideoProjectDetail;
    },

    async createProject(body: {
      title?: string;
      state?: Partial<LongVideoProjectState>;
    }): Promise<LongVideoProjectDetail> {
      const response = await client.post('/api/long-video/projects', body);
      return response.data as LongVideoProjectDetail;
    },

    async updateProject(
      projectId: string,
      body: { title?: string; state?: Partial<LongVideoProjectState> },
    ): Promise<LongVideoProjectDetail> {
      const response = await client.put(
        `/api/long-video/projects/${encodeURIComponent(projectId)}`,
        body,
      );
      return response.data as LongVideoProjectDetail;
    },

    async deleteProject(projectId: string): Promise<void> {
      await client.delete(`/api/long-video/projects/${encodeURIComponent(projectId)}`);
    },

    async listProjectActivity(
      projectId: string,
      params?: {
        limit?: number;
        offset?: number;
        category?: string;
        phase?: string;
        event_type?: string;
        parse_run_id?: string;
        task_id?: string;
        shot_id?: string;
      },
    ): Promise<{ items: LongVideoProjectActivityItem[]; total: number }> {
      const response = await client.get(
        `/api/long-video/projects/${encodeURIComponent(projectId)}/activity`,
        { params },
      );
      return response.data as { items: LongVideoProjectActivityItem[]; total: number };
    },

    async getParseRun(
      projectId: string,
      parseRunId: string,
    ): Promise<LongVideoParseRunDetail> {
      const response = await client.get(
        `/api/long-video/projects/${encodeURIComponent(projectId)}/activity/parse-runs/${encodeURIComponent(parseRunId)}`,
      );
      return response.data as LongVideoParseRunDetail;
    },

    async sceneGroundingDepthFromAsset(body: {
      source_asset_id: string;
      width?: number;
      height?: number;
    }): Promise<{ depth_asset_id: string; panorama_asset_id: string }> {
      const response = await client.post('/api/long-video/scene-grounding/depth-from-asset', body);
      return response.data as { depth_asset_id: string; panorama_asset_id: string };
    },
  },

  loras: {
    async listDatasets(): Promise<unknown> {
      const response = await client.get('/api/loras/datasets');
      return response.data;
    },

    async createDataset(body: Record<string, unknown>): Promise<unknown> {
      const response = await client.post('/api/loras/datasets', body);
      return response.data;
    },

    async getDataset(id: string): Promise<unknown> {
      const response = await client.get(`/api/loras/datasets/${encodeURIComponent(id)}`);
      return response.data;
    },

    async datasetHealth(id: string): Promise<unknown> {
      const response = await client.get(`/api/loras/datasets/${encodeURIComponent(id)}/health`);
      return response.data;
    },

    async datasetHealthVlm(
      id: string,
      opts?: { maxSamples?: number; auditKind?: 'concept' | 'style' }
    ): Promise<unknown> {
      const response = await client.post(
        `/api/loras/datasets/${encodeURIComponent(id)}/health/vlm`,
        {
          max_samples: opts?.maxSamples ?? 0,
          audit_kind: opts?.auditKind,
        },
        { timeout: LONG_REQUEST_TIMEOUT_MS }
      );
      return response.data;
    },

    async patchDataset(id: string, body: Record<string, unknown>): Promise<unknown> {
      const response = await client.patch(`/api/loras/datasets/${encodeURIComponent(id)}`, body);
      return response.data;
    },

    async deleteDataset(id: string): Promise<void> {
      await client.delete(`/api/loras/datasets/${encodeURIComponent(id)}`);
    },

    async uploadImages(datasetId: string, files: File[], defaultPrompt?: string): Promise<unknown> {
      const form = new FormData();
      for (const f of files) form.append('files', f);
      if (defaultPrompt) form.append('default_prompt', defaultPrompt);
      const response = await client.post(
        `/api/loras/datasets/${encodeURIComponent(datasetId)}/images`,
        form,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
          timeout: LONG_REQUEST_TIMEOUT_MS,
        }
      );
      return response.data;
    },

    async updateCaptions(datasetId: string, captions: Array<{ file: string; prompt: string }>): Promise<unknown> {
      const response = await client.patch(
        `/api/loras/datasets/${encodeURIComponent(datasetId)}/captions`,
        { captions }
      );
      return response.data;
    },

    async deleteDatasetImage(datasetId: string, file: string): Promise<unknown> {
      const response = await client.delete(
        `/api/loras/datasets/${encodeURIComponent(datasetId)}/images/${encodeURIComponent(file)}`
      );
      return response.data;
    },

    async importDog6(): Promise<unknown> {
      const response = await client.post('/api/loras/datasets/import-dog6', undefined, {
        timeout: LONG_REQUEST_TIMEOUT_MS,
      });
      return response.data;
    },

    async importAssets(
      datasetId: string,
      assetIds: string[],
      defaultPrompt?: string,
      captions?: Record<string, string>
    ): Promise<unknown> {
      const response = await client.post(
        `/api/loras/datasets/${encodeURIComponent(datasetId)}/import-assets`,
        { asset_ids: assetIds, default_prompt: defaultPrompt || '', captions: captions || {} }
      );
      return response.data;
    },

    async autoCaption(datasetId: string, files?: string[]): Promise<unknown> {
      const response = await client.post(
        `/api/loras/datasets/${encodeURIComponent(datasetId)}/auto-caption`,
        { files: files || [] },
        { timeout: LONG_REQUEST_TIMEOUT_MS }
      );
      return response.data;
    },

    async listUserAdapters(): Promise<unknown> {
      const response = await client.get('/api/loras/user-adapters');
      return response.data;
    },

    async listDownloadedAdapters(): Promise<unknown> {
      const response = await client.get('/api/loras/downloaded-adapters');
      return response.data;
    },

    async deleteUserAdapter(loraId: string, removeFiles = false): Promise<unknown> {
      const response = await client.delete(
        `/api/loras/user-adapters/${encodeURIComponent(loraId)}`,
        { params: { remove_files: removeFiles } }
      );
      return response.data;
    },

    async registerCheckpoint(
      taskId: string,
      body: { checkpoint: string; name?: string }
    ): Promise<unknown> {
      const response = await client.post(
        `/api/loras/trainings/${encodeURIComponent(taskId)}/register`,
        body
      );
      return response.data;
    },

    datasetImageUrl(datasetId: string, file: string): string {
      return `${API_BASE}/api/loras/datasets/${encodeURIComponent(datasetId)}/file/${encodeURIComponent(file)}`;
    },

    async trainableModels(): Promise<unknown> {
      const response = await client.get('/api/loras/trainable-models');
      return response.data;
    },

    async trainingRequirements(baseModel?: string, qloraBits?: number | null): Promise<unknown> {
      const params: Record<string, string | number> = {};
      if (baseModel) params.base_model = baseModel;
      if (qloraBits === 4 || qloraBits === 8) params.qlora_bits = qloraBits;
      const response = await client.get('/api/loras/training/requirements', {
        params: Object.keys(params).length ? params : undefined,
      });
      return response.data;
    },

    async submitTraining(body: Record<string, unknown>): Promise<unknown> {
      const response = await client.post('/api/loras/trainings', body);
      return response.data;
    },

    async trainingArtifacts(taskId: string): Promise<unknown> {
      const response = await client.get(`/api/loras/trainings/${encodeURIComponent(taskId)}/artifacts`);
      return response.data;
    },

    async trainingQualityVlm(taskId: string): Promise<unknown> {
      const response = await client.post(
        `/api/loras/trainings/${encodeURIComponent(taskId)}/quality/vlm`,
        undefined,
        { timeout: LONG_REQUEST_TIMEOUT_MS }
      );
      return response.data;
    },

    artifactFileUrl(taskId: string, filename: string): string {
      return `${API_BASE}/api/loras/trainings/${encodeURIComponent(taskId)}/artifacts/file/${encodeURIComponent(filename)}`;
    },
  },

  tools: {
    async submitZImageMerge(body: Record<string, unknown>): Promise<unknown> {
      const response = await client.post('/api/tools/z-image/merge', body);
      return response.data;
    },

    async listZImageMergeModels(): Promise<{ models: Array<{ id: string; name: unknown }>; mlx_available: boolean }> {
      const response = await client.get('/api/tools/z-image/merge/models');
      return response.data;
    },

    async listUserMergedZImageModels(): Promise<{
      items: Array<{
        id: string;
        name: string;
        local_path: string;
        template_model?: string;
        created_at?: string;
      }>;
    }> {
      const response = await client.get('/api/tools/z-image/merge/merged');
      return response.data;
    },
  },

  audios: {
    async listGenerations(): Promise<unknown> {
      const response = await client.get('/api/audios/generations');
      return response.data;
    },

    async createGeneration(body: Record<string, unknown> = {}): Promise<unknown> {
      const response = await client.post('/api/audios/generations', body);
      return response.data;
    },

    async createEdit(body: Record<string, unknown> = {}): Promise<unknown> {
      const response = await client.post('/api/audios/edits', body);
      return response.data;
    },

    async createDub(body: Record<string, unknown> = {}): Promise<unknown> {
      const response = await client.post('/api/audios/dubs', body);
      return response.data;
    },
  },
};

export default api;