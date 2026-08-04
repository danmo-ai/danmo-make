/**
 * Client-side persistent storage keys (v4)
 */
export const DQ_STORAGE = Object.freeze({
  LANG: 'dq-studio.lang.v4',
  THEME: 'dq-studio.theme.v4',
  ACTIVE_PAGE: 'dq-studio.activePage.v4',
  SETTINGS_TAB: 'dq-studio.settingsTab.v4',
  IMG2IMG_REF: 'dq-studio.img2imgRef.v4',
  IMPORTED_MODELS: 'dq-studio.importedModels.v4',
  IMAGE_CREATE_PROMPT_DRAFT: 'dq-studio.imageCreatePromptDraft.v4',
  VIDEO_CREATE_PROMPT_DRAFT: 'dq-studio.videoCreatePromptDraft.v4',
  AUDIO_CREATE_PROMPT_DRAFT: 'dq-studio.audioCreatePromptDraft.v4',
  AUDIO_CREATE_LYRICS_DRAFT: 'dq-studio.audioCreateLyricsDraft.v4',
  COPILOT_HANDOFF: 'dq-studio.copilotHandoff.v4',
  COPILOT_NAV: 'dq-studio.copilotNav.v4',
  COPILOT_CARDS_COLLAPSED: 'dq-studio.copilotCardsCollapsed.v4',
  MODELS_CATEGORY: 'dq-studio.modelsCategory.v4',
  MODEL_FILTER_INSTALLED: 'dq-studio.modelFilterInstalled.v4',
  MODEL_FILTER_COMMERCIAL: 'dq-studio.modelFilterCommercial.v4',
  MODEL_FILTER_CURRENT_ONLY: 'dq-studio.modelFilterCurrentOnly.v4',
  IMAGE_LAST_SIZE: 'dq-studio.imageLastSize.v4',
  /** Per image model id → ``WxH`` (bump suffix when preset schema changes). */
  IMAGE_SIZE_BY_MODEL: 'dq-studio.imageSizeByModel.v5',
  VIDEO_LAST_SIZE: 'dq-studio.videoLastSize.v4',
  /** Per video model id → ``WxH`` (bump suffix when preset schema changes). */
  VIDEO_SIZE_BY_MODEL: 'dq-studio.videoSizeByModel.v5',
  // Per-mode model cache
  IMAGE_MODEL_TEXT2IMG: 'dq-studio.imageModel.text2img.v4',
  IMAGE_MODEL_IMG2IMG: 'dq-studio.imageModel.img2img.v4',
  IMAGE_MODEL_RETOUCH: 'dq-studio.imageModel.retouch.v4',
  IMAGE_MODEL_EXTEND: 'dq-studio.imageModel.extend.v4',
  IMAGE_MODEL_UPSCALE: 'dq-studio.imageModel.upscale.v4',
  VIDEO_MODEL_CREATE: 'dq-studio.videoModel.create.v4',
  VIDEO_MODEL_ANIMATE: 'dq-studio.videoModel.animate.v4',
  VIDEO_MODEL_EDIT: 'dq-studio.videoModel.edit.v4',
  VIDEO_MODEL_UPSCALE: 'dq-studio.videoModel.upscale.v4',
  AUDIO_MODEL_CREATE: 'dq-studio.audioModel.create.v4',
  AUDIO_MODEL_COVER: 'dq-studio.audioModel.cover.v4',
  CANVAS_IMAGE: 'dq-studio.canvas.image.v4',
  CANVAS_VIDEO: 'dq-studio.canvas.video.v4',
  CANVAS_AUDIO: 'dq-studio.canvas.audio.v4',
  CANVAS_ACTIVE_SESSION: 'dq-studio.canvas.activeSession.image.v4',
  CANVAS_ACTIVE_SESSION_VIDEO: 'dq-studio.canvas.activeSession.video.v4',
  CANVAS_ACTIVE_SESSION_AUDIO: 'dq-studio.canvas.activeSession.audio.v4',
  CANVAS_AUTO_ADD: 'dq-studio.canvas.autoAddResults.v4',
  CANVAS_AUTO_ADD_VIDEO: 'dq-studio.canvas.autoAddResults.video.v4',
  CANVAS_AUTO_ADD_AUDIO: 'dq-studio.canvas.autoAddResults.audio.v4',
  IMAGE_VIEW_MODE: 'dq-studio.imageViewMode.v4',
  VIDEO_VIEW_MODE: 'dq-studio.videoViewMode.v4',
  AUDIO_VIEW_MODE: 'dq-studio.audioViewMode.v4',
  CANVAS_PNG_EXPORT_OPTS: 'dq-studio.canvas.pngExportOpts.v4',
  CANVAS_COMPOSER_COLLAPSED_IMAGE: 'dq-studio.canvas.composerCollapsed.image.v4',
  CANVAS_COMPOSER_COLLAPSED_VIDEO: 'dq-studio.canvas.composerCollapsed.video.v4',
  CANVAS_COMPOSER_COLLAPSED_AUDIO: 'dq-studio.canvas.composerCollapsed.audio.v4',
  COMPOSER_DRAWER_IMAGE: 'dq-studio.composerDrawer.image.v4',
  COMPOSER_DRAWER_VIDEO: 'dq-studio.composerDrawer.video.v4',
  COMPOSER_DRAWER_AUDIO: 'dq-studio.composerDrawer.audio.v4',
  COMPOSER_DRAWER_AVATAR: 'dq-studio.composerDrawer.avatar.v4',
  CANVAS_WORKSPACE_HINT: 'dq-studio.canvas.workspaceHint.v4',
  CANVAS_REGION_GUIDES: 'dq-studio.canvas.regionGuides.v4',
  LORA_TRAIN_DRAFT: 'dq-studio.lora-train-draft.v4',
  SETUP_HINT_DISMISSED: 'dq-studio.setupHintDismissed.v4',
  /** Browser copy of server ``http_api_key`` for non-loopback /api (and SSE query). */
  HTTP_API_KEY: 'dq-studio.httpApiKey.v4',
} as const);

export type StorageKey = (typeof DQ_STORAGE)[keyof typeof DQ_STORAGE];

export function getItem(key: StorageKey): string | null {
  return localStorage.getItem(key);
}

export function setItem(key: StorageKey, value: string): void {
  localStorage.setItem(key, value);
}

export function removeItem(key: StorageKey): void {
  localStorage.removeItem(key);
}

/** Read-once draft handoff (e.g. Assistant → create views). */
export function consumeStringDraft(key: StorageKey): string | null {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    localStorage.removeItem(key);
    return raw;
  } catch {
    return null;
  }
}