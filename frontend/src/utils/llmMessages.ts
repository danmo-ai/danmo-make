import { api } from '@/utils/api';

/** Prompt enhance via product API (server owns the system prompt). */
export async function enhancePromptViaChat(
  prompt: string,
  opts?: { stylePositive?: string; targetAction?: string },
): Promise<string> {
  const res = await api.gen.enhancePrompt({
    prompt,
    style_positive: opts?.stylePositive,
    target_action: opts?.targetAction,
  });
  return (res.enhanced_prompt || '').trim();
}

export async function generateLyricsViaChat(prompt: string, style?: string): Promise<string> {
  const res = await api.gen.assistantRun({
    kind: 'lyrics',
    prompt,
    style,
  });
  return (res.text || '').trim();
}

export async function imageToPromptViaChat(
  assetId: string,
  opts?: { video?: boolean },
): Promise<string> {
  const res = await api.gen.assistantRun({
    kind: 'image_to_prompt',
    asset_id: assetId,
    video: opts?.video,
  });
  return (res.text || '').trim();
}

export async function analyzeReferenceViaChat(
  assetId: string,
  question: string,
  opts?: { locale?: string; video?: boolean },
): Promise<string> {
  const res = await api.gen.assistantRun({
    kind: 'analyze_reference',
    asset_id: assetId,
    question,
    locale: opts?.locale,
    video: opts?.video,
  });
  return (res.text || '').trim();
}

export async function describeCanvasNodeViaChat(
  assetId: string,
  opts?: { preferVision?: boolean },
): Promise<{ note: string; visionUsed: boolean }> {
  const preferVision = opts?.preferVision !== false;
  const res = await api.gen.assistantRun({
    kind: 'canvas_describe',
    asset_id: assetId,
    prefer_vision: preferVision,
  });
  return {
    note: (res.text || '').trim(),
    visionUsed: Boolean(res.vision_used),
  };
}
