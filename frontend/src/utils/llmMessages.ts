import { api, completionText } from '@/utils/api';

export type ChatContentPart =
  | { type: 'text'; text: string }
  | { type: 'image_url'; image_url: { url: string } };

export type ChatMessagePayload = {
  role: 'system' | 'user' | 'assistant';
  content: string | ChatContentPart[];
};

const ENHANCE_IMAGE_SYSTEM = `# Role

Prompt engineer for AI image models (Flux, Z-Image, Qwen-Image).

## Task

Rewrite the user's idea into one vivid, comma-separated description. Keep their subject, names, and intent.

## Style

- Add at most a few cues for lighting, composition, color, texture, and mood.
- If the input is already detailed, lightly polish only — do not lengthen.

## Limits

- Length cap: ~120 CJK characters or ~80 English words.
- Never repeat the same word or phrase; never loop filler at the end.
- Do not write "Okay", explanations, or quotes.

## Output

Output **only** the enhanced prompt.`;

const ENHANCE_VIDEO_SYSTEM = `# Role

Professional prompt engineer for AI video generation.

## Task

Rewrite the user's brief into a detailed prompt for image-to-video or text-to-video models.

## Include

Subject, scene, lighting, style, camera movement, motion dynamics, pacing, and temporal mood.

For LTX audio-video models, hint ambient sound rhythm and dialogue pacing without writing looping lines.
Distinguish static scene description from continuing motion the camera can follow.

## Limits

- One paragraph; at most ~120 CJK characters or ~80 English words.
- Never repeat the same phrase or word. No filler loops.

## Output

Output **only** the enhanced prompt text, without explanation or quotation marks.`;

const ENHANCE_AUDIO_SYSTEM = `# Role

Music producer writing briefs for AI music generation (ACE-Step).

## Task

Expand the user's music idea into a clear, vivid description covering genre, mood, tempo feel, instrumentation, vocal style, and emotional arc.

## Limits

- One short paragraph.
- Never repeat the same phrase or word. No filler loops.

## Output

Output **only** the enhanced brief text, without explanation or quotation marks.`;

const LYRICS_SYSTEM = `# ACE-Step lyrics

Reply with **only** a lyric script. No title, planning, markdown fences, or text before/after the script.

Infer structure and language from the examples below. Match the music description language in the user message.

## Vocal · CJK shape

\`\`\`
[Verse 1]
<line 1>
<line 2>

[Chorus]
<hook line 1>
<hook line 2>

[Outro]
<closing line>
\`\`\`

## Vocal · English shape

\`\`\`
[Verse 1]
<line 1>
<line 2>

[Chorus]
<hook line 1>
<hook line 2>

[Outro]
<closing line>
\`\`\`

## Instrumental

\`\`\`
[Instrumental]
\`\`\`

## Counter-example (invalid — never resemble this)

\`\`\`
[Verse 1]
<title translation in parentheses>
Here is the chorus:
[Chorus]
<unrelated hook>
\`\`\``;

const IMAGE_TO_PROMPT_SYSTEM = `# Role

Expert AI art prompt engineer.

## Task

Analyze the attached image and write a detailed **English** prompt suitable for text-to-image models (Flux, SDXL, etc.).

## Include

Subject, composition, lighting, color palette, art style, mood, camera angle, and fine details.

## Output

Output **only** the prompt text — no quotes, headings, or explanation.`;

const VIDEO_FRAME_TO_PROMPT_SYSTEM = `# Role

Expert AI video prompt engineer.

## Task

The attached image is a keyframe or reference for video generation. Write a detailed **English** prompt describing the scene plus implied motion, camera movement, and temporal atmosphere suitable for image-to-video models.

## Output

Output **only** the prompt text — no quotes, headings, or explanation.`;

const CANVAS_DESCRIBE_SYSTEM = `# Role

Creative studio assistant writing short notes on canvas nodes.

## Task

Describe the attached visual asset in **2–4** concise sentences for an artist's canvas board note.

## Cover

Subject, style, lighting, composition, and one concrete next-step suggestion.

## Output

Output **only** the note text — no quotes or headings.`;

const CANVAS_DESCRIBE_TEXT_SYSTEM = `# Role

Creative studio assistant writing short notes on canvas nodes from asset metadata.

## Task

Write **2–4** concise sentences for an artist's canvas board note.

## Cover

Subject, style, and one concrete next-step suggestion.

## Output

Output **only** the note text.`;

const REFERENCE_VISION_SYSTEM = `# Role

Creative director analyzing a reference image for an artist.

## Task

Answer the user's question about the attached image. Be concise and practical.

## Output

Output **only** the answer — no preamble.`;

function visionUserLocaleBlock(locale?: string): string {
  const lang = (locale || '').startsWith('zh') ? 'zh' : 'en';
  if (lang === 'zh') {
    return '\n\n## Output language\nRespond in concise Simplified Chinese (简体中文).';
  }
  return '\n\n## Output language\nRespond in concise English.';
}


function enhanceUserLocaleHint(text: string, targetAction?: string): string {
  const action = (targetAction || 'image_create').trim().toLowerCase();
  if (!['image_create', 'create', 'image'].includes(action)) return '';
  const raw = (text || '').trim();
  if (raw.length < 60) return '';
  if (/[\u4e00-\u9fff]/.test(raw)) {
    return '\n\n（输入已够详细：只做轻微润色，禁止加长或重复用词。）';
  }
  if (raw.length >= 80) {
    return '\n\n(Input is already detailed: light polish only; do not lengthen or repeat phrases.)';
  }
  return '';
}

export function assetImageUrl(assetId: string, opts?: { thumbnail?: boolean }): string {
  const id = encodeURIComponent(assetId.trim());
  return `/api/assets/${id}/${opts?.thumbnail ? 'thumbnail' : 'file'}`;
}

function enhanceSystemPrompt(targetAction?: string): string {
  const action = (targetAction || 'image_create').trim().toLowerCase();
  if (['video', 'video_create', 'animate', 'video_generation'].includes(action)) {
    return ENHANCE_VIDEO_SYSTEM;
  }
  if (['audio', 'audio_create', 'music', 'audio_generation'].includes(action)) {
    return ENHANCE_AUDIO_SYSTEM;
  }
  return ENHANCE_IMAGE_SYSTEM;
}

function buildEnhanceUserContent(
  prompt: string,
  stylePositive?: string,
  targetAction?: string,
): string {
  const parts = ['## User brief', (prompt || '').trim()];
  parts.push(enhanceUserLocaleHint(prompt, targetAction));
  const style = (stylePositive || '').trim();
  if (style) {
    parts.push('', '## Style cues', style);
  }
  return parts.filter((p) => p !== undefined).join('\n');
}

export function buildEnhanceMessages(
  prompt: string,
  opts?: { stylePositive?: string; targetAction?: string },
): ChatMessagePayload[] {
  return [
    { role: 'system', content: enhanceSystemPrompt(opts?.targetAction) },
    {
      role: 'user',
      content: buildEnhanceUserContent(prompt, opts?.stylePositive, opts?.targetAction),
    },
  ];
}

export function buildLyricsMessages(prompt: string, style?: string): ChatMessagePayload[] {
  const parts = ['## Music description', (prompt || '').trim()];
  const styleText = (style || '').trim();
  if (styleText) {
    parts.push('', '## Style', styleText);
  }
  return [
    { role: 'system', content: LYRICS_SYSTEM },
    { role: 'user', content: parts.join('\n') },
  ];
}

export function buildImageToPromptMessages(
  assetId: string,
  opts?: { video?: boolean },
): ChatMessagePayload[] {
  return [
    { role: 'system', content: opts?.video ? VIDEO_FRAME_TO_PROMPT_SYSTEM : IMAGE_TO_PROMPT_SYSTEM },
    {
      role: 'user',
      content: [
        { type: 'text', text: 'Caption the attached image.' },
        { type: 'image_url', image_url: { url: assetImageUrl(assetId, { thumbnail: opts?.video }) } },
      ],
    },
  ];
}

export function buildReferenceAnalyzeMessages(
  assetId: string,
  question: string,
  opts?: { locale?: string; video?: boolean },
): ChatMessagePayload[] {
  const userText = `${question.trim()}${visionUserLocaleBlock(opts?.locale)}`;
  return [
    { role: 'system', content: REFERENCE_VISION_SYSTEM },
    {
      role: 'user',
      content: [
        { type: 'text', text: userText },
        {
          type: 'image_url',
          image_url: { url: assetImageUrl(assetId, { thumbnail: opts?.video }) },
        },
      ],
    },
  ];
}




export function buildCanvasDescribeMessages(
  assetId: string,
  opts?: { preferVision?: boolean },
): ChatMessagePayload[] {
  if (opts?.preferVision === false) {
    return [
      { role: 'system', content: CANVAS_DESCRIBE_TEXT_SYSTEM },
      {
        role: 'user',
        content: `## Asset\nCanvas node asset id: ${assetId}`,
      },
    ];
  }
  return [
    { role: 'system', content: CANVAS_DESCRIBE_SYSTEM },
    {
      role: 'user',
      content: [
        { type: 'text', text: 'Describe the attached canvas asset.' },
        { type: 'image_url', image_url: { url: assetImageUrl(assetId) } },
      ],
    },
  ];
}

export async function runChatCompletion(
  messages: ChatMessagePayload[],
  opts?: { temperature?: number; max_tokens?: number; model?: string },
): Promise<string> {
  const res = await api.gen.chatCompletion({
    messages,
    temperature: opts?.temperature,
    max_tokens: opts?.max_tokens,
    model: opts?.model,
  });
  return completionText(res);
}

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
  return runChatCompletion(buildLyricsMessages(prompt, style), {
    temperature: 0.65,
    max_tokens: 512,
  });
}

export async function imageToPromptViaChat(assetId: string, opts?: { video?: boolean }): Promise<string> {
  return runChatCompletion(buildImageToPromptMessages(assetId, opts), {
    temperature: 0.4,
    max_tokens: 384,
  });
}

export async function analyzeReferenceViaChat(
  assetId: string,
  question: string,
  opts?: { locale?: string; video?: boolean },
): Promise<string> {
  return runChatCompletion(buildReferenceAnalyzeMessages(assetId, question, opts), {
    temperature: 0.4,
    max_tokens: 384,
  });
}

export async function describeCanvasNodeViaChat(
  assetId: string,
  opts?: { preferVision?: boolean },
): Promise<{ note: string; visionUsed: boolean }> {
  const messages = buildCanvasDescribeMessages(assetId, opts);
  const visionUsed = opts?.preferVision !== false;
  const note = await runChatCompletion(messages, { temperature: 0.4, max_tokens: 256 });
  return { note, visionUsed };
}

