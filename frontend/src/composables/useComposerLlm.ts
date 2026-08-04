import { ref } from 'vue';
import {
  enhancePromptViaChat,
  generateLyricsViaChat,
  imageToPromptViaChat,
} from '@/utils/llmMessages';
import { $tt } from '@/utils/i18n';
import { toast } from '@/utils/feedback';

/** In-composer LLM actions — primary battlefield; copilot is for vision + batch. */
export function useComposerLlm() {
  const isEnhancing = ref(false);
  const isReversing = ref(false);
  const isGeneratingLyrics = ref(false);

  async function enhance(
    prompt: string,
    stylePositive?: string,
    targetAction?: string,
    _modelId?: string,
    options?: { quietSuccess?: boolean },
  ): Promise<string | null> {
    isEnhancing.value = true;
    try {
      const text = await enhancePromptViaChat(prompt, { stylePositive, targetAction });
      if (!options?.quietSuccess) {
        toast.success($tt('create.enhanceComplete'));
      }
      return text;
    } catch (e) {
      const err = e as { code?: string; message?: string; response?: { data?: { detail?: string } } };
      const msg = err.response?.data?.detail
        || (err.code === 'ECONNABORTED' ? $tt('create.enhanceTimeout') : '')
        || err.message
        || String(e);
      toast.error($tt('create.enhanceFailed', { msg }));
      return null;
    } finally {
      isEnhancing.value = false;
    }
  }

  async function reversePrompt(assetId: string, options?: { quietSuccess?: boolean }): Promise<string | null> {
    isReversing.value = true;
    try {
      const text = await imageToPromptViaChat(assetId);
      if (!options?.quietSuccess) {
        toast.success($tt('create.reverseComplete'));
      }
      return text;
    } catch (e) {
      const msg = (e as { response?: { data?: { detail?: string } }; message?: string })?.response?.data?.detail
        || (e as Error).message
        || String(e);
      toast.error($tt('create.reverseFailed', { msg }));
      return null;
    } finally {
      isReversing.value = false;
    }
  }

  async function generateLyrics(prompt: string, options?: { quietSuccess?: boolean }): Promise<string | null> {
    isGeneratingLyrics.value = true;
    try {
      const lyrics = await generateLyricsViaChat(prompt);
      if (!lyrics) {
        toast.error($tt('audio.lyricsGenFailed', { msg: $tt('audio.lyricsGenEmpty') }));
        return null;
      }
      if (!options?.quietSuccess) {
        toast.success($tt('audio.lyricsGenerated'));
      }
      return lyrics;
    } catch (e) {
      const msg = (e as { response?: { data?: { detail?: string } }; message?: string })?.response?.data?.detail
        || (e as Error).message
        || String(e);
      toast.error($tt('audio.lyricsGenFailed', { msg }));
      return null;
    } finally {
      isGeneratingLyrics.value = false;
    }
  }

  return {
    isEnhancing,
    isReversing,
    isGeneratingLyrics,
    enhance,
    reversePrompt,
    generateLyrics,
  };
}
