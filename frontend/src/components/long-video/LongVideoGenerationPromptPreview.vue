<template>
  <section class="lv-gen-prompt-preview">
    <div class="lv-gen-prompt-preview__head">
      <button type="button" class="lv-gen-prompt-preview__toggle" @click="open = !open">
        <DqIcon :size="14"><DocumentCopy /></DqIcon>
        <span>{{ $tt('video.longVideoGenerationPromptPreview') }}</span>
        <span class="lv-gen-prompt-preview__chevron" :class="{ 'is-open': open }" aria-hidden="true">▾</span>
      </button>
      <DqButton
        type="text"
        size="sm"
        :disabled="!preview.trim()"
        @click="copyPreview"
      >
        {{ $tt('gallery.copyPrompt') }}
      </DqButton>
    </div>
    <div v-show="open" class="lv-gen-prompt-preview__body">
      <p v-if="modeHint" class="lv-gen-prompt-preview__hint">{{ modeHint }}</p>
      <pre class="lv-gen-prompt-preview__code">{{ preview.trim() || $tt('video.longVideoGenerationPromptEmpty') }}</pre>
    </div>
  </section>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { useI18n } from 'vue-i18n';
import { DocumentCopy } from '@danqing/dq-shell';
import { toast } from '@/utils/feedback';

const props = defineProps<{
  preview: string;
  modeHint?: string;
}>();

const { t: $tt } = useI18n();
const open = ref(true);

async function copyPreview() {
  const text = props.preview.trim();
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    toast.success($tt('gallery.copied'));
  } catch {
    const ta = document.createElement('textarea');
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    toast.success($tt('gallery.copied'));
  }
}
</script>

<style scoped>
.lv-gen-prompt-preview {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 10px 12px;
  border-radius: 10px;
  border: 0.5px solid var(--dq-glass-border, var(--dq-border-subtle));
  background: color-mix(in srgb, var(--dq-surface-elevated) 72%, transparent);
}

.lv-gen-prompt-preview__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.lv-gen-prompt-preview__toggle {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border: none;
  background: none;
  padding: 0;
  font-size: var(--dq-font-size-caption);
  font-weight: 600;
  color: var(--dq-label-secondary);
  cursor: pointer;
}

.lv-gen-prompt-preview__toggle:hover {
  color: var(--dq-accent);
}

.lv-gen-prompt-preview__chevron {
  font-size: var(--dq-font-size-caption);
  line-height: 1;
  transition: transform 0.2s ease;
  color: var(--dq-label-tertiary);
}

.lv-gen-prompt-preview__chevron.is-open {
  transform: rotate(180deg);
}

.lv-gen-prompt-preview__hint {
  margin: 0;
  font-size: var(--dq-font-size-caption);
  line-height: 1.45;
  color: var(--dq-label-tertiary);
}

.lv-gen-prompt-preview__code {
  margin: 0;
  max-height: 280px;
  overflow: auto;
  padding: 10px 12px;
  border-radius: 8px;
  background: var(--dq-bg-base);
  font-family: var(--dq-font-mono);
  font-size: var(--dq-font-size-caption);
  line-height: 1.55;
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--dq-label-primary);
}
</style>
