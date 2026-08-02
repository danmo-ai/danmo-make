<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue';
import type { ThemeId } from '@/stores/theme';
import { THEME_OPTIONS } from '@/stores/theme';
import { $mn } from '@/utils/i18n';
import { canvasAutoAddEnabled, setCanvasAutoAdd } from '@/composables/useCanvasStore';
import QuickSetupPanel from '@/components/settings/QuickSetupPanel.vue';
import { useRegistryStore } from '@/stores/registry';
import { api } from '@/utils/api';
import { DQ_STORAGE, removeItem, setItem } from '@/utils/storage';
import { toast, confirm } from '@/utils/feedback';
import { useI18n } from 'vue-i18n';

const quickSetupRef = ref<InstanceType<typeof QuickSetupPanel> | null>(null);

const canvasAutoAddImage = ref(canvasAutoAddEnabled('image'));
const canvasAutoAddVideo = ref(canvasAutoAddEnabled('video'));
const canvasAutoAddAudio = ref(canvasAutoAddEnabled('audio'));

function onCanvasAutoAddImageChange(enabled: boolean) {
  canvasAutoAddImage.value = enabled;
  setCanvasAutoAdd(enabled, 'image');
}

function onCanvasAutoAddVideoChange(enabled: boolean) {
  canvasAutoAddVideo.value = enabled;
  setCanvasAutoAdd(enabled, 'video');
}

function onCanvasAutoAddAudioChange(enabled: boolean) {
  canvasAutoAddAudio.value = enabled;
  setCanvasAutoAdd(enabled, 'audio');
}

type SectionId =
  | 'general'
  | 'performance'
  | 'studio'
  | 'quicksetup'
  | 'workspace'
  | 'integrations'
  | 'runtime'
  | 'maintenance'
  | 'systeminfo';

const props = defineProps<{
  activeSection: SectionId;
  settings: Record<string, unknown>;
  workspacePaths: Record<string, string> | null;
  restoreConfigBusy: boolean;
}>();

const themeOptions = THEME_OPTIONS.map((opt) => ({
  label: opt.label,
  value: opt.id,
}));

const emit = defineEmits<{
  save: [];
  languageChange: [lang: string];
  themeChange: [theme: ThemeId];
  pickWorkspace: [];
  restoreModelRegistry: [];
}>();

function onQuickSetupPatch(patch: Record<string, unknown>) {
  Object.assign(props.settings, patch);
}

function applyQuickSetupDefaults() {
  void quickSetupRef.value?.applyRecommendedDefaults();
}

const registryStore = useRegistryStore();
const { t } = useI18n();
const accessKeyBusy = ref<'http' | 'mcp' | null>(null);
const revealOpen = ref(false);
const revealKind = ref<'http' | 'mcp'>('http');
const revealKey = ref('');
const saveHttpKeyInBrowser = ref(true);

async function createAccessKey(kind: 'http' | 'mcp') {
  if (accessKeyBusy.value) return;
  const configured =
    kind === 'http'
      ? !!props.settings.http_api_key_configured
      : !!props.settings.mcp_api_key_configured;
  if (configured) {
    try {
      await confirm(t('settings.accessKeyRotateConfirm'), t('settings.accessKeyRotateConfirmTitle'), {
        type: 'warning',
        confirmButtonText: t('settings.accessKeyCreate'),
        cancelButtonText: t('common.cancel'),
      });
    } catch {
      return;
    }
  }
  accessKeyBusy.value = kind;
  try {
    const res = await api.settings.createAccessKey(kind);
    if (kind === 'http') {
      props.settings.http_api_key_configured = true;
      props.settings.http_api_key_hint = res.hint;
      props.settings.http_api_key_from_env = false;
    } else {
      props.settings.mcp_api_key_configured = true;
      props.settings.mcp_api_key_hint = res.hint;
      props.settings.mcp_api_key_from_env = false;
    }
    revealKind.value = kind;
    revealKey.value = res.key;
    saveHttpKeyInBrowser.value = kind === 'http';
    revealOpen.value = true;
  } catch (e) {
    console.error(e);
    toast.error(t('settings.accessKeyCreateFailed'));
  } finally {
    accessKeyBusy.value = null;
  }
}

async function revokeAccessKey(kind: 'http' | 'mcp') {
  if (accessKeyBusy.value) return;
  try {
    await confirm(t('settings.accessKeyRevokeConfirm'), t('settings.accessKeyRevokeConfirmTitle'), {
      type: 'warning',
      confirmButtonText: t('settings.accessKeyRevoke'),
      cancelButtonText: t('common.cancel'),
    });
  } catch {
    return;
  }
  accessKeyBusy.value = kind;
  try {
    await api.settings.revokeAccessKey(kind);
    if (kind === 'http') {
      props.settings.http_api_key_configured = false;
      props.settings.http_api_key_hint = '';
      removeItem(DQ_STORAGE.HTTP_API_KEY);
    } else {
      props.settings.mcp_api_key_configured = false;
      props.settings.mcp_api_key_hint = '';
    }
    toast.success(t('settings.accessKeyRevoked'));
  } catch (e) {
    console.error(e);
    toast.error(t('settings.accessKeyRevokeFailed'));
  } finally {
    accessKeyBusy.value = null;
  }
}

async function copyRevealedKey() {
  const key = revealKey.value.trim();
  if (!key) return;
  try {
    await navigator.clipboard.writeText(key);
    toast.success(t('settings.accessKeyCopied'));
  } catch {
    toast.error(t('settings.accessKeyCopyFailed'));
  }
}

function closeRevealDialog() {
  const key = revealKey.value.trim();
  if (key && revealKind.value === 'http' && saveHttpKeyInBrowser.value) {
    setItem(DQ_STORAGE.HTTP_API_KEY, key);
  }
  revealKey.value = '';
  revealOpen.value = false;
}

onMounted(() => {
  void registryStore.load();
});

function hasLlmChatAction(actions: unknown): boolean {
  if (!actions || typeof actions !== 'object') return false;
  const row = actions as Record<string, unknown>;
  return row.chat != null || row.enhance != null;
}

function hasVlmDescribeAction(actions: unknown): boolean {
  if (!actions || typeof actions !== 'object') return false;
  return (actions as Record<string, unknown>).describe != null;
}

function isLlmCategory(category: unknown): boolean {
  return category === 'llm_models';
}

function isVlmCategory(category: unknown): boolean {
  return category === 'vlm_models';
}

function llmSupportsThink(modelId: unknown): boolean {
  return /thinking/i.test(String(modelId || '').trim());
}

const llmThinkSupported = computed(() => llmSupportsThink(props.settings.default_model_llm));

watch(
  () => props.settings.default_model_llm,
  (modelId) => {
    if (!llmSupportsThink(modelId)) {
      props.settings.default_model_llm_think = false;
    }
  },
);

const llmModelOptions = computed(() => {
  const models = registryStore.registry?.models || {};
  return Object.entries(models)
    .filter(([, cfg]) => cfg.media === 'llm' && isLlmCategory(cfg.category) && hasLlmChatAction(cfg.actions))
    .map(([id, cfg]) => ({
      value: id,
      label: $mn(cfg, id),
    }))
    .sort((a, b) => a.label.localeCompare(b.label));
});

const vlmModelOptions = computed(() => {
  const models = registryStore.registry?.models || {};
  return Object.entries(models)
    .filter(([, cfg]) => cfg.media === 'llm' && isVlmCategory(cfg.category) && hasVlmDescribeAction(cfg.actions))
    .map(([id, cfg]) => ({
      value: id,
      label: $mn(cfg, id),
    }))
    .sort((a, b) => a.label.localeCompare(b.label));
});
</script>

<template>
  <div class="settings-system-form">
    <!-- General -->
    <template v-if="props.activeSection === 'general'">
      <section class="settings-group-block">
        <h2 class="settings-section-title">{{ $t('settings.general') }}</h2>
        <p class="settings-section-desc">{{ $t('settings.generalDesc') }}</p>
        <DqPrefPane class="settings-grouped-form settings-pref-pane-form settings-pref-pane-form--system">
          <DqPrefRow :label="$t('settings.language')">
            <DqSelect
              v-model="settings.language"
              class="settings-mac-value-control"
              @change="emit('languageChange', $event as string)"
            >
              <DqOption :label="$t('settings.label_zh')" value="zh" />
              <DqOption :label="$t('settings.label_en')" value="en" />
            </DqSelect>
          </DqPrefRow>

          <DqPrefRow :label="$t('settings.theme')">
            <DqSelect
              v-model="settings.theme"
              class="settings-mac-value-control"
              @change="emit('themeChange', $event as ThemeId)"
            >
              <DqOption
                v-for="opt in themeOptions"
                :key="opt.value"
                :label="opt.label"
                :value="opt.value"
              />
            </DqSelect>
          </DqPrefRow>

          <DqPrefRow :label="$t('settings.outputFormat')">
            <DqSelect v-model="settings.output_format" class="settings-mac-value-control">
              <DqOption label="PNG" value="png" />
              <DqOption label="JPEG" value="jpg" />
              <DqOption label="WebP" value="webp" />
            </DqSelect>
          </DqPrefRow>

          <DqPrefRow :label="$t('settings.defaultLlmModel')">
            <DqSelect
              v-model="settings.default_model_llm"
              class="settings-mac-value-control"
              :placeholder="$t('settings.defaultLlmModelPlaceholder')"
            >
              <DqOption
                v-for="opt in llmModelOptions"
                :key="opt.value"
                :label="opt.label"
                :value="opt.value"
              />
            </DqSelect>
          </DqPrefRow>

          <DqPrefRow
            v-if="llmThinkSupported"
            :label="$t('settings.defaultLlmThink')"
            stacked
          >
            <div class="settings-stacked-control">
              <DqSwitch v-model="settings.default_model_llm_think" />
              <p class="settings-form-hint settings-form-hint--below-control">
                {{ $t('settings.defaultLlmThinkDesc') }}
              </p>
            </div>
          </DqPrefRow>

          <DqPrefRow :label="$t('settings.defaultVlmModel')">
            <DqSelect
              v-model="settings.default_model_vlm"
              class="settings-mac-value-control"
              :placeholder="$t('settings.defaultVlmModelPlaceholder')"
            >
              <DqOption
                v-for="opt in vlmModelOptions"
                :key="opt.value"
                :label="opt.label"
                :value="opt.value"
              />
            </DqSelect>
          </DqPrefRow>
        </DqPrefPane>
      </section>
    </template>

    <!-- Performance -->
    <template v-if="props.activeSection === 'performance'">
      <section class="settings-group-block">
        <h2 class="settings-section-title">{{ $t('settings.performance') }}</h2>
        <p class="settings-section-desc">{{ $t('settings.performanceDesc') }}</p>
        <DqPrefPane class="settings-grouped-form settings-pref-pane-form settings-pref-pane-form--system">
          <DqPrefRow :label="$t('settings.memoryLimit')">
            <div class="param-control-row settings-pref-slider-row">
              <div class="param-slider">
                <DqSlider v-model="settings.mlx_memory_limit" :min="32" :max="256" :step="8" />
              </div>
              <span class="settings-slider-suffix">{{ settings.mlx_memory_limit }} GB</span>
            </div>
          </DqPrefRow>

          <DqPrefRow :label="$t('settings.modelCacheTtl')" stacked>
            <div class="settings-stacked-control">
              <div class="param-control-row settings-pref-slider-row">
                <div class="param-slider">
                  <DqSlider v-model="settings.model_cache_ttl_minutes" :min="5" :max="120" :step="5" />
                </div>
                <span class="settings-slider-suffix">{{ settings.model_cache_ttl_minutes }} min</span>
              </div>
              <p class="settings-form-hint settings-form-hint--below-control">
                {{ $t('settings.modelCacheTtlDesc') }}
              </p>
            </div>
          </DqPrefRow>

          <DqPrefRow :label="$t('settings.queueImageFirst')" stacked>
            <div class="settings-stacked-control">
              <DqSwitch v-model="settings.queue_image_first" />
              <p class="settings-form-hint settings-form-hint--below-control">
                {{ $t('settings.queueImageFirstDesc') }}
              </p>
            </div>
          </DqPrefRow>

          <DqPrefRow :label="$t('settings.autoSavePrompts')" stacked>
            <div class="settings-stacked-control">
              <DqSwitch v-model="settings.auto_save_prompts" />
              <p class="settings-form-hint settings-form-hint--below-control">
                {{ $t('settings.autoSavePromptsDesc') }}
              </p>
            </div>
          </DqPrefRow>
        </DqPrefPane>
      </section>
    </template>

    <!-- Studio / Canvas -->
    <template v-if="props.activeSection === 'studio'">
      <section class="settings-group-block">
        <h2 class="settings-section-title">{{ $t('settings.studio') }}</h2>
        <p class="settings-section-desc">{{ $t('settings.studioDesc') }}</p>
        <DqPrefPane class="settings-grouped-form settings-pref-pane-form settings-pref-pane-form--system">
          <DqPrefRow :label="$t('settings.canvasAutoAddImage')" stacked>
            <div class="settings-stacked-control">
              <DqSwitch
                :model-value="canvasAutoAddImage"
                @update:model-value="onCanvasAutoAddImageChange"
              />
              <p class="settings-form-hint settings-form-hint--below-control">
                {{ $t('settings.canvasAutoAddImageDesc') }}
              </p>
            </div>
          </DqPrefRow>

          <DqPrefRow :label="$t('settings.canvasAutoAddVideo')" stacked>
            <div class="settings-stacked-control">
              <DqSwitch
                :model-value="canvasAutoAddVideo"
                @update:model-value="onCanvasAutoAddVideoChange"
              />
              <p class="settings-form-hint settings-form-hint--below-control">
                {{ $t('settings.canvasAutoAddVideoDesc') }}
              </p>
            </div>
          </DqPrefRow>

          <DqPrefRow :label="$t('settings.canvasAutoAddAudio')" stacked>
            <div class="settings-stacked-control">
              <DqSwitch
                :model-value="canvasAutoAddAudio"
                @update:model-value="onCanvasAutoAddAudioChange"
              />
              <p class="settings-form-hint settings-form-hint--below-control">
                {{ $t('settings.canvasAutoAddAudioDesc') }}
              </p>
            </div>
          </DqPrefRow>
        </DqPrefPane>
      </section>
    </template>

    <template v-if="props.activeSection === 'quicksetup'">
      <QuickSetupPanel ref="quickSetupRef" @patch-settings="onQuickSetupPatch" />
    </template>

    <!-- Workspace -->
    <template v-if="props.activeSection === 'workspace'">
      <section class="settings-group-block">
        <h2 class="settings-section-title">{{ $t('settings.workspace') }}</h2>
        <p class="settings-section-desc">{{ $t('settings.workspaceDesc') }}</p>
        <DqPrefPane class="settings-grouped-form settings-pref-pane-form settings-pref-pane-form--system">
          <DqPrefRow :label="$t('settings.customWorkspace')" stacked>
            <div class="settings-stacked-control settings-workspace-picker">
              <div class="settings-workspace-input-row">
                <DqInput
                  v-model="settings.custom_workspace_dir"
                  :placeholder="$t('settings.customWorkspacePlaceholder')"
                />
                <DqButton size="sm" class="settings-workspace-pick-btn" @click="emit('pickWorkspace')">
                  {{ $t('settings.pickWorkspace') }}
                </DqButton>
              </div>
              <p class="settings-form-hint settings-form-hint--below-control">
                {{ $t('settings.workspaceSetupEmptyHint') }}
              </p>
              <p class="settings-form-hint settings-form-hint--below-control">
                {{ $t('settings.customWorkspaceHint') }}
              </p>
              <p class="settings-form-hint settings-form-hint--below-control">
                {{ $t('settings.customWorkspaceRestartHint') }}
              </p>
              <div v-if="workspacePaths" class="settings-workspace-paths">
                <div class="settings-workspace-paths-title">{{ $t('settings.workspaceLayoutTitle') }}</div>
                <ul class="settings-workspace-paths-list">
                  <li v-for="(p, key) in workspacePaths" :key="key">
                    <span class="settings-workspace-paths-key">{{ key }}</span>
                    <span class="settings-workspace-paths-val">{{ p }}</span>
                  </li>
                </ul>
              </div>
            </div>
          </DqPrefRow>
        </DqPrefPane>
      </section>
    </template>

    <!-- Integrations -->
    <template v-if="props.activeSection === 'integrations'">
      <section class="settings-group-block">
        <h2 class="settings-section-title">{{ $t('settings.integrations') }}</h2>
        <p class="settings-section-desc">{{ $t('settings.integrationsDesc') }}</p>
        <DqPrefPane class="settings-grouped-form settings-pref-pane-form settings-pref-pane-form--system">
          <DqPrefRow :label="$t('settings.huggingfaceToken')" stacked>
            <div class="settings-stacked-control">
              <DqInput
                v-model="settings.huggingface_token"
                type="password"
                show-password
                :placeholder="$t('settings.huggingfaceTokenPlaceholder')"
              >
                <template #prefix>
                  <DqIcon><document /></DqIcon>
                </template>
              </DqInput>
              <p class="settings-form-hint settings-form-hint--below-control">
                {{ $t('settings.huggingfaceTokenDesc') }}
              </p>
            </div>
          </DqPrefRow>

          <DqPrefRow :label="$t('settings.civitaiToken')" stacked>
            <div class="settings-stacked-control">
              <DqInput
                v-model="settings.civitai_token"
                type="password"
                show-password
                :placeholder="$t('settings.civitaiTokenPlaceholder')"
              >
                <template #prefix>
                  <DqIcon><document /></DqIcon>
                </template>
              </DqInput>
              <p class="settings-form-hint settings-form-hint--below-control">
                {{ $t('settings.civitaiTokenDesc') }}
              </p>
            </div>
          </DqPrefRow>

          <DqPrefRow v-if="settings.civitai_token" no-label>
            <div class="settings-stacked-control">
              <DqCheckbox v-model="settings.nsfw_enabled" size="large">
                <DqText type="danger">{{ $t('settings.nsfwContent') }}</DqText>
              </DqCheckbox>
              <p class="settings-form-hint settings-form-hint--below-control">
                {{ $t('settings.nsfwDesc') }}
              </p>
            </div>
          </DqPrefRow>

          <DqPrefRow :label="$t('settings.httpApiKey')" stacked>
            <div class="settings-stacked-control">
              <p class="settings-access-key-status">
                <template v-if="settings.http_api_key_from_env">
                  {{ $t('settings.accessKeyFromEnv') }}
                </template>
                <template v-else-if="settings.http_api_key_configured">
                  {{ $t('settings.accessKeyHint', { hint: settings.http_api_key_hint || '••••' }) }}
                </template>
                <template v-else>
                  {{ $t('settings.accessKeyNotConfigured') }}
                </template>
              </p>
              <div class="settings-inline-actions">
                <DqButton size="sm" :disabled="!!accessKeyBusy" @click="createAccessKey('http')">
                  {{
                    settings.http_api_key_configured
                      ? $t('settings.accessKeyRotate')
                      : $t('settings.accessKeyCreate')
                  }}
                </DqButton>
                <DqButton
                  v-if="settings.http_api_key_configured && !settings.http_api_key_from_env"
                  size="sm"
                  variant="ghost"
                  :disabled="!!accessKeyBusy"
                  @click="revokeAccessKey('http')"
                >
                  {{ $t('settings.accessKeyRevoke') }}
                </DqButton>
              </div>
              <p class="settings-form-hint settings-form-hint--below-control">
                {{ $t('settings.httpApiKeyDesc') }}
              </p>
            </div>
          </DqPrefRow>

          <DqPrefRow :label="$t('settings.mcpApiKey')" stacked>
            <div class="settings-stacked-control">
              <p class="settings-access-key-status">
                <template v-if="settings.mcp_api_key_from_env">
                  {{ $t('settings.accessKeyFromEnv') }}
                </template>
                <template v-else-if="settings.mcp_api_key_configured">
                  {{ $t('settings.accessKeyHint', { hint: settings.mcp_api_key_hint || '••••' }) }}
                </template>
                <template v-else>
                  {{ $t('settings.accessKeyNotConfigured') }}
                </template>
              </p>
              <div class="settings-inline-actions">
                <DqButton size="sm" :disabled="!!accessKeyBusy" @click="createAccessKey('mcp')">
                  {{
                    settings.mcp_api_key_configured
                      ? $t('settings.accessKeyRotate')
                      : $t('settings.accessKeyCreate')
                  }}
                </DqButton>
                <DqButton
                  v-if="settings.mcp_api_key_configured && !settings.mcp_api_key_from_env"
                  size="sm"
                  variant="ghost"
                  :disabled="!!accessKeyBusy"
                  @click="revokeAccessKey('mcp')"
                >
                  {{ $t('settings.accessKeyRevoke') }}
                </DqButton>
              </div>
              <p class="settings-form-hint settings-form-hint--below-control">
                {{ $t('settings.mcpApiKeyDesc') }}
              </p>
            </div>
          </DqPrefRow>
        </DqPrefPane>
      </section>

      <DqDialog
        v-model:open="revealOpen"
        :title="$t('settings.accessKeyRevealTitle')"
        width="min(520px, 96vw)"
        :close-on-click-overlay="false"
        @update:open="(v: boolean) => { if (!v) closeRevealDialog(); }"
      >
        <p class="settings-form-hint">{{ $t('settings.accessKeyRevealOnce') }}</p>
        <DqInput
          :model-value="revealKey"
          type="textarea"
          :rows="3"
          readonly
          class="settings-access-key-reveal"
        />
        <DqCheckbox
          v-if="revealKind === 'http'"
          v-model="saveHttpKeyInBrowser"
          class="settings-access-key-browser"
        >
          {{ $t('settings.accessKeySaveInBrowser') }}
        </DqCheckbox>
        <template #footer>
          <DqButton @click="copyRevealedKey">{{ $t('settings.accessKeyCopy') }}</DqButton>
          <DqButton type="primary" @click="closeRevealDialog">{{ $t('settings.accessKeyDone') }}</DqButton>
        </template>
      </DqDialog>
    </template>

    <!-- Maintenance -->
    <template v-if="props.activeSection === 'maintenance'">
      <section class="settings-group-block">
        <h2 class="settings-section-title">{{ $t('settings.maintenance') }}</h2>
        <p class="settings-section-desc">{{ $t('settings.maintenanceDesc') }}</p>
        <div
          class="settings-grouped-form settings-grouped-form--action-list"
          role="group"
          :aria-label="$t('settings.maintenance')"
        >
          <button
            type="button"
            class="settings-action-row settings-action-row--destructive"
            :disabled="restoreConfigBusy"
            @click="emit('restoreModelRegistry')"
          >
            <span class="settings-action-row__label">{{ $t('settings.restoreModelRegistry') }}</span>
            <DqIcon class="settings-action-row__chevron"><arrow-right /></DqIcon>
          </button>
        </div>
        <p class="settings-group-footnote">{{ $t('settings.restoreModelRegistryDesc') }}</p>
      </section>
    </template>

    <!-- Save row -->
    <div class="settings-system-save-row" v-if="props.activeSection === 'quicksetup'">
      <DqButton
        type="primary"
        class="settings-system-save-btn"
        :loading="quickSetupRef?.applyingDefaults"
        @click="applyQuickSetupDefaults"
      >
        <DqIcon><check /></DqIcon>
        {{ $t('settings.quickSetupApplyDefaults') }}
      </DqButton>
    </div>
    <div class="settings-system-save-row" v-else-if="props.activeSection !== 'systeminfo' && props.activeSection !== 'maintenance'">
      <DqButton type="primary" class="settings-system-save-btn" @click="emit('save')">
        <DqIcon><check /></DqIcon>
        {{ $t('common.save') }}
      </DqButton>
    </div>
  </div>
</template>
