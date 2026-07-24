<script setup lang="ts">
import { computed, ref } from 'vue';
import { $tt } from '@/utils/i18n';
import PresetTemplateDialog from '@/components/settings/PresetTemplateDialog.vue';
import PromptTemplatesCategoryNav from '@/components/settings/PromptTemplatesCategoryNav.vue';

const props = defineProps<{
  presets: Record<string, Record<string, unknown>>;
  restoreConfigBusy: boolean;
  editingPresetName: string;
  presetForm: {
    name: string;
    positive: string;
    negative: string;
    trigger_words: string;
    media_scope: string;
    applies_to: string[];
  };
  activeCategory: string;
  activeAction: string;
  searchQuery: string;
}>();

const dialogOpen = defineModel<boolean>('dialogOpen', { required: true });

const emit = defineEmits<{
  add: [];
  edit: [name: string, preset: Record<string, unknown>];
  delete: [name: string];
  save: [];
  restore: [];
  'update:activeCategory': [category: string];
  'update:activeAction': [action: string];
  'update:searchQuery': [query: string];
}>();

const totalCount = computed(() => Object.keys(props.presets).length);

/* ───── Expand / collapse ───── */
const expandedCards = ref<Set<string>>(new Set());

function isExpanded(name: string): boolean {
  return expandedCards.value.has(name);
}

function toggleExpand(name: string) {
  if (expandedCards.value.has(name)) {
    expandedCards.value.delete(name);
  } else {
    expandedCards.value.add(name);
  }
}

const presetList = computed(() => {
  let list = Object.entries(props.presets).map(([name, preset]) => ({ name, preset }));

  // Filter by media scope
  if (props.activeCategory !== 'all') {
    list = list.filter((row) => row.preset.media_scope === props.activeCategory);
  }

  // Filter by action
  if (props.activeAction) {
    list = list.filter((row) => {
      const actions = row.preset.applies_to as string[] || [];
      return actions.includes(props.activeAction);
    });
  }

  // Search
  if (props.searchQuery.trim()) {
    const query = props.searchQuery.toLowerCase();
    list = list.filter((row) => {
      const nameMatch = row.name.toLowerCase().includes(query);
      const positiveMatch = String(row.preset.positive || '').toLowerCase().includes(query);
      const negativeMatch = String(row.preset.negative || '').toLowerCase().includes(query);
      return nameMatch || positiveMatch || negativeMatch;
    });
  }

  return list;
});

const presetAppliesSummary = (preset: Record<string, unknown>) => {
  const actions = preset.applies_to as string[] || [];
  return actions.map((a) => $tt(`action.image.${a}`) || $tt(`action.video.${a}`) || a);
};

const presetMediaLabel = (preset: Record<string, unknown>) =>
  preset.media_scope === 'video' ? $tt('settings.presetMediaVideo') : $tt('settings.presetMediaImage');

const presetMediaTagType = (preset: Record<string, unknown>) =>
  preset.media_scope === 'video' ? 'primary' : 'success';

function handleCategorySelect(category: string) {
  emit('update:activeCategory', category);
  emit('update:activeAction', '');
}

function handleActionSelect(action: string) {
  emit('update:activeAction', action);
  if (props.activeCategory === 'all') {
    // Keep all, just filter by action
  }
}
</script>

<template>
  <div class="templates-page">
    <!-- Left sidebar -->
    <div class="templates-page__sidebar">
      <DqSurfaceCard class="templates-page__sidebar-card">
        <div class="card-title">
          <DqIcon><collection /></DqIcon>
          {{ $t('settings.promptTemplates') }}
        </div>
        <div class="templates-page__sidebar-intro">
          {{ $t('settings.promptTemplatesDesc') }}
        </div>

        <PromptTemplatesCategoryNav
          :active-category="activeCategory"
          :active-action="activeAction"
          :total-count="totalCount"
          @select-category="handleCategorySelect"
          @select-action="handleActionSelect"
        />
      </DqSurfaceCard>
    </div>

    <!-- Right main area -->
    <div class="templates-page__main">
      <!-- Header -->
      <div class="templates-page__header">
        <div class="templates-page__header-title">
          <DqIcon v-if="activeAction" class="templates-page__header-icon"><MagicStick /></DqIcon>
          <span>{{ activeAction ? $tt(`action.image.${activeAction}`) || $tt(`action.video.${activeAction}`) : (activeCategory === 'all' ? $t('settings.allTemplates') : (activeCategory === 'video' ? $t('settings.presetMediaVideo') : $t('settings.presetMediaImage'))) }}</span>
          <DqTag v-if="presetList.length > 0" size="sm" type="info" class="templates-page__count-tag">{{ presetList.length }}</DqTag>
        </div>

        <div class="templates-page__header-center">
          <DqInput
            :model-value="searchQuery"
            :placeholder="$t('settings.searchTemplate')"
            class="templates-page__search-input"
            clearable
            @update:model-value="emit('update:searchQuery', $event)"
          >
            <template #prefix>
              <DqIcon><search /></DqIcon>
            </template>
          </DqInput>
        </div>

        <div class="templates-page__header-actions">
          <DqButton
            type="text"
            size="sm"
            class="settings-link-destructive"
            :loading="restoreConfigBusy"
            @click="emit('restore')"
          >
            {{ $t('settings.restorePromptTemplates') }}
          </DqButton>
          <DqButton type="primary" @click="emit('add')">
            <DqIcon><plus /></DqIcon>
            {{ $t('settings.addTemplate') }}
          </DqButton>
        </div>
      </div>

      <!-- Empty state -->
      <DqInspectorEmpty v-if="presetList.length === 0" class="templates-page__empty">
        {{ searchQuery ? $t('common.noResults') : $t('settings.noTemplates') }}
      </DqInspectorEmpty>

      <!-- Template list -->
      <div v-else class="templates-list">
        <div
          v-for="row in presetList"
          :key="row.name"
          class="template-card"
          :class="{ 'is-expanded': isExpanded(row.name) }"
        >
          <!-- Card header: name + tags + actions -->
          <div class="template-card__header">
            <div class="template-card__title-row">
              <span class="template-card__name">{{ row.name }}</span>
              <div class="template-card__tags">
                <DqTag :type="presetMediaTagType(row.preset)" size="sm" effect="plain">
                  {{ presetMediaLabel(row.preset) }}
                </DqTag>
                <DqTag
                  v-for="action in presetAppliesSummary(row.preset)"
                  :key="action"
                  size="sm"
                  type="info"
                  effect="plain"
                >
                  {{ action }}
                </DqTag>
              </div>
            </div>

            <div class="template-card__actions">
              <DqIconButton
                type="text"
                size="sm"
                :label="$t('settings.editTemplate')"
                @click="emit('edit', row.name, row.preset)"
              >
                <DqIcon><edit /></DqIcon>
              </DqIconButton>
              <DqIconButton
                type="danger"
                size="sm"
                :label="$t('common.delete')"
                @click="emit('delete', row.name)"
              >
                <DqIcon><delete /></DqIcon>
              </DqIconButton>
            </div>
          </div>

          <!-- Card body: prompts -->
          <div class="template-card__body" @click="toggleExpand(row.name)">
            <div class="template-card__prompt-section">
              <div class="template-card__prompt-label">{{ $t('settings.positivePrompt') }}</div>
              <div
                class="template-card__prompt-text template-card__prompt-text--positive"
                :class="{ 'is-truncated': !isExpanded(row.name) }"
              >
                {{ row.preset.positive || '-' }}
              </div>
            </div>

            <div v-if="row.preset.negative" class="template-card__prompt-section">
              <div class="template-card__prompt-label">{{ $t('settings.negativePrompt') }}</div>
              <div
                class="template-card__prompt-text template-card__prompt-text--negative"
                :class="{ 'is-truncated': !isExpanded(row.name) }"
              >
                {{ row.preset.negative }}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <PresetTemplateDialog
      v-model:open="dialogOpen"
      :editing-name="editingPresetName"
      :preset-form="presetForm"
      @save="emit('save')"
      @cancel="dialogOpen = false"
    />
  </div>
</template>
