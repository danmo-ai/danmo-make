<script setup lang="ts">
defineProps<{
  editingName: string;
  presetForm: {
    name: string;
    positive: string;
    negative: string;
    trigger_words: string;
    media_scope: string;
    applies_to: string[];
  };
}>();

const open = defineModel<boolean>('open', { required: true });

const emit = defineEmits<{
  save: [];
  cancel: [];
}>();
</script>

<template>
  <DqDialog
    v-model:open="open"
    class="preset-template-dialog"
    :title="editingName ? $t('settings.editTemplate') : $t('settings.addTemplate')"
    width="min(720px, 96vw)"
    @update:open="(v: boolean) => { if (!v) emit('cancel'); }"
  >
    <DqPrefPane class="settings-pref-pane-form settings-pref-pane-form--dialog settings-pref-pane-form--preset-template">
      <DqPrefRow :label="$t('settings.templateName')" stacked>
        <DqInput v-model="presetForm.name" :placeholder="$t('settings.presetNamePlaceholder')" />
      </DqPrefRow>

      <DqPrefRow :label="$t('settings.presetMediaScope')" stacked>
        <div class="settings-stacked-control">
          <DqSegmented
            v-model="presetForm.media_scope"
            size="sm"
            class="settings-media-scope-group"
            :options="[
              { label: $t('settings.presetMediaImage'), value: 'image' },
              { label: $t('settings.presetMediaVideo'), value: 'video' },
            ]"
          />
          <p class="settings-form-hint settings-form-hint--below-control">
            {{ $t('settings.presetMediaScopeHint') }}
          </p>
        </div>
      </DqPrefRow>

      <DqPrefRow :label="$t('settings.positivePrompt')" stacked class="settings-preset-dialog-prompt-row">
        <DqInput
          v-model="presetForm.positive"
          type="textarea"
          class="settings-preset-dialog-textarea settings-preset-dialog-textarea--positive"
          :rows="10"
          :placeholder="$t('settings.positivePlaceholder')"
        />
      </DqPrefRow>

      <DqPrefRow :label="$t('settings.negativePrompt')" stacked class="settings-preset-dialog-prompt-row">
        <div class="settings-stacked-control">
          <DqInput
            v-model="presetForm.negative"
            type="textarea"
            class="settings-preset-dialog-textarea settings-preset-dialog-textarea--negative"
            :rows="5"
            :placeholder="$t('settings.negativePlaceholder')"
          />
          <p class="studio-field-footnote">{{ $t('studio.optional') }}</p>
        </div>
      </DqPrefRow>

      <DqPrefRow :label="$t('settings.loraTriggerWords')" stacked class="settings-preset-dialog-prompt-row">
        <div class="settings-stacked-control">
          <DqInput
            v-model="presetForm.trigger_words"
            type="textarea"
            class="settings-preset-dialog-textarea"
            :rows="3"
            :placeholder="$t('settings.loraTriggerWordsHint')"
          />
          <p class="studio-field-footnote">{{ $t('studio.optional') }}</p>
        </div>
      </DqPrefRow>

      <DqPrefRow :label="$t('settings.presetAppliesTo')" stacked>
        <DqCheckboxGroup v-model="presetForm.applies_to">
          <DqCheckbox label="create">{{ $t('action.image.create') }}</DqCheckbox>
          <DqCheckbox label="rewrite">{{ $t('action.image.rewrite') }}</DqCheckbox>
          <DqCheckbox label="retouch">{{ $t('action.image.retouch') }}</DqCheckbox>
          <DqCheckbox label="extend">{{ $t('action.image.extend') }}</DqCheckbox>
          <DqCheckbox label="upscale">{{ $t('action.image.upscale') }}</DqCheckbox>
          <DqCheckbox label="animate">{{ $t('action.video.animate') }}</DqCheckbox>
        </DqCheckboxGroup>
      </DqPrefRow>
    </DqPrefPane>

    <template #footer>
      <DqButton size="sm" @click="emit('cancel')">{{ $t('common.cancel') }}</DqButton>
      <DqButton type="primary" @click="emit('save')">{{ $t('common.save') }}</DqButton>
    </template>
  </DqDialog>
</template>
