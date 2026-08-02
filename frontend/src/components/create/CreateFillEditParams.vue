<script setup lang="ts">
defineProps<{
  params: { steps: number; guidance: number };
}>();
</script>

<template>
  <DqPrefPane class="studio-create-pref-pane studio-editor-drawer-pref-pane">
    <DqPrefRow :label="$t('create.steps')">
      <DqInputNumber
        v-model="params.steps"
        :min="1"
        :max="50"
        :step="1"
        size="sm"
        style="width: 120px"
      />
    </DqPrefRow>
    <DqPrefRow :label="$t('create.guidance')">
      <div class="fill-edit-params__guidance">
        <DqSlider v-model="params.guidance" :min="1" :max="50" :step="1" />
        <DqInputNumber
          v-model="params.guidance"
          :min="1"
          :max="50"
          :step="1"
          size="sm"
          style="width: 72px"
        />
      </div>
    </DqPrefRow>
  </DqPrefPane>
</template>

<style scoped>
.fill-edit-params__guidance {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
}
</style>
