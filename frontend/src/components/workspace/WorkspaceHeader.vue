<template>
  <header class="workspace-header">
    <div class="workspace-header__heading">
      <div class="workspace-header__title">
        <span v-if="$slots.icon" class="workspace-header__icon" aria-hidden="true">
          <slot name="icon" />
        </span>
        <h2>{{ title }}</h2>
        <slot name="title-meta" />
      </div>
      <p v-if="description">{{ description }}</p>
    </div>
    <div v-if="$slots.filters" class="workspace-header__filters">
      <slot name="filters" />
    </div>
    <div v-if="$slots.actions" class="workspace-header__actions">
      <slot name="actions" />
    </div>
  </header>
</template>

<script setup lang="ts">
defineProps<{
  title: string;
  description?: string;
}>();
</script>
