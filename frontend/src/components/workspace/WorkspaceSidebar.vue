<template>
  <aside class="workspace-sidebar">
    <header class="workspace-sidebar__header">
      <div class="workspace-sidebar__title">
        <span v-if="$slots.icon" class="workspace-sidebar__icon" aria-hidden="true">
          <slot name="icon" />
        </span>
        <h1>{{ title }}</h1>
      </div>
      <p v-if="description">{{ description }}</p>
    </header>
    <div class="workspace-sidebar__body">
      <slot />
    </div>
    <footer v-if="$slots.footer" class="workspace-sidebar__footer">
      <slot name="footer" />
    </footer>
  </aside>
</template>

<script setup lang="ts">
defineProps<{
  title: string;
  description?: string;
}>();
</script>
