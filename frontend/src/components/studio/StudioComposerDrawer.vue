<template>
  <DqDrawer
    :open="open"
    :title="title"
    direction="rtl"
    :size="drawerSize"
    class="studio-composer-drawer"
    @update:open="$emit('update:open', $event)"
  >
    <div class="studio-composer-drawer__body">
      <slot />
    </div>
  </DqDrawer>
</template>

<script setup lang="ts">
defineProps<{
  open: boolean;
  title: string;
}>();

defineEmits<{
  (e: 'update:open', value: boolean): void;
}>();

const drawerSize = 'min(560px, 92vw)';
</script>

<style scoped>
.studio-composer-drawer__body {
  display: flex;
  flex-direction: column;
  min-width: 0;
  max-width: 100%;
  min-height: 0;
  height: 100%;
  overflow-x: hidden;
  overflow-y: auto;
  padding: 4px 2px 16px;
  box-sizing: border-box;
}

:deep(.dq-drawer-body) {
  display: flex;
  flex-direction: column;
  min-width: 0;
  max-width: 100%;
  min-height: 0;
  overflow: hidden;
}

:deep(.studio-composer-shell.dq-glass--panel) {
  border: none;
  box-shadow: none;
  background: transparent;
  -webkit-backdrop-filter: none;
  backdrop-filter: none;
}
</style>
