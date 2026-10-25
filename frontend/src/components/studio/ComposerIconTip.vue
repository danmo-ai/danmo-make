<template>
  <DqTooltip :content="content" :placement="placement" :popper-class="popperClass">
    <span class="composer-icon-tip">
      <slot />
    </span>
  </DqTooltip>
</template>

<script setup lang="ts">
withDefaults(
  defineProps<{
    content: string;
    placement?: 'top' | 'bottom' | 'left' | 'right';
    popperClass?: string;
  }>(),
  { placement: 'top', popperClass: '' },
);
</script>

<style scoped>
.composer-icon-tip {
  display: inline-flex;
  align-items: center;
  vertical-align: middle;
  max-width: 100%;
}
</style>
