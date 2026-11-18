<template>
  <div class="studio-composer-toolbar">
    <div v-if="$slots.mode || $slots.controls" class="studio-composer-toolbar__left">
      <div v-if="$slots.mode" class="studio-composer-toolbar__mode">
        <slot name="mode" />
      </div>
      <slot name="controls" />
    </div>
    <div class="studio-composer-toolbar__right">
      <slot name="actions" />
    </div>
  </div>
</template>

<script setup lang="ts">
/**
 * Shared drawer toolbar shell for Image / Video / Audio composers.
 * Mode segmented control: #mode slot (first row, full width).
 */
</script>
