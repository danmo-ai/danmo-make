<template>
  <div class="workspace-layout">
    <aside v-if="$slots.sidebar" class="workspace-layout__sidebar">
      <slot name="sidebar" />
    </aside>
    <section class="workspace-layout__main">
      <slot />
    </section>
  </div>
</template>
