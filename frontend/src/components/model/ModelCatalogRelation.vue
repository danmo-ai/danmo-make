<script setup lang="ts">
import { computed } from 'vue';

const props = defineProps<{
  /** Static role label on this card, e.g. 蒸馏版 / 基础版 / 有新版 */
  roleLabel: string;
  /** Prefix before target name on the navigation control */
  navLabel: string;
  targetName: string;
  targetId: string;
  roleTagType?: 'info' | 'success' | 'warning';
}>();

defineEmits<{
  (e: 'navigate', targetId: string): void;
}>();

const navTitle = computed(() => `${props.navLabel} ${props.targetName}`.trim());
</script>

<template>
  <div class="model-catalog-relation">
    <DqTag
      size="sm"
      effect="plain"
      :type="roleTagType || 'info'"
      class="model-catalog-relation__role"
    >
      {{ roleLabel }}
    </DqTag>
    <button
      type="button"
      class="model-catalog-relation__nav"
      :title="navTitle"
      :aria-label="navTitle"
      @click="$emit('navigate', targetId)"
    >
      <span class="model-catalog-relation__nav-prefix">{{ navLabel }}</span>
      <span class="model-catalog-relation__nav-name">{{ targetName }}</span>
      <span class="model-catalog-relation__nav-arrow" aria-hidden="true">›</span>
    </button>
  </div>
</template>

<style scoped>
.model-catalog-relation {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 6px;
}

.model-catalog-relation__role {
  flex-shrink: 0;
}

.model-catalog-relation__nav {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  max-width: 100%;
  margin: 0;
  padding: 0;
  border: 0;
  background: transparent;
  color: var(--dq-accent);
  font: inherit;
  font-size: var(--dq-font-size-caption);
  line-height: 1.4;
  cursor: pointer;
  text-align: left;
}

.model-catalog-relation__nav:hover .model-catalog-relation__nav-name,
.model-catalog-relation__nav:focus-visible .model-catalog-relation__nav-name {
  text-decoration: underline;
}

.model-catalog-relation__nav:focus-visible {
  outline: 2px solid color-mix(in srgb, var(--dq-accent) 45%, transparent);
  outline-offset: 2px;
  border-radius: 2px;
}

.model-catalog-relation__nav-prefix {
  color: var(--dq-label-secondary);
  white-space: nowrap;
}

.model-catalog-relation__nav-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-weight: 500;
}

.model-catalog-relation__nav-arrow {
  flex-shrink: 0;
  font-size: var(--dq-font-size-body);
  line-height: 1;
}
</style>
