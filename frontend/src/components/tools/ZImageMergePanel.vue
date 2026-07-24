<template>
  <details class="zimage-merge-panel">
    <summary>{{ $t('tools.zImageMergeTitle') }}</summary>
    <p class="zimage-merge-panel__hint">{{ $t('tools.zImageMergeHint') }}</p>
    <p v-if="!mlxAvailable" class="zimage-merge-panel__warn">{{ $t('tools.mergeMlxRequired') }}</p>
    <DqPrefPane class="zimage-merge-panel__form">
      <DqPrefRow :label="$t('tools.mergeModelA')">
        <DqSelect v-model="form.model_a" size="sm" filterable :placeholder="$t('tools.mergePickModel')">
          <DqOption v-for="m in mergeModels" :key="m.id" :label="modelLabel(m)" :value="m.id" />
        </DqSelect>
      </DqPrefRow>
      <DqPrefRow :label="$t('tools.mergeModelB')">
        <DqSelect v-model="form.model_b" size="sm" filterable clearable :placeholder="$t('tools.mergePickModel')">
          <DqOption v-for="m in mergeModels" :key="`b-${m.id}`" :label="modelLabel(m)" :value="m.id" />
        </DqSelect>
      </DqPrefRow>
      <DqPrefRow :label="$t('tools.mergeMethod')">
        <DqSelect v-model="form.method" size="sm">
          <DqOption value="weighted_sum" :label="$t('tools.mergeWeightedSum')" />
          <DqOption value="add_difference" :label="$t('tools.mergeAddDifference')" />
        </DqSelect>
      </DqPrefRow>
      <DqPrefRow v-if="form.method === 'add_difference'" :label="$t('tools.mergeModelC')">
        <DqSelect v-model="form.model_c" size="sm" filterable clearable :placeholder="$t('tools.mergePickModel')">
          <DqOption v-for="m in mergeModels" :key="`c-${m.id}`" :label="modelLabel(m)" :value="m.id" />
        </DqSelect>
      </DqPrefRow>
      <DqPrefRow :label="$t('tools.mergeAlpha')">
        <DqSlider v-model="form.alpha" :min="0" :max="1" :step="0.05" />
        <span class="zimage-merge-panel__alpha">{{ form.alpha.toFixed(2) }}</span>
      </DqPrefRow>
      <DqPrefRow :label="$t('tools.mergeOutputName')">
        <DqInput v-model="form.output_name" size="sm" />
      </DqPrefRow>
      <DqPrefRow :label="$t('tools.mergeAutoRegister')">
        <DqSwitch v-model="form.auto_register" />
      </DqPrefRow>
      <p v-if="form.auto_register" class="zimage-merge-panel__hint">{{ $t('tools.mergeAutoRegisterHint') }}</p>
    </DqPrefPane>
    <DqButton type="primary" :loading="submitting" :disabled="!mlxAvailable" @click="submitMerge">
      {{ $t('tools.mergeSubmit') }}
    </DqButton>
  </details>
</template>

<script setup lang="ts">
import { onMounted, reactive, ref } from 'vue';
import { api } from '@/utils/api';
import { $mn, $tt } from '@/utils/i18n';
import { toast } from '@/utils/feedback';
import { useRegistryStore } from '@/stores/registry';
import { useTasksStore } from '@/stores/tasks';

const emit = defineEmits<{ (e: 'merged-complete'): void }>();

type MergeModelRow = { id: string; name: unknown };

const submitting = ref(false);
const mlxAvailable = ref(true);
const mergeModels = ref<MergeModelRow[]>([]);
const registryStore = useRegistryStore();
const tasksStore = useTasksStore();
const form = reactive({
  model_a: 'z-image-turbo',
  model_b: '',
  model_c: '',
  method: 'weighted_sum',
  alpha: 0.5,
  output_name: '',
  auto_register: true,
});

function modelLabel(m: MergeModelRow): string {
  return $mn(m as { name?: string | { zh?: string; en?: string } }, m.id);
}

async function refreshMergeModels() {
  try {
    const data = await api.tools.listZImageMergeModels();
    mergeModels.value = data.models || [];
    mlxAvailable.value = data.mlx_available !== false;
    if (!form.model_a && mergeModels.value.length) {
      form.model_a = mergeModels.value[0].id;
    }
  } catch {
    mergeModels.value = [];
  }
}

onMounted(() => {
  void refreshMergeModels();
});

async function submitMerge() {
  if (!form.model_a.trim() || !form.model_b.trim() || !form.output_name.trim()) {
    toast.warning($tt('tools.mergeFieldsRequired'));
    return;
  }
  submitting.value = true;
  try {
    const res = await api.tools.submitZImageMerge({
      model_a: form.model_a.trim(),
      model_b: form.model_b.trim(),
      model_c: form.method === 'add_difference' ? form.model_c.trim() || undefined : undefined,
      method: form.method,
      alpha: form.alpha,
      output_name: form.output_name.trim(),
      auto_register: form.auto_register,
    }) as { task?: { id?: string } };
    const tid = res?.task?.id;
    toast.success(tid ? $tt('tools.mergeQueued', { id: tid }) : $tt('tools.mergeQueuedGeneric'));
    if (tid) {
      let mergeResult: { metadata?: { z_image_merge?: { registered_model_id?: string } } } | null = null;
      tasksStore.openTaskLogStream(tid, {
        onResult: (resultData) => {
          mergeResult = (resultData as { metadata?: { z_image_merge?: { registered_model_id?: string } } }) || null;
        },
        onDone: async (doneData) => {
          const row = doneData as { status?: string };
          if (row.status !== 'completed') return;
          await registryStore.load(true);
          await refreshMergeModels();
          const modelId = String(mergeResult?.metadata?.z_image_merge?.registered_model_id || '').trim();
          if (modelId) {
            toast.success($tt('tools.mergeComplete', { modelId }));
          } else {
            toast.success($tt('tools.mergeCompleteGeneric'));
          }
          emit('merged-complete');
        },
      });
      tasksStore.pollQueueOnce();
    }
  } catch (e) {
    toast.error((e as Error).message || String(e));
  } finally {
    submitting.value = false;
  }
}
</script>

<style scoped>
.zimage-merge-panel {
  margin-top: 12px;
  padding-top: 8px;
  border-top: 1px solid var(--dq-border-subtle);
}
.zimage-merge-panel__hint,
.zimage-merge-panel__warn {
  margin: 8px 0 12px;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
  line-height: 1.45;
}
.zimage-merge-panel__warn {
  color: var(--dq-warning);
}
.zimage-merge-panel__form {
  margin-bottom: 12px;
}
.zimage-merge-panel__alpha {
  min-width: 36px;
  text-align: right;
  font-size: var(--dq-font-size-caption);
  color: var(--dq-label-secondary);
}
</style>
