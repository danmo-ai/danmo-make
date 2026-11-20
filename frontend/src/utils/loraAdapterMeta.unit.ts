/**
 * Run: npx tsx frontend/src/utils/loraAdapterMeta.unit.ts
 */
import { applyLoraComposeOverrides, registryComposeOverrides, withRegistryLoraMeta } from './loraAdapterMeta';

function assert(cond: boolean, msg: string) {
  if (!cond) throw new Error(msg);
}

const lightning = {
  parameters: {
    compose_overrides: { steps: 4, guidance: 1, lora_scale: 1 },
  },
  metadata: { lora: { hint_key: 'studio.loraHint.qwenLightning', tags: ['lightning'] } },
  versions: {
    'v8-bf16': { compose_overrides: { steps: 8, guidance: 1, lora_scale: 1 } },
  },
};

const card = registryComposeOverrides(lightning, 'qwen-image-2512-lightning-lora');
assert(card?.steps === 4 && card?.guidance === 1, 'card steps and guidance');

const eight = registryComposeOverrides(lightning, 'qwen-image-edit-2511-lightning-lora:v8-bf16');
assert(eight?.steps === 8 && eight?.guidance === 1, '8-step version overrides steps');

const params: Record<string, unknown> = { steps: 20, guidance: 4, lora_scale: 0.8 };
applyLoraComposeOverrides(params, { id: 'qwen-image-2512-lightning-lora' }, lightning);
assert(params.steps === 4 && params.guidance === 1 && params.lora_scale === 1, 'selecting the lora writes advanced params');

const row = withRegistryLoraMeta({ id: 'qwen-image-2512-lightning-lora', ready: true }, lightning);
assert(row.hint_key === 'studio.loraHint.qwenLightning', 'hint comes from the card');
assert(row.compose_overrides?.steps === 4, 'row carries compose overrides');

const ranged = withRegistryLoraMeta(
  { id: 'awportraitcn-hf', ready: true },
  { metadata: { lora: { recommended_lora_scale: [0.9, 1] } } },
);
assert(ranged.recommended_lora_scale?.[0] === 0.9 && ranged.recommended_lora_scale?.[1] === 1, 'card publishes the weight range');

const trained = { steps: 9, guidance: 0 };
applyLoraComposeOverrides(
  params,
  { id: 'user-lora', compose_overrides: trained },
  lightning,
);
assert(params.steps === 9 && params.guidance === 0, 'trained inference block wins over the card');

console.log('loraAdapterMeta.unit: ok');
