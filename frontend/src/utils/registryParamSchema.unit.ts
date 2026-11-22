/**
 * Unit checks for registry resolution helpers.
 * Run: npx tsx frontend/src/utils/registryParamSchema.unit.ts
 */
import {
  buildResolutionSizeOptions,
  extraReferenceSlotCap,
  pickClosestResolutionPreset,
  referenceImageSlotCap,
  schedulerForModel,
  type ResolutionSizeOption,
} from './registryParamSchema';

function assert(cond: boolean, msg: string) {
  if (!cond) throw new Error(msg);
}

const videoParams = {
  resolution_presets: {
    default: '512x768',
    options: [
      '480x704',
      '512x768',
      '704x1280',
      '768x1280',
      '1280x704',
      '1280x768',
    ],
  },
};

const options = buildResolutionSizeOptions(videoParams);
assert(options.some((o) => o.value === '768x1280'), 'missing 768x1280 preset');
assert(!options.some((o) => o.value.startsWith('720x')), '720 presets must be removed');

const portraitPick = pickClosestResolutionPreset(options, 768, 1280);
assert(portraitPick === '768x1280', `768x1280 source expected 768x1280, got ${portraitPick}`);

const landscapePick = pickClosestResolutionPreset(options, 1280, 768);
assert(landscapePick === '1280x768', `1280x768 source expected 1280x768, got ${landscapePick}`);

const nearPick = pickClosestResolutionPreset(options, 720, 1280);
assert(nearPick === '704x1280', `720x1280 source expected closest 704x1280, got ${nearPick}`);

assert(referenceImageSlotCap({ strength: { type: 'float', default: 0.4 } }) === 1, 'single slot without multi flag');
assert(extraReferenceSlotCap({ edit_plus_multi_image: true, edit_max_reference_images: 3 }) === 2, 'qwen extra slots');
assert(referenceImageSlotCap({ reference_images_support: true, max_reference_images: 3 }) === 3, 'explicit slot cap');
assert(extraReferenceSlotCap({ edit_max_reference_images: 1 }) === 0, 'edit path stays on the primary image');

const klein = {
  scheduler: {
    type: 'enum',
    default: 'flow_match_euler_discrete',
    options: ['flow_match_euler_discrete', 'flow_match_euler'],
  },
};
assert(schedulerForModel(klein, 'linear') === 'flow_match_euler_discrete', 'stale linear falls back to the model default');
assert(schedulerForModel(klein, 'flow_match_euler') === 'flow_match_euler', 'listed scheduler is kept');
assert(schedulerForModel({ steps: { type: 'int', default: 4 } }, 'linear') === undefined, 'model without a scheduler omits it');

console.log('registryParamSchema.unit: ok');
