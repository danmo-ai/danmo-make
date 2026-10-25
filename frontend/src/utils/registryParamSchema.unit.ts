/**
 * Unit checks for registry resolution helpers.
 * Run: npx tsx frontend/src/utils/registryParamSchema.unit.ts
 */
import {
  buildResolutionSizeOptions,
  extraReferenceSlotCap,
  pickClosestResolutionPreset,
  referenceImageSlotCap,
  referenceUse,
  type ResolutionSizeOption,
} from './registryParamSchema';

function assert(cond: boolean, msg: string) {
  if (!cond) throw new Error(msg);
}

const ltxParams = {
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

const options = buildResolutionSizeOptions(ltxParams);
assert(options.some((o) => o.value === '768x1280'), 'missing 768x1280 preset');
assert(!options.some((o) => o.value.startsWith('720x')), '720 presets must be removed');

const portraitPick = pickClosestResolutionPreset(options, 768, 1280);
assert(portraitPick === '768x1280', `768x1280 source expected 768x1280, got ${portraitPick}`);

const landscapePick = pickClosestResolutionPreset(options, 1280, 768);
assert(landscapePick === '1280x768', `1280x768 source expected 1280x768, got ${landscapePick}`);

const nearPick = pickClosestResolutionPreset(options, 720, 1280);
assert(nearPick === '704x1280', `720x1280 source expected closest 704x1280, got ${nearPick}`);

assert(referenceImageSlotCap({ strength: { type: 'float', default: 0.4 } }) === 1, 'single slot when no cap is declared');
assert(referenceImageSlotCap({ edit_plus_multi_image: true }) === 1, 'boolean alone does not add slots');
assert(extraReferenceSlotCap({ edit_max_reference_images: 3 }) === 2, 'qwen extra slots');
assert(referenceImageSlotCap({ max_reference_images: 3 }) === 3, 'hidream slot cap');
assert(referenceImageSlotCap({ edit_max_reference_images: 2, max_reference_images: 5 }) === 2, 'edit max wins');
assert(extraReferenceSlotCap({ edit_max_reference_images: 1 }) === 0, 'boogu stays on the primary image');
assert(referenceUse({ edit_use_vl_vision: true }) === 'instruct', 'vl edit');
assert(referenceUse({ edit_conditioning_concat: true }) === 'instruct', 'concat edit');
assert(referenceUse({ reference_images_support: true }) === 'instruct', 'hidream refs');
assert(
  referenceUse({ image_pipeline_shape: 'family_generator' }, { rewrite: {} }) === 'instruct',
  'family generator edit',
);
assert(
  referenceUse({ image_pipeline_shape: 'family_generator' }, { create: {} }) === 'redraw',
  'family generator create stays a redraw',
);
assert(referenceUse({ strength: { type: 'float', default: 0.4 } }, { rewrite: {} }) === 'redraw', 'flux redraw');

console.log('registryParamSchema.unit: ok');
