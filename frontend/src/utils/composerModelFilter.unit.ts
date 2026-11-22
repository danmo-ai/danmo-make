/**
 * Run: npx tsx frontend/src/utils/composerModelFilter.unit.ts
 */
import { isStructuralComposerControlNet } from './composerModelFilter';

function assert(cond: boolean, msg: string) {
  if (!cond) throw new Error(msg);
}

assert(
  isStructuralComposerControlNet({
    category: 'controlnets',
    binding: 'structural',
    actions: { create: {}, retouch: {} },
  }),
  'structural controlnet stays out of the composer',
);

assert(
  !isStructuralComposerControlNet({
    category: 'controlnets',
    binding: 'standalone_edit',
    actions: { retouch: {}, extend: {} },
  }),
  'standalone edit stays selectable',
);

assert(
  isStructuralComposerControlNet({ category: 'controlnets' }),
  'controlnet without binding is not a composer model',
);

assert(
  !isStructuralComposerControlNet({ category: 'base_models', binding: 'structural' }),
  'base models stay in the composer',
);

assert(!isStructuralComposerControlNet(null), 'missing row is not a controlnet');

console.log('composerModelFilter.unit: ok');
