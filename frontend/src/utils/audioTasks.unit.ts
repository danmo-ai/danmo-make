import {
  buildAudioEditPayload,
  buildAudioGenerationPayload,
  dcwEnabledForSubmit,
  normalizeAudioRegion,
  parseAudioTracks,
  serializeAudioTracks,
  supportsAudioAction,
  validateAudioEditInput,
} from './audioTasks';

function assert(value: unknown, message: string) {
  if (!value) throw new Error(message);
}

const sft = { actions: { create: {}, cover: {}, repaint: {}, lego: {}, extract: {}, complete: {}, lm: {} } };
const turbo = { actions: { create: {}, cover: {}, repaint: {}, lm: {} } };
assert(supportsAudioAction(sft, 'complete'), 'SFT supports complete');
assert(!supportsAudioAction(turbo, 'lego'), 'Turbo hides base-only lego');
assert(serializeAudioTracks('complete', ['drums', 'bass']) === 'drums | bass', 'multi-track wire format');
assert(serializeAudioTracks('lego', ['drums', 'bass']) === 'drums', 'single-track mode');
assert(parseAudioTracks('drums | bass | invalid').join(',') === 'drums,bass', 'known tracks only');
assert(normalizeAudioRegion(-1, 20, 10)?.end === 10, 'region clamps to source');
assert(normalizeAudioRegion(5, 5, 10) === null, 'empty region rejected');
assert(dcwEnabledForSubmit('auto') === null && dcwEnabledForSubmit('on') === true, 'DCW tri-state');

const edit = buildAudioEditPayload('lego', 'ast_source', {
  model: 'ace-step-xl-sft',
  title: 'Layered',
  tracks: ['drums'],
  region_start_sec: 1,
  region_end_sec: 4,
  dcw_enabled: 'off',
});
assert(edit.task === 'lego' && edit.track === 'drums', 'edit task payload');
assert(edit.region_start_sec === 1 && edit.region_end_sec === 4, 'region payload');
assert(edit.dcw_enabled === false, 'explicit DCW off survives');
assert(validateAudioEditInput('lego', {
  model: 'ace-step-xl-sft',
  sourceAssetId: 'ast_source',
  sourceDuration: 10,
  tracks: ['drums'],
  region_start_sec: 1,
  region_end_sec: 4,
}) === null, 'valid lego input');

const generated = buildAudioGenerationPayload({
  model: 'ace-step-xl-turbo',
  audio_code_indices: [1, 2],
  lm_expansion: 'off',
});
assert(generated.lm_expansion === 'off', 'planner codes skip another expansion');

console.log('audioTasks.unit: ok');
