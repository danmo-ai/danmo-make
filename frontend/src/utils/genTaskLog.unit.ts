/**
 * Unit checks for generation task log parsing (TeaCache / inference plan).
 * Run: npx tsx frontend/src/utils/genTaskLog.unit.ts
 */
import {
  buildInferenceResultLogMessage,
  buildLogDisplayItems,
  classifyLogEntry,
  parseInferencePlanLog,
  parseTeacacheSummaryLog,
} from './genTaskLog';

function assert(cond: boolean, msg: string) {
  if (!cond) throw new Error(msg);
}

const plan = parseInferencePlanLog(
  '[inference] family=wan steps=40 teacache=quality batched_cfg=on attn=mlx',
);
assert(plan?.family === 'wan', 'family');
assert(plan?.teacache === 'quality', 'teacache mode');
assert(plan?.batched_cfg === 'on', 'batched cfg');

const summary = parseTeacacheSummaryLog(
  'TeaCache skipped 7/25 steps (28%), thresh=0.350',
);
assert(summary?.skipped === '7', 'skipped count');
assert(summary?.total === '25', 'total steps');
assert(summary?.skip_pct === '28', 'skip pct');

assert(
  classifyLogEntry('[inference] family=z_image steps=28 teacache=auto', 'info') === 'milestone',
  'inference plan milestone',
);
assert(
  classifyLogEntry('TeaCache skipped 3/20 steps (15%), thresh=0.200', 'info') === 'milestone',
  'teacache summary milestone',
);

const planItems = buildLogDisplayItems(
  [
    {
      time: '12:00:00',
      message: '[inference] family=z_image steps=28 teacache=quality preview=stream/auto',
      level: 'info',
    },
  ],
  false,
);
assert(planItems.length === 1, 'one plan item');
assert(
  (planItems[0].chips?.some((chip) => chip.key === 'teacache' && chip.value.length > 0)) ?? false,
  'teacache chip',
);

const summaryItems = buildLogDisplayItems(
  [
    {
      time: '12:00:01',
      message: 'TeaCache skipped 7/25 steps (28%), thresh=0.350',
      level: 'info',
    },
  ],
  false,
);
assert(summaryItems.length === 1, 'one summary item');
assert(
  (summaryItems[0].chips?.some((chip) => chip.key === 'skipped' && chip.value === '7/25')) ?? false,
  'skipped chip',
);

const fromMeta = buildInferenceResultLogMessage({
  teacache_skipped: 7,
  teacache_computed: 18,
  teacache_skip_rate: 0.28,
  teacache_thresh: 0.35,
});
assert(fromMeta === 'TeaCache skipped 7/25 steps (28%), thresh=0.350', 'result metadata log line');

console.log('genTaskLog.unit.ts OK');
