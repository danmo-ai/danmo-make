/**
 * Unit checks for synced lyric timeline helpers.
 * Run: npx tsx frontend/src/utils/lyricsTimeline.unit.ts
 */
import {
  activeLineIndex,
  lineProgress,
  linesFromSegments,
  linesToLrc,
  parseLrc,
} from './lyricsTimeline';

function assert(cond: boolean, msg: string) {
  if (!cond) throw new Error(msg);
}

const fromSeg = linesFromSegments(
  [
    { label: 'chorus', start_sec: 20, end_sec: 24, text: 'second' },
    { label: 'verse', start_sec: 5.5, end_sec: 0, text: 'first' },
    { label: 'verse', start_sec: 1, end_sec: 2, text: '[verse]' },
    { start_sec: 30, end_sec: 31, text: '   ' },
  ],
  60,
);
assert(fromSeg.length === 2, 'drops section-only and blank lines');
assert(fromSeg[0].text === 'first' && fromSeg[1].text === 'second', 'sorted by start');
assert(fromSeg[0].end === 20, 'open end closes at next start');
assert(fromSeg[1].end === 24, 'keeps explicit end');
assert(linesFromSegments(null).length === 0, 'non-array → empty');

const lrc = parseLrc('[ti:x]\n[00:01.50]hello\n[01:02.25][01:10.00]repeat\n[00:03]\nnot timed', 90);
assert(lrc.length === 3, 'multi-stamp line expands; empty text skipped');
assert(lrc[0].start === 1.5 && lrc[1].start === 62.25 && lrc[2].start === 70, 'lrc times');
assert(lrc[2].end === 90, 'last line ends at duration');

assert(activeLineIndex(lrc, 0) === -1, 'before first line');
assert(activeLineIndex(lrc, 1.5) === 0, 'exact start');
assert(activeLineIndex(lrc, 65) === 1, 'middle');
assert(activeLineIndex(lrc, 1000) === 2, 'after last');
assert(activeLineIndex([], 3) === -1, 'empty');

assert(lineProgress({ start: 10, end: 20, text: 'a' }, 15) === 0.5, 'progress mid');
assert(lineProgress({ start: 10, end: 20, text: 'a' }, 25) === 1, 'progress clamp');
assert(lineProgress(undefined, 3) === 0, 'no line');

const out = linesToLrc([{ start: 62.25, end: 63, text: 'hey' }]);
assert(out === '[01:02.25]hey\n', `lrc format: ${out}`);
assert(parseLrc(out)[0].start === 62.25, 'round trip');

console.log('lyricsTimeline.unit: ok');
