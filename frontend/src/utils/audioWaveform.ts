import { api } from '@/utils/api';

const cache = new Map<string, Float32Array>();
const CACHE_LIMIT = 8;

export async function loadAudioPeaks(src: string, columns = 960): Promise<Float32Array> {
  const key = `${src}:${columns}`;
  const cached = cache.get(key);
  if (cached) {
    cache.delete(key);
    cache.set(key, cached);
    return cached;
  }
  const blob = await api.gen.urlToBlob(src);
  const bytes = await blob.arrayBuffer();
  const AudioContextCtor = window.AudioContext;
  if (!AudioContextCtor) throw new Error('Web Audio is unavailable');
  const context = new AudioContextCtor();
  try {
    const decoded = await context.decodeAudioData(bytes.slice(0));
    const peaks = new Float32Array(columns);
    const channels = Array.from({ length: decoded.numberOfChannels }, (_, index) => decoded.getChannelData(index));
    const framesPerColumn = decoded.length / columns;
    for (let column = 0; column < columns; column += 1) {
      const start = Math.floor(column * framesPerColumn);
      const end = Math.min(decoded.length, Math.max(start + 1, Math.floor((column + 1) * framesPerColumn)));
      let peak = 0;
      const stride = Math.max(1, Math.floor((end - start) / 256));
      for (const channel of channels) {
        for (let frame = start; frame < end; frame += stride) {
          peak = Math.max(peak, Math.abs(channel[frame] || 0));
        }
      }
      peaks[column] = peak;
    }
    cache.set(key, peaks);
    if (cache.size > CACHE_LIMIT) {
      const oldest = cache.keys().next().value;
      if (oldest) cache.delete(oldest);
    }
    return peaks;
  } finally {
    await context.close().catch(() => {});
  }
}
