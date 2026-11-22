const pauseListeners = new Set<() => void>();

export function registerAudioPlayback(pause: () => void): () => void {
  pauseListeners.add(pause);
  return () => pauseListeners.delete(pause);
}

export function claimExclusiveAudioPlayback(current: () => void): void {
  for (const pause of pauseListeners) {
    if (pause !== current) pause();
  }
}
