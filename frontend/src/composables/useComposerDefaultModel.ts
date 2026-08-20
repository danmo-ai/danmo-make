import { watch, type WatchSource } from 'vue';

export type ComposerModelOption = { value: string; disabled?: boolean };

/** First dropdown row when current selection is empty or not in the list. */
export function pickDefaultComposerModel(
  options: ComposerModelOption[],
  current: string | undefined | null,
): string | null {
  if (!options.length) return null;
  const key = String(current || '');
  if (key && options.some((o) => o.value === key)) return null;
  return options[0].value;
}

export function useComposerDefaultModel(
  options: WatchSource<ComposerModelOption[]>,
  current: WatchSource<string | undefined>,
  apply: (value: string) => void,
) {
  watch(
    [options, current],
    ([opts, cur]) => {
      const next = pickDefaultComposerModel(opts || [], cur);
      if (next) apply(next);
    },
    { immediate: true },
  );
}
