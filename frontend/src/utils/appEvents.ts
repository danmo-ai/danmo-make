/**
 * Typed in-app event bus (no `window.dispatchEvent` / `CustomEvent`).
 * Extend `AppEventMap` and `ensureKnownKeys` when adding signals.
 */
export type AppEventMap = {
  'open-global-task-queue': { taskId?: string };
  'open-task-log': { taskId: string };
  /** Re-open the first-run onboarding wizard from Settings → Quick Setup. */
  'open-onboarding': void;
};

type Handler<K extends keyof AppEventMap> = (payload: AppEventMap[K]) => void;

type AnyHandler = (payload: unknown) => void;

const store = new Map<keyof AppEventMap, Set<AnyHandler>>();

function bucket<K extends keyof AppEventMap>(type: K): Set<AnyHandler> {
  let s = store.get(type);
  if (!s) {
    s = new Set();
    store.set(type, s);
  }
  return s;
}

export const appEvents = {
  on<K extends keyof AppEventMap>(type: K, fn: Handler<K>): void {
    bucket(type).add(fn as AnyHandler);
  },
  off<K extends keyof AppEventMap>(type: K, fn: Handler<K>): void {
    bucket(type).delete(fn as AnyHandler);
  },
  emit<K extends keyof AppEventMap>(type: K, payload: AppEventMap[K]): void {
    for (const fn of bucket(type)) {
      try {
        (fn as Handler<K>)(payload);
      } catch (e) {
        console.error('[appEvents]', type, e);
      }
    }
  },
};

/** Open the task page (optionally focused on one task). */
export function openGlobalTaskQueue(taskId?: string): void {
  const id = String(taskId || '').trim();
  appEvents.emit('open-global-task-queue', id ? { taskId: id } : {});
}

/** Open generation task log dialog for a task id. */
export function openTaskLog(taskId: string): void {
  const id = String(taskId || '').trim();
  if (!id) return;
  appEvents.emit('open-task-log', { taskId: id });
}

/** Re-run the first-run onboarding wizard (Settings → Quick Setup). */
export function openOnboarding(): void {
  appEvents.emit('open-onboarding', undefined as void);
}
