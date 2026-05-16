import { useCallback, useEffect, useRef, useState } from "react";

interface UseDebouncedSaveOptions {
  /** Milliseconds to wait after the last save() call before firing. Default 600. */
  delay?: number;
  /** When true, calling save() while a save is in-flight queues a re-run with
   *  the latest value once the current save settles. Default true. */
  coalesce?: boolean;
}

interface UseDebouncedSaveReturn<T> {
  /** Schedule a save. Cancels any prior pending timer. If a save is in-flight
   *  and coalesce is true, the latest value runs again after settle. */
  save: (value: T) => void;
  /** Immediately fire the pending value. Resolves when the save settles. Use
   *  on explicit Save buttons so the user can't double-save by clicking
   *  while a debounced save is queued. */
  flush: () => Promise<void>;
  /** Drop the pending value without firing. Use on cancel/discard flows. */
  cancel: () => void;
  /** A save is currently executing. */
  isSaving: boolean;
  /** A debounce timer is armed but not yet fired. */
  isPending: boolean;
  /** Last rejection from saveFn. Consumer renders via <FieldError>. */
  error: Error | null;
  clearError: () => void;
}

// Debounced async-save hook. Generic over the value shape so each form field
// (display name, bio, handle, social URL, payout account name, etc.) passes
// its own draft type without rolling its own setTimeout dance.
//
// Design rules (locked here, inherited by every auto-save consumer):
//   - flush() bypasses the debounce — Save buttons must call it.
//   - error is the rejection from saveFn; this hook does NOT toast. Toast
//     vs inline is a per-feature decision left to the consumer.
//   - Unmount cancels pending timers without firing — no stale saves leaking
//     after a navigation.
//   - Coalesce queues a single re-run, not a queue. Rapid typing → at most
//     two saves total (the first that fires + one more after it settles).
export function useDebouncedSave<T>(
  saveFn: (value: T) => Promise<void>,
  options: UseDebouncedSaveOptions = {},
): UseDebouncedSaveReturn<T> {
  const { delay = 600, coalesce = true } = options;

  const saveFnRef = useRef(saveFn);
  saveFnRef.current = saveFn;

  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingValueRef = useRef<{ value: T } | null>(null);
  const queuedValueRef = useRef<{ value: T } | null>(null);
  const isMountedRef = useRef(true);

  const [isSaving, setIsSaving] = useState(false);
  const [isPending, setIsPending] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  // Drain loop. Recursion + a conditional finally used to leave isSaving=true
  // stuck-on when a coalesced save() landed between the try's queue-check and
  // the finally's queue-check (the queued value would sit forever, never
  // fired, because the running runSave had already exited). Now: loop until
  // queue is empty, then always clear isSaving unconditionally.
  const runSave = useCallback(async (initialValue: T) => {
    setIsSaving(true);
    setIsPending(false);
    setError(null);
    let cur: T = initialValue;
    while (true) {
      try {
        await saveFnRef.current(cur);
      } catch (err) {
        if (isMountedRef.current) {
          setError(err instanceof Error ? err : new Error("Save failed."));
        }
        break;
      }
      if (!isMountedRef.current) return;
      const queued = queuedValueRef.current;
      queuedValueRef.current = null;
      if (!queued) break;
      cur = queued.value;
    }
    if (isMountedRef.current) {
      setIsSaving(false);
    }
  }, []);

  const save = useCallback(
    (value: T) => {
      if (isSaving && coalesce) {
        queuedValueRef.current = { value };
        return;
      }
      pendingValueRef.current = { value };
      clearTimer();
      setIsPending(true);
      timerRef.current = setTimeout(() => {
        const pending = pendingValueRef.current;
        pendingValueRef.current = null;
        timerRef.current = null;
        if (pending) void runSave(pending.value);
      }, delay);
    },
    [clearTimer, coalesce, delay, isSaving, runSave],
  );

  const flush = useCallback(async () => {
    clearTimer();
    setIsPending(false);
    const pending = pendingValueRef.current;
    pendingValueRef.current = null;
    if (pending) {
      await runSave(pending.value);
    }
  }, [clearTimer, runSave]);

  const cancel = useCallback(() => {
    clearTimer();
    pendingValueRef.current = null;
    queuedValueRef.current = null;
    setIsPending(false);
  }, [clearTimer]);

  const clearError = useCallback(() => setError(null), []);

  useEffect(() => {
    return () => {
      isMountedRef.current = false;
      clearTimer();
      pendingValueRef.current = null;
      queuedValueRef.current = null;
    };
  }, [clearTimer]);

  return { save, flush, cancel, isSaving, isPending, error, clearError };
}
