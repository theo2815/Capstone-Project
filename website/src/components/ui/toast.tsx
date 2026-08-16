"use client";

import { useEffect, useRef } from "react";
import Link from "next/link";
import {
  useToastStore,
  type Toast as ToastT,
  type ToastKind,
} from "@/store/toast-store";
import { cn } from "@/lib/utils";
import { KICKER_SIZE_CLASS } from "@/components/ui/kicker";

const MAX_VISIBLE = 3;

export function ToastContainer() {
  const toasts = useToastStore((s) => s.toasts);
  // Show the latest 3; older queued toasts wait for room. Newest sits at top
  // of the visible stack via flex-col-reverse.
  const visible = toasts.slice(-MAX_VISIBLE);

  return (
    <div
      role="region"
      aria-label="Notifications"
      className="fixed inset-x-4 bottom-4 sm:inset-x-auto sm:right-6 sm:bottom-6 z-[60] flex flex-col-reverse gap-2.5 pointer-events-none sm:w-[28rem]"
    >
      {visible.map((t) => (
        <ToastItem key={t.id} toast={t} />
      ))}
    </div>
  );
}

const DOT_CLASS: Record<ToastKind, string> = {
  success: "bg-fresh",
  error: "bg-error",
  info: "bg-slate",
};

function ToastItem({ toast }: { toast: ToastT }) {
  const dismiss = useToastStore((s) => s.dismissToast);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const remainingRef = useRef<number>(toast.duration);
  const startTimeRef = useRef<number>(Date.now());
  const isPausedRef = useRef<boolean>(false);

  useEffect(() => {
    remainingRef.current = toast.duration;
    startTimeRef.current = Date.now();
    timerRef.current = setTimeout(
      () => dismiss(toast.id),
      remainingRef.current,
    );
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [toast.id, toast.duration, dismiss]);

  function pause() {
    if (isPausedRef.current) return;
    isPausedRef.current = true;
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      remainingRef.current = Math.max(
        0,
        remainingRef.current - (Date.now() - startTimeRef.current),
      );
    }
  }

  function resume() {
    if (!isPausedRef.current) return;
    isPausedRef.current = false;
    startTimeRef.current = Date.now();
    timerRef.current = setTimeout(
      () => dismiss(toast.id),
      remainingRef.current,
    );
  }

  const hasActions = Boolean(toast.link || toast.action);

  return (
    <div
      role={toast.kind === "error" ? "alert" : "status"}
      aria-live={toast.kind === "error" ? "assertive" : "polite"}
      onMouseEnter={pause}
      onMouseLeave={resume}
      onFocus={pause}
      onBlur={resume}
      className={cn(
        "pointer-events-auto w-full rounded-2xl border border-line bg-bone",
        "shadow-[0_18px_50px_-12px_rgba(17,17,17,0.20)]",
        // Wraps below `sm`, where the toast is only viewport-minus-32 wide.
        // Message + two actions + divider + close never fit one row there: the
        // message column collapsed to ~93px and broke "Saved <event name>."
        // across three lines even before the mono labels moved up to the 13px
        // comfort floor. Letting the action group drop to its own line gives
        // the message the full width instead of a sliver.
        "px-5 py-4 flex items-start gap-4 flex-wrap sm:flex-nowrap",
      )}
      style={{ animation: "toast-in 0.3s cubic-bezier(0.16, 1, 0.3, 1) both" }}
    >
      {/* Left: dot + message */}
      {/* min-w forces the wrap above: with `min-w-0` the message shrinks
          indefinitely instead, so the row never breaks and the text slivers. */}
      <div className="flex items-start gap-3 flex-1 min-w-[13rem] sm:min-w-0">
        <span
          aria-hidden="true"
          className={cn(
            "mt-[0.55rem] size-1.5 rounded-full shrink-0",
            DOT_CLASS[toast.kind],
          )}
        />
        <p className="flex-1 min-w-0 font-display text-base text-ink leading-snug">
          {toast.message}
        </p>
      </div>

      {/* Right: actions + divider + close, vertically centered as a group */}
      <div className="flex items-center gap-4 self-center shrink-0 ml-auto">
        {hasActions && (
          <>
            <div className="flex items-center gap-4">
              {toast.link && (
                <Link
                  href={toast.link.href}
                  onClick={() => dismiss(toast.id)}
                  className={cn(
                    "font-mono uppercase text-fresh hover:text-fresh-deep transition-colors inline-flex items-center gap-1 whitespace-nowrap rounded-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone",
                    KICKER_SIZE_CLASS.sm,
                  )}
                >
                  {toast.link.label}
                  <span aria-hidden="true">→</span>
                </Link>
              )}
              {toast.action && (
                <button
                  type="button"
                  onClick={() => {
                    toast.action?.onClick();
                    dismiss(toast.id);
                  }}
                  className={cn(
                    "font-mono uppercase text-ink hover:text-fresh transition-colors whitespace-nowrap",
                    KICKER_SIZE_CLASS.sm,
                  )}
                >
                  {toast.action.label}
                </button>
              )}
            </div>
            <span
              aria-hidden="true"
              className="w-px h-5 bg-line shrink-0"
            />
          </>
        )}
        <button
          type="button"
          onClick={() => dismiss(toast.id)}
          aria-label="Dismiss notification"
          className="size-5 rounded-full text-slate-soft hover:text-ink transition-colors flex items-center justify-center"
        >
          <svg
            viewBox="0 0 12 12"
            className="size-3"
            fill="none"
            aria-hidden="true"
          >
            <path
              d="M2 2 L10 10 M10 2 L2 10"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
            />
          </svg>
        </button>
      </div>
    </div>
  );
}
