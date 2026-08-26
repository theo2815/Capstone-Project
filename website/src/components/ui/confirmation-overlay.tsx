"use client";

import { useEffect, useRef } from "react";
import { Modal } from "@/components/ui/modal";
import { useConfirmationStore } from "@/store/confirmation-store";
import { cn } from "@/lib/utils";

// Singleton overlay — reads the active request from the store and renders
// one Modal-wrapped confirmation at a time. Mounted globally in providers.tsx
// alongside <ToastContainer />.
export function ConfirmationOverlay() {
  const active = useConfirmationStore((s) => s.active);
  const loading = useConfirmationStore((s) => s.loading);
  const error = useConfirmationStore((s) => s.error);
  const close = useConfirmationStore((s) => s.close);
  const setLoading = useConfirmationStore((s) => s.setLoading);
  const setError = useConfirmationStore((s) => s.setError);

  // Restore focus to the element that triggered the confirmation when it closes.
  const triggerRef = useRef<HTMLElement | null>(null);
  useEffect(() => {
    if (active) {
      triggerRef.current = document.activeElement as HTMLElement | null;
    } else if (triggerRef.current) {
      triggerRef.current.focus?.();
      triggerRef.current = null;
    }
  }, [active]);

  // Enter triggers confirm while overlay is open and not already working.
  useEffect(() => {
    if (!active) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Enter" && !loading) {
        e.preventDefault();
        runConfirm();
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
    // runConfirm captures `active` via closure; re-bind on each new request.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, loading]);

  if (!active) return null;

  const { config } = active;
  const confirmLabel = config.confirmLabel ?? "Confirm";
  const cancelLabel = config.cancelLabel ?? "Cancel";

  function handleCancel() {
    if (loading) return;
    close(false);
  }

  async function runConfirm() {
    if (!active) return;
    const cb = active.config.onConfirm;
    if (!cb) {
      close(true);
      return;
    }
    setError(null);
    setLoading(true);
    try {
      await cb();
      close(true);
    } catch (err) {
      setLoading(false);
      const msg =
        err instanceof Error && err.message
          ? err.message
          : "Something went wrong. Please try again.";
      setError(msg);
    }
  }

  return (
    <Modal isOpen={true} onClose={handleCancel} title={config.title} hideClose>
      <div className="space-y-5">
        {typeof config.message === "string" ? (
          <p className="font-sans text-sm text-ink-soft">{config.message}</p>
        ) : (
          <div className="font-sans text-sm text-ink-soft">{config.message}</div>
        )}

        {error && (
          <p
            role="alert"
            className="font-mono uppercase text-[14px] min-[400px]:text-[15px] md:text-[13px] tracking-[0.18em] text-error"
          >
            {error}
          </p>
        )}

        <div className="flex items-center justify-end gap-3 pt-2">
          <button
            type="button"
            onClick={handleCancel}
            disabled={loading}
            className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate hover:text-ink transition-colors disabled:opacity-50 disabled:cursor-not-allowed px-2"
          >
            {cancelLabel}
          </button>
          <button
            type="button"
            onClick={runConfirm}
            disabled={loading}
            className={cn(
              "font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] rounded-full px-5 py-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed",
              config.danger
                ? "bg-bone-deep border border-line text-error hover:bg-line"
                : "bg-fresh text-surface hover:bg-fresh-deep",
            )}
          >
            {loading ? "Working…" : confirmLabel}
          </button>
        </div>
      </div>
    </Modal>
  );
}
