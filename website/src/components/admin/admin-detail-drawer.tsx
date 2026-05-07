"use client";

import { useEffect, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { useScrollLock } from "@/lib/scroll-lock";
import { Kicker } from "@/components/ui/kicker";
import { useAdminLegendStore } from "@/store/admin-legend-store";
import { useAdminPaletteStore } from "@/store/admin-palette-store";
import { cn } from "@/lib/utils";

interface AdminDetailDrawerProps {
  open: boolean;
  onClose: () => void;
  kicker?: string;
  title: ReactNode;
  rightHeader?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  /**
   * Suppress the ESC keydown listener while a stacked modal (e.g. reject
   * reason form) is open above the drawer, so a single ESC press only
   * closes the topmost surface. Mirrors the Modal escDisabled pattern.
   */
  escDisabled?: boolean;
  className?: string;
}

// Phase 3 admin redesign — full-screen detail surface for queue rows.
// Portals to document.body so it escapes any transform/stacking-context
// ancestor (stagger-children, fade-up, sticky containers etc.). The list
// keeps running underneath but is fully covered by bone background. ESC
// and the [×] in the header close. Sticky action footer renders the
// primary verbs; destructive confirm modals (reject reason, refund
// amount) stack on top at z-60 — pass escDisabled while they're open so
// ESC steps one surface at a time.
//
// One drawer mounts at a time per page; the host queue owns the
// open/close state via useUrlState("row").
export function AdminDetailDrawer({
  open,
  onClose,
  kicker,
  title,
  rightHeader,
  subtitle,
  actions,
  children,
  escDisabled = false,
  className,
}: AdminDetailDrawerProps) {
  const [mounted, setMounted] = useState(false);
  const legendOpen = useAdminLegendStore((s) => s.open);
  const paletteOpen = useAdminPaletteStore((s) => s.open);
  // The keyboard legend and command palette both render as Modals at z-60
  // above the drawer — each owns its own ESC handler. Stand the drawer's
  // ESC down while either is open so a single ESC press only dismisses
  // the topmost surface, not both.
  const effectiveEscDisabled = escDisabled || legendOpen || paletteOpen;
  useScrollLock(open);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!open || effectiveEscDisabled) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, effectiveEscDisabled, onClose]);

  if (!open || !mounted) return null;

  const content = (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={typeof title === "string" ? title : undefined}
      className={cn(
        "fixed inset-0 z-50 bg-bone flex flex-col",
        "animate-[fade-up_0.18s_ease-out_both]",
        className,
      )}
    >
      <header className="border-b border-line bg-bone shrink-0">
        <div className="max-w-3xl mx-auto px-6 md:px-8 py-5 md:py-6">
          <div className="flex items-start gap-4">
            <div className="min-w-0 flex-1">
              {kicker && (
                <Kicker tone="default" tnum className="block">
                  {kicker}
                </Kicker>
              )}
              <div
                className={cn(
                  "flex items-start justify-between gap-4 flex-wrap",
                  kicker && "mt-2",
                )}
              >
                <h2 className="font-display text-2xl md:text-3xl font-medium tracking-tight leading-[1.1] text-ink min-w-0">
                  {title}
                </h2>
                {rightHeader && <div className="shrink-0">{rightHeader}</div>}
              </div>
              {subtitle && <div className="mt-2">{subtitle}</div>}
            </div>
            <button
              type="button"
              onClick={onClose}
              aria-label="Close"
              className="shrink-0 size-9 rounded-full border border-line text-ink hover:bg-ink hover:text-bone hover:border-ink transition-colors flex items-center justify-center"
            >
              <span aria-hidden="true" className="text-lg leading-none">
                ×
              </span>
            </button>
          </div>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-6 md:px-8 py-8 md:py-10">
          {children}
        </div>
      </div>

      {actions && (
        <footer className="border-t border-line bg-bone shrink-0">
          <div className="max-w-3xl mx-auto px-6 md:px-8 py-4 md:py-5 flex items-center justify-end gap-3 flex-wrap">
            {actions}
          </div>
        </footer>
      )}
    </div>
  );

  return createPortal(content, document.body);
}
