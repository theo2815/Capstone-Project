"use client";

import { useId, useState, type ReactNode } from "react";

// Local collapse primitives used by /admin/photographers/[handle] and the
// inbox verifications drawer so admin can scan a long review surface
// without scrolling through every cover/watermark/socials/payouts block.
// Default closed — trailing metadata (counts, status) stays visible so
// the admin still gets at-a-glance state without expanding.

const TOGGLE_CLS =
  "font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate group-hover:text-ink transition-colors tnum shrink-0";

interface CollapsibleReviewSlabProps {
  id?: string;
  number: string;
  title: string;
  caption?: string;
  trailing?: ReactNode;
  defaultOpen?: boolean;
  children: ReactNode;
}

// Slab-shaped wrapper for /admin/photographers/[handle]. Mirrors the Slab
// primitive's kicker rhythm (number · title · caption + right-aligned
// trailing) but folds the body away when collapsed and tightens vertical
// padding so an all-collapsed page reads as a scannable index.
export function CollapsibleReviewSlab({
  id,
  number,
  title,
  caption,
  trailing,
  defaultOpen = false,
  children,
}: CollapsibleReviewSlabProps) {
  const [open, setOpen] = useState(defaultOpen);
  const bodyId = useId();
  return (
    <section
      id={id}
      className={`border-t border-line scroll-mt-24 first:border-0 ${
        open
          ? "py-12 md:py-16 first:pt-10 md:first:pt-20"
          : "py-6 md:py-8 first:pt-8 md:first:pt-12"
      }`}
    >
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-controls={bodyId}
        className={`group w-full flex items-baseline justify-between gap-4 md:gap-6 text-left ${
          open ? "mb-8 md:mb-10" : ""
        }`}
      >
        <div className="flex items-baseline gap-4 min-w-0">
          <span className="font-mono text-[13px] min-[400px]:text-[14px] md:text-[12px] tracking-[0.15em] text-slate-soft tnum">
            {number}
          </span>
          <p className="font-mono uppercase tracking-[0.3em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-ink shrink-0">
            {title}
          </p>
          {caption && (
            <p className="hidden md:block font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft truncate">
              {caption}
            </p>
          )}
        </div>
        <div className="flex items-baseline gap-3 md:gap-4 shrink-0">
          {trailing && (
            <div className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft tnum">
              {trailing}
            </div>
          )}
          <span aria-hidden className={TOGGLE_CLS}>
            {open ? "Hide ↑" : "Show ↓"}
          </span>
        </div>
      </button>
      {open && <div id={bodyId}>{children}</div>}
    </section>
  );
}

interface CollapsibleReviewSectionProps {
  kicker: ReactNode;
  defaultOpen?: boolean;
  children: ReactNode;
}

// Drawer-shaped wrapper for VerificationDetailBody. Lighter chrome than
// CollapsibleReviewSlab — just the mono kicker header + toggle, since the
// drawer body already separates sections with space-y-10. Kicker accepts
// ReactNode so callers can embed tnum spans around counts (e.g.
// "Social & verification links · 2 links").
export function CollapsibleReviewSection({
  kicker,
  defaultOpen = false,
  children,
}: CollapsibleReviewSectionProps) {
  const [open, setOpen] = useState(defaultOpen);
  const bodyId = useId();
  return (
    <section>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-controls={bodyId}
        className={`group w-full flex items-baseline justify-between gap-4 text-left ${
          open ? "mb-3" : ""
        }`}
      >
        <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
          {kicker}
        </p>
        <span aria-hidden className={TOGGLE_CLS}>
          {open ? "Hide ↑" : "Show ↓"}
        </span>
      </button>
      {open && <div id={bodyId}>{children}</div>}
    </section>
  );
}
