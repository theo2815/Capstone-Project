"use client";

import { useId, useState, type ReactNode } from "react";
import { Kicker, KICKER_SIZE_CLASS } from "@/components/ui/kicker";

// Local collapse primitives used by /admin/photographers/[handle] and the
// inbox verifications drawer so admin can scan a long review surface
// without scrolling through every cover/watermark/socials/payouts block.
// Default closed — trailing metadata (counts, status) stays visible so
// the admin still gets at-a-glance state without expanding.

const TOGGLE_CLS =
  `font-mono uppercase ${KICKER_SIZE_CLASS.sm} text-slate group-hover:text-ink transition-colors tnum shrink-0`;

// The word is part of the button's name (FU-21); only the arrow is hidden from screen readers.
function ToggleWord({ open }: { open: boolean }) {
  return (
    <span className={TOGGLE_CLS}>
      {open ? "Hide" : "Show"} <span aria-hidden>{open ? "↑" : "↓"}</span>
    </span>
  );
}

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
        aria-controls={open ? bodyId : undefined}
        className={`group w-full min-h-11 flex flex-wrap content-center items-baseline justify-between gap-4 md:gap-6 text-left ${
          open ? "mb-8 md:mb-10" : ""
        }`}
      >
        <div className="flex flex-wrap items-baseline gap-4 min-w-0">
          <span className="font-mono text-[14px] min-[400px]:text-[15px] md:text-[13px] tracking-[0.15em] text-slate-soft tnum">
            {number}
          </span>
          <Kicker as="p" className="text-ink break-words min-w-0">
            {title}
          </Kicker>
          {caption && (
            <Kicker as="p" tone="soft" className="break-words min-w-0">
              {caption}
            </Kicker>
          )}
        </div>
        <div className="ml-auto flex items-baseline gap-3 md:gap-4 min-w-0">
          {/* A string trailer is a count or status (capitals); a node renders as given, e.g. a handle read as typed. */}
          {typeof trailing === "string" ? (
            <Kicker as="div" tone="soft" className="tnum">
              {trailing}
            </Kicker>
          ) : trailing}
          <ToggleWord open={open} />
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
        aria-controls={open ? bodyId : undefined}
        className={`group w-full min-h-11 flex items-center justify-between gap-4 text-left ${
          open ? "mb-3" : ""
        }`}
      >
        <Kicker as="p" tone="soft">
          {kicker}
        </Kicker>
        <ToggleWord open={open} />
      </button>
      {open && <div id={bodyId}>{children}</div>}
    </section>
  );
}
