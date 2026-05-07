"use client";

import { createPortal } from "react-dom";
import type { SectionGuideEntry } from "@/lib/section-guide";

// Shared per-rail-row contextual tooltip. Portals to document.body so it
// escapes the rail aside's `md:overflow-y-auto` (which would otherwise
// clip the popover horizontally — see ui-pitfalls.md). Anchor is the
// label `<span>` inside the rail row, NOT the row's right edge — labels
// are short and the popover sits 12 px to the right of the label,
// vertically centered.

export function RailTipPortal({
  anchor,
  guide,
  tipId,
}: {
  anchor: HTMLElement;
  guide: SectionGuideEntry;
  tipId: string;
}) {
  const rect = anchor.getBoundingClientRect();
  const TIP_HEIGHT_APPROX = 96;
  return createPortal(
    <div
      id={tipId}
      role="tooltip"
      style={{
        position: "fixed",
        top: rect.top + rect.height / 2 - TIP_HEIGHT_APPROX / 2,
        left: rect.right + 12,
        width: 240,
        zIndex: 50,
      }}
      className="border border-line rounded-2xl bg-bone shadow-lg p-4"
    >
      <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft tnum">
        {guide.number} · {guide.label}
      </p>
      <p className="font-sans text-sm text-ink-soft leading-relaxed mt-2">
        {guide.body}
      </p>
    </div>,
    document.body,
  );
}
