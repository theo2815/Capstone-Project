"use client";

import { useAdminLegendStore } from "@/store/admin-legend-store";

// Bottom-right floating hint that lives on every /admin/* page. Click to
// open the keyboard legend; pressing `?` from anywhere outside an input
// does the same thing. Hidden below md because the keyboard layer is a
// desktop-only affordance — phone keyboards don't make `?` reachable.
//
// z-40 sits above queue content but below the AdminDetailDrawer (z-50)
// and the legend modal itself (z-60), so the pill auto-hides while the
// user is in a focus surface. The Quiet Studio look — bone background,
// line border, mono `?` keycap — matches the rest of the admin chrome.
export function KeyboardHintPill() {
  const setOpen = useAdminLegendStore((s) => s.setOpen);
  return (
    <button
      type="button"
      onClick={() => setOpen(true)}
      aria-label="Keyboard shortcuts (press ?)"
      className="hidden md:inline-flex fixed bottom-5 right-5 z-40 items-center gap-2 rounded-full border border-line bg-bone hover:border-ink transition-colors px-3 py-1.5 shadow-sm group"
    >
      <span
        aria-hidden="true"
        className="font-mono text-[14px] text-slate group-hover:text-ink transition-colors"
      >
        ⌨
      </span>
      <kbd className="font-mono text-[11px] text-ink rounded border border-line bg-bone-deep px-1.5 py-0.5">
        ?
      </kbd>
    </button>
  );
}
