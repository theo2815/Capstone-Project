"use client";

import { useAdminLegendStore } from "@/store/admin-legend-store";

// Keyboard-legend hint in the admin rail's footer (it used to float
// bottom-right and covered row content at the page edge). Click to open the
// keyboard legend; pressing `?` from anywhere outside an input does the same.
// Hidden below md because the keyboard layer is desktop-only, like the rail.
// The key-cap look, shared by the drawer buttons that answer a key (FT-36).
export const KBD_CLASS = "font-mono text-[12px] text-ink rounded border border-line bg-bone-deep px-1.5 py-0.5";

export function KeyboardHintPill() {
  const setOpen = useAdminLegendStore((s) => s.setOpen);
  return (
    <button
      type="button"
      onClick={() => setOpen(true)}
      aria-label="Keyboard shortcuts (press ?)"
      className="hidden md:inline-flex items-center gap-2 rounded-full border border-line bg-bone hover:border-ink transition-colors px-3 py-1.5 group"
    >
      <span className="font-mono text-[14px] text-slate group-hover:text-ink transition-colors">
        Shortcuts
      </span>
      <kbd className={KBD_CLASS}>
        ?
      </kbd>
    </button>
  );
}
