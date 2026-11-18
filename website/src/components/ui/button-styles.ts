/**
 * Finish Line button vocabulary — exported class constants, same pattern as
 * KICKER_SIZE_CLASS in kicker.tsx. Compose with cn(): layout classes
 * (w-full, flex-1, mt-*) stay at the call site.
 *
 *   className={cn(BTN_PRIMARY, BTN_SIZE.md, "w-full")}
 *
 * Canonical shapes come from shipped pages: primary from runners/page.tsx
 * hero CTA, secondary from the /orders ink-outline, danger from the
 * confirmation-overlay danger recipe. Disabled is opacity-50 +
 * cursor-not-allowed everywhere — never pointer-events-none (kills tooltips).
 */

export const BTN_PRIMARY =
  "inline-flex items-center justify-center gap-2 rounded-full bg-fresh font-display font-bold text-surface transition-colors hover:bg-fresh-deep disabled:opacity-50 disabled:cursor-not-allowed";

export const BTN_SECONDARY =
  "inline-flex items-center justify-center gap-2 rounded-full border border-ink font-display font-bold text-ink transition-colors hover:bg-ink hover:text-surface disabled:opacity-50 disabled:cursor-not-allowed";

export const BTN_DANGER =
  "inline-flex items-center justify-center gap-2 rounded-full border border-line bg-bone-deep font-display font-bold text-error transition-colors hover:bg-line disabled:opacity-50 disabled:cursor-not-allowed";

export const BTN_GHOST =
  "inline-flex items-center justify-center gap-2 rounded-full font-display font-bold text-slate transition-colors hover:text-ink disabled:opacity-50 disabled:cursor-not-allowed";

export const BTN_SIZE = {
  /** Page-level CTAs (canonical hero pill). */
  md: "px-6 py-3.5 text-[16px]",
  /** Dense contexts: admin asides, bulk bars, inline rows. */
  sm: "px-5 py-2.5 text-sm min-h-[44px]",
} as const;

/** Title button of a row that opens a drawer (AX-4): its ::after covers the
 *  `relative` row so a click anywhere opens it, and the focus outline rings
 *  the whole row. Inner controls take `relative z-10` to stay on top. */
export const ROW_OPEN_BUTTON =
  "cursor-pointer text-left outline-none after:absolute after:inset-0 after:content-[''] focus-visible:after:outline-2 focus-visible:after:-outline-offset-2 focus-visible:after:outline-fresh";

/** Pill for dark (pine) surfaces, e.g. the footer CTA band. */
export const BTN_ON_DARK =
  "inline-flex items-center justify-center gap-2 rounded-full bg-paper font-display font-bold text-pine transition-colors hover:bg-fresh-tint disabled:opacity-50 disabled:cursor-not-allowed";
