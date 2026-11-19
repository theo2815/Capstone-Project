"use client";

import type { EventDetail } from "@/types/event";

// Shown when a bib filter is active but no photos match. Status-aware copy
// (DRAFT/ACTIVE → "not here yet"; COMPLETED/ARCHIVED → "not found" /
// "wrapped"). There is deliberately no email form: nothing on the backend
// stores a bib alert, so offering one would promise an email that never comes.
export function BibEmptyResult({
  event,
  bib,
  onClear,
  ctaLabel = "Or skim the full gallery →",
}: {
  event: EventDetail;
  bib: string;
  onClear: () => void;
  /** Override copy for the secondary CTA, e.g. on per-photographer galleries. */
  ctaLabel?: string;
}) {
  const { title, body } = emptyResultCopy(event.status, bib);

  return (
    <div className="px-6 md:px-10 py-16 md:py-24">
      <div className="max-w-2xl mx-auto rounded-2xl bg-bone-deep border border-line p-8 md:p-12">
        <p className="font-mono uppercase tracking-[0.14em] text-[10px] text-slate mb-4">
          BIB {bib} · {event.name}
        </p>
        <p className="font-display text-3xl md:text-4xl font-medium text-ink tracking-tight leading-tight">
          {title}
        </p>
        <p className="mt-4 font-sans text-base md:text-lg text-ink-soft max-w-md">
          {body}
        </p>
        <button
          type="button"
          onClick={onClear}
          className="mt-6 font-mono uppercase tracking-[0.14em] text-[10px] text-slate hover:text-fresh transition-colors"
        >
          {ctaLabel}
        </button>
      </div>
    </div>
  );
}

function emptyResultCopy(
  status: EventDetail["status"],
  bib: string,
): { title: string; body: string } {
  switch (status) {
    case "DRAFT":
    case "ACTIVE":
      return {
        title: `No photos with bib ${bib} yet.`,
        body: "Double-check the number. Photographers may still be adding photos, and a bib that's covered or blurred can't be read — so check back later, or skim the gallery.",
      };
    case "COMPLETED":
      return {
        title: "Bib not found.",
        body: `All photos for this race have been uploaded — ${bib} isn't in there. Double-check the number, or skim the wall.`,
      };
    case "ARCHIVED":
      return {
        title: "This race has wrapped.",
        body: `Photos for ${bib} never landed in this archive. The wall's still here if you want to skim.`,
      };
    default: {
      const _exhaustive: never = status;
      return _exhaustive;
    }
  }
}
