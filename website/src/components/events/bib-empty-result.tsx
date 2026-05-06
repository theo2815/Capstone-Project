"use client";

import { useState } from "react";
import type { EventDetail } from "@/types/event";

// Shown when a bib filter is active but no photos match. Status-aware copy
// (DRAFT/ACTIVE → "still uploading" + notify form; COMPLETED/ARCHIVED →
// "not found" / "wrapped"). The notify form is local-state only — backend
// wiring lands when the events service ships.
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
  const { title, body, showNotify } = emptyResultCopy(event.status, bib);

  return (
    <div className="px-6 md:px-10 py-16 md:py-24">
      <div className="max-w-2xl mx-auto rounded-2xl bg-bone-deep border border-line p-8 md:p-12">
        <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate mb-4">
          BIB {bib} · {event.name}
        </p>
        <p className="font-display text-3xl md:text-4xl font-medium text-ink tracking-tight leading-tight">
          {title}
        </p>
        <p className="mt-4 font-sans text-base md:text-lg text-ink-soft max-w-md">
          {body}
        </p>
        {showNotify && <NotifyForm bib={bib} />}
        <button
          type="button"
          onClick={onClear}
          className="mt-6 font-mono uppercase tracking-[0.25em] text-[10px] text-slate hover:text-fresh transition-colors"
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
): { title: string; body: string; showNotify: boolean } {
  switch (status) {
    case "DRAFT":
    case "ACTIVE":
      return {
        title: "Still uploading.",
        body: `Photographers are still working through this race. Drop your email and we'll ping you the moment ${bib} appears.`,
        showNotify: true,
      };
    case "COMPLETED":
      return {
        title: "Bib not found.",
        body: `All photos for this race have been uploaded — ${bib} isn't in there. Double-check the number, or skim the wall.`,
        showNotify: false,
      };
    case "ARCHIVED":
      return {
        title: "This race has wrapped.",
        body: `Photos for ${bib} never landed in this archive. The wall's still here if you want to skim.`,
        showNotify: false,
      };
    default: {
      const _exhaustive: never = status;
      return _exhaustive;
    }
  }
}

function NotifyForm({ bib }: { bib: string }) {
  const [email, setEmail] = useState("");
  const [submitted, setSubmitted] = useState(false);
  if (submitted) {
    return (
      <p className="mt-8 font-mono uppercase tracking-[0.25em] text-[10px] text-fresh">
        ✓ We&apos;ll email you when bib {bib} appears.
      </p>
    );
  }
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        if (email.trim()) setSubmitted(true);
      }}
      className="mt-8 flex flex-col sm:flex-row items-stretch sm:items-end gap-3 max-w-md"
    >
      <label className="flex-1">
        <span className="font-mono uppercase tracking-[0.25em] text-[10px] text-slate block mb-2">
          Notify me
        </span>
        <input
          type="email"
          required
          placeholder="you@email.com"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="block w-full border-b border-line bg-transparent focus:border-fresh outline-none font-sans text-base py-2 placeholder:text-slate-soft text-ink"
        />
      </label>
      <button
        type="submit"
        className="bg-ink hover:bg-ink-soft text-bone px-5 py-3 rounded-full font-mono uppercase tracking-[0.2em] text-[13px] min-[400px]:text-[14px] md:text-[12px] transition-colors whitespace-nowrap"
      >
        Notify me →
      </button>
    </form>
  );
}
