"use client";

import { RelevanceCases } from "@/components/admin/relevance/relevance-cases";
import { Kicker } from "@/components/ui/kicker";

// RV-1.11a: the Photo Relevance page, read-only. Actions arrive with part 2; the rail item with RV-1.11b.
export default function AdminRelevancePage() {
  return (
    <>
      <header className="pb-8 md:pb-12 border-b border-line">
        <Kicker as="p">Photo Relevance</Kicker>
        <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-[1.05] text-ink mt-3">
          Photo Relevance.
        </h1>
        <p className="font-sans text-sm md:text-base text-ink-soft mt-3 max-w-xl">
          Uploaded batches checked for junk, wrong-event and copied photos. &ldquo;Would have hidden&rdquo;
          means recorded only; nothing was hidden.
        </p>
      </header>
      <RelevanceCases />
    </>
  );
}
