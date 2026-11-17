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
          Each case is one photographer&apos;s batch in one event, checked after upload for junk, wrong-event and
          copied photos. A case marked &ldquo;would have hidden&rdquo; was only recorded: nothing was hidden.
        </p>
      </header>
      <RelevanceCases />
    </>
  );
}
