"use client";

import { RelevanceCases } from "@/components/admin/relevance/relevance-cases";
import { Kicker } from "@/components/ui/kicker";
import { AdminSectionIntro } from "@/components/admin/admin-section-intro";
import { ROUTES } from "@/lib/constants";
import { useAdminQueueCounts } from "@/lib/admin-queues";

// The Photo Relevance page: the cases and appeals with their actions (RelevanceCases).
export default function AdminRelevancePage() {
  const waiting = useAdminQueueCounts().byKey.relevance;
  return (
    <>
      <header className="pb-8 md:pb-12 border-b border-line">
        <Kicker as="p">Photo Relevance{waiting !== null && <> · <span className="tnum">{waiting}</span> waiting</>}</Kicker>
        <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-[1.05] text-ink mt-3">
          Photo Relevance.
        </h1>
        <AdminSectionIntro href={ROUTES.ADMIN_RELEVANCE} />
      </header>
      <RelevanceCases />
    </>
  );
}
