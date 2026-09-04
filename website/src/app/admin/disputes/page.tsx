"use client";

import {
  DisputesQueue,
  useOpenDisputesCount,
  useRefundedThisWeekTotal,
} from "@/components/admin/disputes-queue";
import { Kicker } from "@/components/ui/kicker";
import { formatPrice } from "@/lib/utils";

// Phase 2 admin redesign — /admin/disputes is now the focus-mode route
// for the disputes queue, sharing its body with /admin/inbox via the
// lifted <DisputesQueue> component. Header stays page-specific.

export default function AdminDisputesPage() {
  const openCount = useOpenDisputesCount();
  const refundedThisWeek = useRefundedThisWeekTotal();

  return (
    <>
      <Header openCount={openCount} refundedThisWeek={refundedThisWeek} />
      <DisputesQueue />
    </>
  );
}

function Header({
  openCount,
  refundedThisWeek,
}: {
  openCount: number;
  refundedThisWeek: number;
}) {
  return (
    <header className="pb-8 md:pb-12 border-b border-line">
      <Kicker as="p">
        Disputes · <span className="tnum">{openCount}</span> open ·{" "}
        <span className="tnum">{formatPrice(refundedThisWeek)}</span> refunded
        this week
      </Kicker>
      <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-[1.05] text-ink mt-3">
        Disputes.
      </h1>
      <p className="font-sans text-sm md:text-base text-ink-soft mt-3 max-w-xl">
        Refund requests and complaints from runners. Resolve before they
        escalate.
      </p>
    </header>
  );
}
