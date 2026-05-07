"use client";

import {
  PayoutsQueue,
  usePendingPayoutsCount,
  usePendingPayoutsTotal,
} from "@/components/admin/payouts-queue";
import { PayoutReportsSection } from "@/components/admin/payout-reports-section";
import { formatPrice } from "@/lib/utils";

// Phase 2 admin redesign — /admin/payouts is now the focus-mode route
// for the payouts queue, sharing its body (four slabs + bulk-select bar
// + three confirm modals) with /admin/inbox via the lifted
// <PayoutsQueue> component. Header stays page-specific.
//
// Adds a fifth slab below: "Reports from photographers" — the return
// channel populated by the /dashboard/billing File-a-report flow. Lives
// only on /admin/payouts (not /admin/inbox), since the inbox is for
// triage queues, not photographer-filed cases.

export default function AdminPayoutsPage() {
  const pendingCount = usePendingPayoutsCount();
  const totalPending = usePendingPayoutsTotal();

  return (
    <>
      <Header pendingCount={pendingCount} totalPending={totalPending} />
      <PayoutsQueue />
      <PayoutReportsSection />
    </>
  );
}

function Header({
  pendingCount,
  totalPending,
}: {
  pendingCount: number;
  totalPending: number;
}) {
  return (
    <header className="pb-8 md:pb-12 border-b border-line">
      <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate">
        Payouts · <span className="tnum">{pendingCount}</span> cycles pending
        · <span className="tnum">{formatPrice(totalPending)}</span> total
      </p>
      <h1 className="font-display text-3xl md:text-4xl font-medium tracking-tight leading-[1.05] text-ink mt-3">
        Payouts.
      </h1>
      <p className="font-sans text-sm md:text-base text-ink-soft mt-3 max-w-xl">
        Weekly cycle status across all photographers. Approve, hold, or mark
        a cycle paid. Tick checkboxes to act on multiple at once.
      </p>
    </header>
  );
}
