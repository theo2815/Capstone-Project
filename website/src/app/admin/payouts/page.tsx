"use client";

import {
  PayoutsQueue,
  usePendingPayoutsCount,
  usePendingPayoutsTotal,
} from "@/components/admin/payouts-queue";
import { PayoutReportsSection } from "@/components/admin/payout-reports-section";
import { formatPrice } from "@/lib/utils";

// /admin/payouts hosts the payouts queue (pending → approved → held → paid)
// plus the "Reports from photographers" section. In the request-based flow
// photographers create cycles via /dashboard/billing — admin never generates
// them, so no Generate button here. The legacy BE generator endpoint is
// kept dead-code as a safety hatch but not surfaced in the UI.

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
        Payouts · <span className="tnum">{pendingCount}</span> awaiting review
        · <span className="tnum">{formatPrice(totalPending)}</span> total
      </p>
      <h1 className="font-display text-3xl md:text-4xl font-medium tracking-tight leading-[1.05] text-ink mt-3">
        Payouts.
      </h1>
      <p className="font-sans text-sm md:text-base text-ink-soft mt-3 max-w-xl">
        Photographer-requested payouts. Approve, hold with a reason, or mark a
        cycle paid after transferring manually. Tick checkboxes to act on
        multiple at once.
      </p>
    </header>
  );
}
