"use client";

import {
  PayoutsQueue,
} from "@/components/admin/payouts-queue";
import { useAdminPayoutCounts, usePendingPayoutsTotal } from "@/hooks/use-admin-payouts";
import { PayoutReportsSection } from "@/components/admin/payout-reports-section";
import { Kicker } from "@/components/ui/kicker";
import { formatPrice } from "@/lib/utils";
import { AdminSectionIntro } from "@/components/admin/admin-section-intro";
import { ROUTES } from "@/lib/constants";

// /admin/payouts hosts the payouts queue (pending → approved → held → paid)
// plus the "Reports from photographers" section. In the request-based flow
// photographers create cycles via /dashboard/billing — admin never generates
// them, so no Generate button here. The legacy BE generator endpoint is
// kept dead-code as a safety hatch but not surfaced in the UI.

export default function AdminPayoutsPage() {
  // AX-30: the server's count, not a filter of one loaded page; none on a backend without the counts route.
  const pendingCount = useAdminPayoutCounts()?.pending_review;
  const totalPending = usePendingPayoutsTotal(pendingCount);

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
  pendingCount: number | undefined;
  totalPending: number | undefined;
}) {
  return (
    <header className="pb-8 md:pb-12 border-b border-line">
      <Kicker as="p">
        Payouts
        {pendingCount !== undefined && <> · <span className="tnum">{pendingCount}</span> waiting</>}
        {totalPending !== undefined && <> · <span className="tnum">{formatPrice(totalPending)}</span> to approve</>}
      </Kicker>
      <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-[1.05] text-ink mt-3">
        Payouts.
      </h1>
      <AdminSectionIntro href={ROUTES.ADMIN_PAYOUTS} />
    </header>
  );
}
