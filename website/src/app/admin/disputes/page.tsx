"use client";

import { useRefundedThisWeekTotal } from "@/hooks/use-admin-disputes";
import { useAdminKpis } from "@/hooks/use-admin-data";
import { DisputesQueue } from "@/components/admin/disputes-queue";
import { Kicker } from "@/components/ui/kicker";
import { formatPrice } from "@/lib/utils";
import { AdminSectionIntro } from "@/components/admin/admin-section-intro";
import { ROUTES } from "@/lib/constants";

// Phase 2 admin redesign — /admin/disputes is now the focus-mode route
// for the disputes queue, sharing its body with /admin/inbox via the
// lifted <DisputesQueue> component. Header stays page-specific.

export default function AdminDisputesPage() {
  const kpis = useAdminKpis();
  const needCount = kpis?.disputesNeedingAdmin ?? kpis?.openDisputes ?? 0;
  const overdueCount = kpis?.disputesOverdue ?? 0;
  const refundedThisWeek = useRefundedThisWeekTotal();

  return (
    <>
      <Header
        needCount={needCount}
        overdueCount={overdueCount}
        refundedThisWeek={refundedThisWeek} />
      <DisputesQueue />
    </>
  );
}

function Header({
  needCount,
  overdueCount,
  refundedThisWeek,
}: {
  needCount: number;
  overdueCount: number;
  refundedThisWeek: number | undefined;
}) {
  return (
    <header className="pb-8 md:pb-12 border-b border-line">
      <Kicker as="p">
        Disputes · <span className="tnum">{needCount}</span> need you ·{" "}
        <span className="tnum">{overdueCount}</span> overdue ·{" "}
        <span className="tnum">{refundedThisWeek === undefined ? "—" : formatPrice(refundedThisWeek)}</span> refunded
        this week
      </Kicker>
      <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-[1.05] text-ink mt-3">
        Disputes.
      </h1>
      <AdminSectionIntro href={ROUTES.ADMIN_DISPUTES} />
    </header>
  );
}
