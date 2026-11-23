"use client";

import { useEffect, type ReactNode } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import {
  ADMIN_QUEUES,
  useAdminAttentionTarget,
  useAdminQueueCounts,
  type AdminQueueKey,
} from "@/lib/admin-queues";
import { useAdminKpisQuery } from "@/hooks/use-admin-data";
import { ROUTES } from "@/lib/constants";
import { ADMIN_QUEUE_LINES } from "@/lib/admin-section-guide";

import { AdminKpiStrip } from "@/components/admin/admin-kpi-strip";
import { AdminInboxFilterChips } from "@/components/admin/admin-inbox-filter-chips";
import { AdminSectionIntro } from "@/components/admin/admin-section-intro";
import {
  VerificationsQueue,
} from "@/components/admin/verifications-queue";
import { DisputesQueue } from "@/components/admin/disputes-queue";
import { EventRequestsQueue } from "@/components/admin/event-requests-queue";
import { FlagsQueue } from "@/components/admin/flags-queue";
import { PayoutsQueue } from "@/components/admin/payouts-queue";
import { PayoutReportsSection } from "@/components/admin/payout-reports-section";
import { SellerDetailsQueue } from "@/components/admin/seller-details-queue";
import { RemovalRequestList } from "@/components/admin/photo-removal/request-list";
import { RelevanceCases } from "@/components/admin/relevance/relevance-cases";
import { Kicker } from "@/components/ui/kicker";
import { Skeleton } from "@/components/ui/skeleton";

// One body per queue (a new ADMIN_QUEUES key fails to compile until it has one);
// each list reads its own URL params as on its full page.
const CONTENT: Record<AdminQueueKey, ReactNode> = {
  disputes: <DisputesQueue />,
  flags: <FlagsQueue />,
  verifications: <VerificationsQueue />,
  sellerDetails: <SellerDetailsQueue />,
  eventRequests: <EventRequestsQueue />,
  payouts: <PayoutsQueue />,
  payoutReports: <PayoutReportsSection />,
  relevance: <RelevanceCases defaultTab="cant_tell" />,
  removals: <RemovalRequestList needsActionOnly />,
};

// Phase 2 admin redesign — daily landing with chip-driven queue switch.
// With no ?type, the Inbox waits for the KPI counts, then writes the first
// queue with work into the URL (Verifications when nothing waits or the
// counts failed); once ?type is in the URL a later count change never moves
// the admin. ?type=<inboxType> swaps the body without unmounting the KPI
// strip or header, so deep links work.
//
// /admin redirects here. The dedicated pages (/admin/disputes etc.)
// still exist as focus-mode entries reachable from the rail and render
// the same lifted queue body under their own header.

const VERIFICATIONS = ADMIN_QUEUES.find((q) => q.key === "verifications")!;

export default function AdminInboxPage() {
  // Read straight from the URL (not useUrlState, whose copy lags one render) so the
  // queue written by the replace below mounts first, with no Verifications flash.
  const typeRaw = useSearchParams().get("type");
  const typeInUrl = typeRaw !== null;
  // ADMIN_QUEUES carries the Flags gate, so a gated-off or unknown type falls back to verifications.
  const active = ADMIN_QUEUES.find((q) => q.inboxType === typeRaw) ?? VERIFICATIONS;
  // The same total as the rail's "Queues · N waiting", so the two never disagree.
  const pendingCount = useAdminQueueCounts().total;
  const router = useRouter();
  const { isPending: kpisPending } = useAdminKpisQuery();
  const attention = useAdminAttentionTarget();

  useEffect(() => {
    if (typeInUrl || kpisPending) return;
    const first = ADMIN_QUEUES.find((q) => q.key === attention?.key) ?? VERIFICATIONS;
    router.replace(`${ROUTES.ADMIN_INBOX}?type=${first.inboxType}`, { scroll: false });
  }, [typeInUrl, kpisPending, attention?.key, router]);

  return (
    <>
      <Header pendingCount={pendingCount} />
      <AdminKpiStrip />
      <AdminInboxFilterChips value={typeInUrl ? active.inboxType : ""} />
      {typeInUrl && (
        <p className="font-sans text-sm text-ink-soft mt-4 max-w-xl">{ADMIN_QUEUE_LINES[active.key]}</p>
      )}
      {typeInUrl ? CONTENT[active.key] : <Skeleton className="h-72 w-full rounded-2xl mt-8 md:mt-10" />}
    </>
  );
}

function Header({ pendingCount }: { pendingCount: number | null }) {
  return (
    <header className="pb-8 md:pb-12 border-b border-line">
      <Kicker as="p">
        Inbox{pendingCount !== null && <> · <span className="tnum">{pendingCount}</span> waiting</>}
      </Kicker>
      <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-[1.05] text-ink mt-3">
        Inbox.
      </h1>
      <AdminSectionIntro href={ROUTES.ADMIN_INBOX} />
    </header>
  );
}
