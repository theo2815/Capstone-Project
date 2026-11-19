"use client";

import type { ReactNode } from "react";
import { ADMIN_QUEUES, useAdminQueueCounts, type AdminQueueKey } from "@/lib/admin-queues";

import { AdminKpiStrip } from "@/components/admin/admin-kpi-strip";
import { AdminInboxFilterChips } from "@/components/admin/admin-inbox-filter-chips";
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
import { useUrlState } from "@/hooks/use-url-state";

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
// Default is verifications (fixed); ?type=<inboxType> swaps
// the body without unmounting the KPI strip or header, so deep links work.
//
// /admin redirects here. The dedicated pages (/admin/disputes etc.)
// still exist as focus-mode entries reachable from the rail and render
// the same lifted queue body under their own header.

export default function AdminInboxPage() {
  const [activeRaw] = useUrlState<string>("type", "verifications");
  // ADMIN_QUEUES carries the Flags gate, so a gated-off or unknown type falls back to verifications.
  const active =
    ADMIN_QUEUES.find((q) => q.inboxType === activeRaw) ??
    ADMIN_QUEUES.find((q) => q.key === "verifications")!;
  // The same total as the rail's "Queues · N open", so the two never disagree.
  const pendingCount = useAdminQueueCounts().total;

  return (
    <>
      <Header pendingCount={pendingCount} />
      <AdminKpiStrip />
      <AdminInboxFilterChips value={active.inboxType} />
      {CONTENT[active.key]}
    </>
  );
}

function Header({ pendingCount }: { pendingCount: number | null }) {
  return (
    <header className="pb-8 md:pb-12 border-b border-line">
      <Kicker as="p">
        Inbox{pendingCount !== null && <> · <span className="tnum">{pendingCount}</span> open</>}
      </Kicker>
      <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-[1.05] text-ink mt-3">
        Today.
      </h1>
      {/* Hidden on phones so the queue starts on the first screen. */}
      <p className="hidden md:block font-sans text-sm md:text-base text-ink-soft mt-3 max-w-xl">
        Everything waiting on an admin today, one queue at a time.
      </p>
    </header>
  );
}
