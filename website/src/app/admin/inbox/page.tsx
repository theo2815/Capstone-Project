"use client";

import { useAdminRouteCounts } from "@/components/admin/admin-rail";

import { AdminKpiStrip } from "@/components/admin/admin-kpi-strip";
import {
  AdminInboxFilterChips,
  isInboxQueueType,
  type InboxQueueType,
} from "@/components/admin/admin-inbox-filter-chips";
import {
  VerificationsQueue,
} from "@/components/admin/verifications-queue";
import { DisputesQueue } from "@/components/admin/disputes-queue";
import { EventRequestsQueue } from "@/components/admin/event-requests-queue";
import { FlagsQueue } from "@/components/admin/flags-queue";
import { PayoutsQueue } from "@/components/admin/payouts-queue";
import { RelevanceAlertsInbox } from "@/components/admin/relevance/relevance-alerts-inbox";
import { Kicker } from "@/components/ui/kicker";
import { useUrlState } from "@/hooks/use-url-state";
import { ADMIN_FLAGS_ENABLED } from "@/lib/constants";

// Phase 2 admin redesign — daily landing with chip-driven queue switch.
// Default is verifications (fixed); ?type=disputes|flags|payouts swaps
// the body without unmounting the KPI strip or header. URL state via
// useUrlState so deep-links + back-button work; default omits the query.
//
// /admin redirects here. The dedicated pages (/admin/disputes etc.)
// still exist as focus-mode entries reachable from the rail and render
// the same lifted queue body under their own header.

export default function AdminInboxPage() {
  const [activeRaw, setActive] = useUrlState<string>("type", "verifications");
  const active: InboxQueueType = isInboxQueueType(activeRaw)
    ? activeRaw
    : "verifications";
  // The same total as the rail's "Queues · N open", so the two never disagree.
  const pendingCount = useAdminRouteCounts().queueTotal;

  return (
    <>
      <Header pendingCount={pendingCount} />
      <AdminKpiStrip />
      <AdminInboxFilterChips
        value={active}
        onChange={(next) => setActive(next)}
      />
      {active === "verifications" && <VerificationsQueue />}
      {active === "events" && <EventRequestsQueue />}
      {active === "disputes" && <DisputesQueue />}
      {active === "flags" && ADMIN_FLAGS_ENABLED && <FlagsQueue />}
      {active === "payouts" && <PayoutsQueue />}
      {active === "relevance" && <RelevanceAlertsInbox />}
    </>
  );
}

function Header({ pendingCount }: { pendingCount: number }) {
  return (
    <header className="pb-6 md:pb-8">
      <Kicker as="p">
        Inbox · <span className="tnum">{pendingCount}</span> open
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
