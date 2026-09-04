"use client";

import { AdminKpiStrip } from "@/components/admin/admin-kpi-strip";
import {
  AdminInboxFilterChips,
  isInboxQueueType,
  type InboxQueueType,
} from "@/components/admin/admin-inbox-filter-chips";
import {
  VerificationsQueue,
  usePendingVerificationsCount,
} from "@/components/admin/verifications-queue";
import { DisputesQueue } from "@/components/admin/disputes-queue";
import { EventRequestsQueue } from "@/components/admin/event-requests-queue";
import { FlagsQueue } from "@/components/admin/flags-queue";
import { PayoutsQueue } from "@/components/admin/payouts-queue";
import { Kicker } from "@/components/ui/kicker";
import { useAdminKpis } from "@/hooks/use-admin-data";
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
  // "Waiting" = the two queues that block a photographer: verifications +
  // event requests (V46). Disputes/payouts keep their own KPI-strip counts.
  const pendingCount =
    usePendingVerificationsCount() +
    (useAdminKpis()?.pendingEventRequests ?? 0);

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
    </>
  );
}

function Header({ pendingCount }: { pendingCount: number }) {
  return (
    <header className="pb-6 md:pb-8">
      <Kicker as="p">
        Inbox · <span className="tnum">{pendingCount}</span> waiting
      </Kicker>
      <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-[1.05] text-ink mt-3">
        Today.
      </h1>
      <p className="font-sans text-sm md:text-base text-ink-soft mt-3 max-w-xl">
        {ADMIN_FLAGS_ENABLED
          ? "Switch between Verifications, Event requests, Disputes, Flags, and Payouts with the chips below. Pick a metric to drill in, or use the rail for focus mode on a single queue."
          : "Switch between Verifications, Event requests, Disputes, and Payouts with the chips below. Pick a metric to drill in, or use the rail for focus mode on a single queue."}
      </p>
    </header>
  );
}
