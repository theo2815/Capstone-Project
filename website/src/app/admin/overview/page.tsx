"use client";

import { useMemo } from "react";
import { Slab } from "@/components/profile-shell";
import { AdminStatTile } from "@/components/admin/admin-stat-tile";
import { AdminOverviewTrend } from "@/components/admin/admin-overview-trend";
import { AdminDecisionsTimeline } from "@/components/admin/admin-decisions-timeline";
import { useAdminUserStore } from "@/store/admin-user-store";
import { useAdminUsersData } from "@/lib/admin-users-data";
import {
  useAdminDisputeStore,
  getEffectiveDisputes,
} from "@/store/admin-dispute-store";
import {
  useAdminFlagStore,
  getEffectiveFlags,
} from "@/store/admin-flag-store";
import {
  useAdminPayoutStore,
  mergePayoutsWithOverrides,
} from "@/store/admin-payout-store";
import { useAdminPayouts } from "@/hooks/use-admin-data";
import { useEventCatalog } from "@/lib/event-catalog";
import { type AdminUserRow } from "@/lib/admin-user-registry";
import { ROUTES, ADMIN_FLAGS_ENABLED } from "@/lib/constants";

const SEVEN_DAYS_MS = 7 * 24 * 60 * 60 * 1000;

// Phase 1 admin redesign — /admin/overview is now the weekly-review tab.
// 8-tile KPI grid + 30-day trend + decisions timeline. The daily landing
// moved to /admin/inbox; /admin redirects to /admin/inbox so this page
// is reached only via the rail's last entry or direct URL.
//
// Subscribes to all four stores' overrides + log directly and derives
// counts via useMemo to avoid React 19's useSyncExternalStore Object.is
// trap.

export default function AdminOverviewPage() {
  const { rows: effective } = useAdminUsersData();
  const userLog = useAdminUserStore((s) => s.log);
  const disputeOverrides = useAdminDisputeStore((s) => s.overrides);
  const disputeSubmissions = useAdminDisputeStore((s) => s.submissions);
  const flagOverrides = useAdminFlagStore((s) => s.overrides);
  const payoutOverrides = useAdminPayoutStore((s) => s.overrides);
  // A-1 followup: hydrate payouts from BE instead of the mock seed so the
  // pending-payouts KPI reflects real photographer-submitted requests.
  const serverPayouts = useAdminPayouts() ?? [];
  const catalog = useEventCatalog();

  const kpis = useMemo(() => {
    const photographers = effective.filter((u) => u.role === "PHOTOGRAPHER");
    const pending = photographers.filter(
      (u) => u.verificationStatus === "pending" && u.suspendedAt === null,
    ).length;
    const approved = photographers.filter(
      (u) => u.verificationStatus === "approved" && u.suspendedAt === null,
    ).length;
    const suspended = effective.filter((u) => u.suspendedAt !== null).length;
    const liveEvents = catalog.filter((e) => e.state === "live").length;
    const cutoff = Date.now() - SEVEN_DAYS_MS;
    const weekDecisions = userLog.filter((e) => {
      const t = new Date(e.decidedAt).getTime();
      return Number.isFinite(t) && t >= cutoff;
    }).length;

    const openDisputes = getEffectiveDisputes(
      disputeOverrides,
      disputeSubmissions,
    ).filter((d) => d.status === "open").length;
    const openFlags = ADMIN_FLAGS_ENABLED
      ? getEffectiveFlags(flagOverrides).filter((f) => f.status === "open")
          .length
      : 0;
    const pendingPayouts = mergePayoutsWithOverrides(
      serverPayouts,
      payoutOverrides,
    ).filter((p) => p.status === "pending_review").length;

    return {
      pending,
      approved,
      suspended,
      liveEvents,
      weekDecisions,
      openDisputes,
      openFlags,
      pendingPayouts,
    };
  }, [
    effective,
    catalog,
    userLog,
    disputeOverrides,
    disputeSubmissions,
    flagOverrides,
    payoutOverrides,
    serverPayouts,
  ]);

  return (
    <>
      <Header />

      <Slab id="metrics" number="01" title="Key metrics" caption="Live counts">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 md:gap-8">
          <AdminStatTile
            number="01"
            kicker="Pending verifications"
            value={kpis.pending}
            caption={
              kpis.pending === 1
                ? "photographer waiting"
                : "photographers waiting"
            }
            href={ROUTES.ADMIN_INBOX}
          />
          <AdminStatTile
            number="02"
            kicker="Open disputes"
            value={kpis.openDisputes}
            caption={
              kpis.openDisputes === 1
                ? "refund decision pending"
                : "refund decisions pending"
            }
            href={ROUTES.ADMIN_DISPUTES}
          />
          {ADMIN_FLAGS_ENABLED && (
            <AdminStatTile
              number="03"
              kicker="Open flags"
              value={kpis.openFlags}
              caption={
                kpis.openFlags === 1
                  ? "photo under review"
                  : "photos under review"
              }
              href={ROUTES.ADMIN_FLAGS}
            />
          )}
          <AdminStatTile
            number={ADMIN_FLAGS_ENABLED ? "04" : "03"}
            kicker="Pending payouts"
            value={kpis.pendingPayouts}
            caption={
              kpis.pendingPayouts === 1
                ? "payout awaiting review"
                : "payouts awaiting review"
            }
            href={ROUTES.ADMIN_PAYOUTS}
          />
          <AdminStatTile
            number={ADMIN_FLAGS_ENABLED ? "05" : "04"}
            kicker="Live events"
            value={kpis.liveEvents}
            caption={
              kpis.liveEvents === 1
                ? "race accepting uploads"
                : "races accepting uploads"
            }
            href={ROUTES.ADMIN_EVENTS}
          />
          <AdminStatTile
            number={ADMIN_FLAGS_ENABLED ? "06" : "05"}
            kicker="Approved photographers"
            value={kpis.approved}
            caption="active on platform"
            href={ROUTES.ADMIN_PHOTOGRAPHERS}
          />
          <AdminStatTile
            number={ADMIN_FLAGS_ENABLED ? "07" : "06"}
            kicker="Active suspensions"
            value={kpis.suspended}
            caption={
              kpis.suspended === 1 ? "frozen account" : "frozen accounts"
            }
            href={ROUTES.ADMIN_PHOTOGRAPHERS}
          />
          <AdminStatTile
            number={ADMIN_FLAGS_ENABLED ? "08" : "07"}
            kicker="This week's decisions"
            value={kpis.weekDecisions}
            caption={
              kpis.weekDecisions === 1
                ? "action logged"
                : "actions logged"
            }
          />
        </div>
      </Slab>

      <Slab
        id="trend"
        number="02"
        title="Last 30 days"
        caption="Daily activity"
      >
        <AdminOverviewTrend />
      </Slab>

      <Slab
        id="decisions"
        number="03"
        title="Latest decisions"
        caption="Most recent first"
      >
        <AdminDecisionsTimeline
          limit={10}
          emptyCopy="No decisions yet — approve or reject a photographer in /admin/inbox to start the timeline."
        />
      </Slab>
    </>
  );
}

function Header() {
  return (
    <header className="pb-8 md:pb-12 border-b border-line">
      <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate">
        Overview · Weekly review
      </p>
      <h1 className="font-display text-3xl md:text-4xl font-medium tracking-tight leading-[1.05] text-ink mt-3">
        The week.
      </h1>
      <p className="font-sans text-sm md:text-base text-ink-soft mt-3 max-w-xl">
        Eight metrics across every queue, the platform&apos;s last 30 days,
        and your recent decisions. Open the Inbox for today&apos;s work.
      </p>
    </header>
  );
}
