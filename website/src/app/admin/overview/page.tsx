"use client";

import { Slab } from "@/components/profile-shell";
import { Kicker } from "@/components/ui/kicker";
import { AdminStatTile } from "@/components/admin/admin-stat-tile";
import { AdminOverviewTrend } from "@/components/admin/admin-overview-trend";
import { AdminDecisionsTimeline } from "@/components/admin/admin-decisions-timeline";
import { useAdminKpis } from "@/hooks/use-admin-data";
import type { AdminKpis } from "@/lib/api-admin";
import { ROUTES, ADMIN_FLAGS_ENABLED } from "@/lib/constants";

// Phase 1 admin redesign — /admin/overview is now the weekly-review tab.
// 8-tile KPI grid + 30-day trend + decisions timeline. The daily landing
// moved to /admin/inbox; /admin redirects to /admin/inbox so this page
// is reached only via the rail's last entry or direct URL.
//
// All eight tiles read GET /admin/kpis — the server counts whole tables.
// The old client-side derivation counted over capped fetches (200 users /
// 50 payouts), so tiles saturated at the caps, and "Live events" read the
// retired event-catalog seed — structurally 0 forever. /admin/inbox's KPI
// strip reads the same ["admin","kpis"] key (so the two surfaces agree),
// and useAdminQueueRealtime already invalidates it on its tick.

// Zero-filled placeholder while the KPI query loads — the tiles render 0,
// the same first paint the old derivation produced from empty arrays.
const EMPTY_KPIS: AdminKpis = {
  pendingVerifications: 0,
  approvedPhotographers: 0,
  suspended: 0,
  liveEvents: 0,
  decisionsThisWeek: 0,
  openDisputes: 0,
  openFlags: 0,
  pendingPayouts: 0,
  pendingEventRequests: 0,
};

export default function AdminOverviewPage() {
  const kpis = useAdminKpis() ?? EMPTY_KPIS;

  return (
    <>
      <Header />

      <Slab id="metrics" number="01" title="Key metrics" caption="Live counts">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 md:gap-8">
          <AdminStatTile
            number="01"
            kicker="Pending verifications"
            value={kpis.pendingVerifications}
            caption={
              kpis.pendingVerifications === 1
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
            value={kpis.approvedPhotographers}
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
            value={kpis.decisionsThisWeek}
            caption={
              kpis.decisionsThisWeek === 1
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
      <Kicker as="p">
        Overview · Weekly review
      </Kicker>
      <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-[1.05] text-ink mt-3">
        The week.
      </h1>
      <p className="font-sans text-sm md:text-base text-ink-soft mt-3 max-w-xl">
        Eight metrics across every queue, the platform&apos;s last 30 days,
        and your recent decisions. Open the Inbox for today&apos;s work.
      </p>
    </header>
  );
}
