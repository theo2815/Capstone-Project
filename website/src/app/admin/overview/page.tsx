"use client";

import { Slab } from "@/components/profile-shell";
import { Kicker } from "@/components/ui/kicker";
import { AdminStatTile } from "@/components/admin/admin-stat-tile";
import { AdminOverviewTrend } from "@/components/admin/admin-overview-trend";
import { AdminDecisionsTimeline } from "@/components/admin/admin-decisions-timeline";
import { useAdminKpisQuery } from "@/hooks/use-admin-data";
import { Skeleton } from "@/components/ui/skeleton";
import { BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { cn } from "@/lib/utils";
import type { AdminKpis } from "@/lib/api-admin";
import { ROUTES, ADMIN_FLAGS_ENABLED } from "@/lib/constants";
import { AdminSectionIntro } from "@/components/admin/admin-section-intro";

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

// Zero-filled placeholder so the tile expressions type-check; the tiles only
// render once real data is in, so an admin never sees a fake 0.
const EMPTY_KPIS: AdminKpis = {
  pendingVerifications: 0,
  approvedPhotographers: 0,
  suspended: 0,
  liveEvents: 0,
  decisionsThisWeek: 0,
  openDisputes: 0,
  disputesNeedingAdmin: 0,
  disputesOverdue: 0,
  openFlags: 0,
  pendingPayouts: 0,
  pendingEventRequests: 0,
};

// Columns follow the room beside the admin rail, not the window: each step leaves a
// tile 230px or more, so its longest label word ("VERIFICATIONS") ends left of its arrow.
const TILE_GRID = "grid grid-cols-1 @lg:grid-cols-2 @3xl:grid-cols-3 @5xl:grid-cols-4 gap-6 md:gap-8";

export default function AdminOverviewPage() {
  const kpisQuery = useAdminKpisQuery();
  const kpis = kpisQuery.data ?? EMPTY_KPIS;
  const needingAdmin = kpis.disputesNeedingAdmin ?? kpis.openDisputes;
  const overdue = kpis.disputesOverdue ?? 0;

  return (
    <>
      <Header />

      <Slab id="metrics" number="01" title="Key metrics" caption="Live counts">
        <div className="@container">
        {!kpisQuery.data && kpisQuery.isError ? (
          <div role="alert" className="space-y-3">
            <p className="font-sans text-sm text-error">
              Couldn&apos;t load the metrics. Please try again.
            </p>
            <button
              type="button"
              onClick={() => void kpisQuery.refetch()}
              disabled={kpisQuery.isFetching}
              className={cn(BTN_SECONDARY, BTN_SIZE.sm)}
            >
              Retry
            </button>
          </div>
        ) : !kpisQuery.data ? (
          <div className={TILE_GRID} aria-busy="true">
            {Array.from({ length: ADMIN_FLAGS_ENABLED ? 8 : 7 }, (_, i) => (
              <Skeleton key={i} className="h-40 w-full rounded-2xl" />
            ))}
          </div>
        ) : (
        <div className={TILE_GRID}>
          <AdminStatTile
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
            kicker="Disputes needing you"
            value={needingAdmin}
            caption={
              (overdue > 0 ? `${overdue} overdue · ` : "") +
              (needingAdmin === 1
                ? "refund decision pending"
                : "refund decisions pending")
            }
            href={ROUTES.ADMIN_DISPUTES}
          />
          {ADMIN_FLAGS_ENABLED && (
            <AdminStatTile
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
            kicker="Approved photographers"
            value={kpis.approvedPhotographers}
            caption="active on platform"
            href={ROUTES.ADMIN_PHOTOGRAPHERS}
          />
          <AdminStatTile
            kicker="Active suspensions"
            value={kpis.suspended}
            caption={
              kpis.suspended === 1 ? "frozen account" : "frozen accounts"
            }
            href={ROUTES.ADMIN_PHOTOGRAPHERS}
          />
          <AdminStatTile
            kicker="This week's decisions"
            value={kpis.decisionsThisWeek}
            caption={
              kpis.decisionsThisWeek === 1
                ? "action logged"
                : "actions logged"
            }
            href="#decisions"
          />
        </div>
        )}
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
        title="Decision history"
      >
        <AdminDecisionsTimeline recent limit={20} />
      </Slab>
    </>
  );
}

function Header() {
  return (
    <header className="pb-8 md:pb-12 border-b border-line">
      <Kicker as="p">
        Overview · Platform report
      </Kicker>
      <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-[1.05] text-ink mt-3">
        Overview.
      </h1>
      <AdminSectionIntro href={ROUTES.ADMIN_OVERVIEW} />
    </header>
  );
}
