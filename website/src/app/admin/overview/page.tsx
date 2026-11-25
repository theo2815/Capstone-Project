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
import Link from "next/link";

const CAPTION_LINK = "inline-flex min-h-[44px] items-center underline decoration-line-strong underline-offset-4 hover:decoration-fresh";

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
  // The Inbox's count (pending only); an older backend without it falls back to pending plus scheduled.
  const payoutsWaiting = kpis.payoutsAwaitingReview ?? kpis.pendingPayouts;

  return (
    <>
      <Header />

      <Slab titleStyle="heading" id="metrics" number="01" title="Key metrics" caption="Totals right now. The last tile counts this week.">
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
            href={`${ROUTES.ADMIN_INBOX}?type=verifications`}
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
            kicker="Payouts waiting"
            value={payoutsWaiting}
            caption={
              payoutsWaiting === 1
                ? "payout waiting"
                : "payouts waiting"
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
            caption="approved to upload"
            href={ROUTES.ADMIN_PHOTOGRAPHERS}
          />
          {/* FU-18 (U18.2): the count is two lists, so each word opens its own. */}
          <AdminStatTile
            kicker="Suspended accounts"
            value={kpis.suspended}
            caption={
              <>
                <Link href={`${ROUTES.ADMIN_PHOTOGRAPHERS}?status=suspended`} className={CAPTION_LINK}>photographers</Link>
                {" and "}
                <Link href={`${ROUTES.ADMIN_RUNNERS}?status=suspended`} className={CAPTION_LINK}>runners</Link>
              </>
            }
          />
          <AdminStatTile
            kicker="This week's decisions"
            value={kpis.decisionsThisWeek}
            caption={
              kpis.decisionsThisWeek === 1
                ? "action logged since Monday"
                : "actions logged since Monday"
            }
            href="#decisions"
          />
        </div>
        )}
        </div>
      </Slab>

      <Slab
        titleStyle="heading"
        id="trend"
        number="02"
        title="Last 30 days"
        caption="Daily activity"
      >
        <AdminOverviewTrend />
      </Slab>

      <Slab
        titleStyle="heading"
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
