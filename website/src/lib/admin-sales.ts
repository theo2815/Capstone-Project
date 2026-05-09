"use client";

import { useMemo } from "react";
import {
  PHOTO_PRICE_PHP,
  splitGmv,
} from "@/lib/platform-economics";
import { useEventCatalog } from "@/lib/event-catalog";
import {
  useAdminPayoutStore,
  getEffectivePayouts,
} from "@/store/admin-payout-store";
import {
  useAdminDisputeStore,
  getEffectiveDisputes,
} from "@/store/admin-dispute-store";
import {
  useAdminSalesKpisLive,
  useAdminSalesByEventLive,
} from "@/hooks/use-admin-data";

// Admin sales — derivation hooks. Pure read-side. Sources in mock mode:
//   - ADMIN_PAYOUT_SEED (via store overrides)  → cycle-level itemCount × PRICE
//   - ADMIN_DISPUTES (via store overrides + submissions) → refund $
//   - EVENT_CATALOG (via admin-event-overrides store) → events with sales
//
// In live mode (NEXT_PUBLIC_BACKEND_LIVE=true), `useSalesKpis` and
// `useSalesByEvent` prefer server data from `/admin/sales/*`; the derivation
// stays as fallback when the server query is still pending.
//
// Caveat: orders mock has no photographer attribution, so events are scored
// via "implied GMV" = photoCount × PHOTO_PRICE_PHP. The page surfaces this
// as an Implied badge until the backend lands order-level event attribution.

export interface SalesKpis {
  gmv: number;
  platformRevenue: number;
  refundsIssued: number;
  netPlatformRevenue: number;
  photographerKeep: number;
  totalSalesCount: number;
}

export function useSalesKpis(): SalesKpis {
  const payoutOverrides = useAdminPayoutStore((s) => s.overrides);
  const disputeOverrides = useAdminDisputeStore((s) => s.overrides);
  const disputeSubmissions = useAdminDisputeStore((s) => s.submissions);

  const derived = useMemo(() => {
    const cycles = getEffectivePayouts(payoutOverrides);
    const totalSalesCount = cycles.reduce((s, c) => s + c.itemCount, 0);
    const gmv = totalSalesCount * PHOTO_PRICE_PHP;
    const split = splitGmv(gmv);

    const disputes = getEffectiveDisputes(disputeOverrides, disputeSubmissions);
    const refundsIssued = disputes.reduce(
      (s, d) => s + (d.refundAmount ?? 0),
      0,
    );
    const netPlatformRevenue = Math.max(0, split.cut - refundsIssued);

    return {
      gmv,
      platformRevenue: split.cut,
      refundsIssued,
      netPlatformRevenue,
      photographerKeep: split.keep,
      totalSalesCount,
    };
  }, [payoutOverrides, disputeOverrides, disputeSubmissions]);

  // Live mode: prefer server data when present, fall back to derivation while
  // the query is loading or in mock mode.
  const live = useAdminSalesKpisLive("ytd");
  return live ?? derived;
}

export interface WeeklyGmvPoint {
  /** ISO Monday for the week (YYYY-MM-DD). */
  weekOf: string;
  gmv: number;
}

export const SALES_TREND_WEEKS = 12;

function pad2(n: number): string {
  return n.toString().padStart(2, "0");
}

function formatLocalIso(d: Date): string {
  return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`;
}

function isoMondayOf(d: Date): string {
  // JS day: 0 = Sun, 1 = Mon, …, 6 = Sat. Map so Monday → 0 offset.
  const offset = (d.getDay() + 6) % 7;
  const monday = new Date(d.getFullYear(), d.getMonth(), d.getDate() - offset);
  return formatLocalIso(monday);
}

/** Last `weeks` Mondays inclusive of the current week, ordered ascending. */
export function useWeeklyGmvSeries(
  weeks: number = SALES_TREND_WEEKS,
): WeeklyGmvPoint[] {
  const payoutOverrides = useAdminPayoutStore((s) => s.overrides);

  return useMemo(() => {
    const cycles = getEffectivePayouts(payoutOverrides);
    const byWeek = new Map<string, number>();
    for (const cycle of cycles) {
      const gmv = cycle.itemCount * PHOTO_PRICE_PHP;
      byWeek.set(cycle.weekOf, (byWeek.get(cycle.weekOf) ?? 0) + gmv);
    }

    const currentMondayIso = isoMondayOf(new Date());
    const baseMonday = new Date(`${currentMondayIso}T00:00:00`);
    const points: WeeklyGmvPoint[] = [];
    for (let i = weeks - 1; i >= 0; i--) {
      const monday = new Date(baseMonday);
      monday.setDate(monday.getDate() - i * 7);
      const weekOf = formatLocalIso(monday);
      points.push({ weekOf, gmv: byWeek.get(weekOf) ?? 0 });
    }
    return points;
  }, [payoutOverrides, weeks]);
}

export interface TopPhotographer {
  photographerId: string;
  photographerName: string;
  brandName: string | null;
  handle: string | null;
  gmv: number;
  photosSold: number;
  cycles: number;
}

export function useTopPhotographers(limit: number = 10): TopPhotographer[] {
  const payoutOverrides = useAdminPayoutStore((s) => s.overrides);

  return useMemo(() => {
    const cycles = getEffectivePayouts(payoutOverrides);
    const byPhotographer = new Map<string, TopPhotographer>();
    for (const cycle of cycles) {
      const gmv = cycle.itemCount * PHOTO_PRICE_PHP;
      const existing = byPhotographer.get(cycle.photographerId);
      if (existing) {
        existing.gmv += gmv;
        existing.photosSold += cycle.itemCount;
        existing.cycles += 1;
      } else {
        byPhotographer.set(cycle.photographerId, {
          photographerId: cycle.photographerId,
          photographerName: cycle.photographerName,
          brandName: cycle.brandName,
          handle: cycle.handle,
          gmv,
          photosSold: cycle.itemCount,
          cycles: 1,
        });
      }
    }
    return Array.from(byPhotographer.values())
      .sort((a, b) => b.gmv - a.gmv)
      .slice(0, limit);
  }, [payoutOverrides, limit]);
}

export interface SalesEventRow {
  id: string;
  slug: string;
  name: string;
  date: string;
  city: string;
  status: string;
  state: string;
  photoCount: number;
  impliedGmv: number;
  impliedCut: number;
  refundsIssued: number;
}

export function useSalesByEvent(): SalesEventRow[] {
  const catalog = useEventCatalog();
  const disputeOverrides = useAdminDisputeStore((s) => s.overrides);
  const disputeSubmissions = useAdminDisputeStore((s) => s.submissions);

  const derived = useMemo(() => {
    const disputes = getEffectiveDisputes(disputeOverrides, disputeSubmissions);
    const refundsByEvent = new Map<string, number>();
    for (const d of disputes) {
      if (d.refundAmount === null || d.refundAmount === undefined) continue;
      refundsByEvent.set(
        d.eventId,
        (refundsByEvent.get(d.eventId) ?? 0) + d.refundAmount,
      );
    }

    return catalog
      .filter((e) => e.photoCount > 0)
      .map<SalesEventRow>((e) => {
        const impliedGmv = e.photoCount * PHOTO_PRICE_PHP;
        const split = splitGmv(impliedGmv);
        return {
          id: e.id,
          slug: e.slug,
          name: e.name,
          date: e.date,
          city: e.city,
          status: e.status,
          state: e.state,
          photoCount: e.photoCount,
          impliedGmv,
          impliedCut: split.cut,
          refundsIssued: refundsByEvent.get(e.id) ?? 0,
        };
      });
  }, [catalog, disputeOverrides, disputeSubmissions]);

  // Live mode: prefer server data when present, fall back to derivation.
  const live = useAdminSalesByEventLive();
  return live ?? derived;
}

export function useTopEvents(limit: number = 10): SalesEventRow[] {
  const all = useSalesByEvent();
  return useMemo(
    () => [...all].sort((a, b) => b.impliedGmv - a.impliedGmv).slice(0, limit),
    [all, limit],
  );
}

export interface RefundPulse {
  refundsIssued: number;
  refundCount: number;
  refundRatePct: number; // % of GMV refunded; 0 when GMV = 0
  openCount: number;
  escalatedCount: number;
}

export function useRefundPulse(): RefundPulse {
  const payoutOverrides = useAdminPayoutStore((s) => s.overrides);
  const disputeOverrides = useAdminDisputeStore((s) => s.overrides);
  const disputeSubmissions = useAdminDisputeStore((s) => s.submissions);

  return useMemo(() => {
    const cycles = getEffectivePayouts(payoutOverrides);
    const gmv =
      cycles.reduce((s, c) => s + c.itemCount, 0) * PHOTO_PRICE_PHP;
    const disputes = getEffectiveDisputes(disputeOverrides, disputeSubmissions);
    let refundsIssued = 0;
    let refundCount = 0;
    let openCount = 0;
    let escalatedCount = 0;
    for (const d of disputes) {
      if (d.refundAmount && d.refundAmount > 0) {
        refundsIssued += d.refundAmount;
        refundCount += 1;
      }
      if (d.status === "open") openCount += 1;
      if (d.status === "escalated") escalatedCount += 1;
    }
    const refundRatePct = gmv > 0 ? (refundsIssued / gmv) * 100 : 0;
    return {
      refundsIssued,
      refundCount,
      refundRatePct,
      openCount,
      escalatedCount,
    };
  }, [payoutOverrides, disputeOverrides, disputeSubmissions]);
}
