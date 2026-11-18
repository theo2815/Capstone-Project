"use client";

import { useMemo } from "react";
import { useAdminPayouts, EMPTY_PAYOUTS } from "@/hooks/use-admin-payouts";
import {
  PHOTO_PRICE_PHP,
  splitGmv,
} from "@/lib/platform-economics";
import { useAdminEvents, EMPTY_ADMIN_EVENTS } from "@/hooks/use-admin-events";
import { useAdminDisputes, EMPTY_DISPUTES } from "@/hooks/use-admin-disputes";
import { useAdminSalesByEventLive } from "@/hooks/use-admin-data";
import { formatPrice } from "@/lib/utils";

// Admin sales — derivation hooks. Pure read-side.
//
// The money tiles read the server only (`useAdminSalesKpisQuery`). The weekly
// trend and top photographers are still estimated from payout photo counts
// until AX-18 wires the server series; both carry SALES_ESTIMATE_NOTE.
// `useSalesByEvent` prefers the server's real order sums and marks rows it
// had to derive in the browser as `implied`.

export const SALES_ESTIMATE_NOTE = `Estimated from payout photo counts × ${formatPrice(PHOTO_PRICE_PHP)} until server figures arrive.`;

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
  const serverPayouts = useAdminPayouts().data ?? EMPTY_PAYOUTS;

  return useMemo(() => {
    const cycles = serverPayouts;
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
  }, [serverPayouts, weeks]);
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
  const serverPayouts = useAdminPayouts().data ?? EMPTY_PAYOUTS;

  return useMemo(() => {
    const cycles = serverPayouts;
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
      // Tie-break on id: equal-GMV photographers otherwise fall back to Map
      // insertion order, which follows whatever order the payout cycles
      // arrived in, so the leaderboard reshuffled between reloads.
      .sort(
        (a, b) =>
          b.gmv - a.gmv || a.photographerId.localeCompare(b.photographerId),
      )
      .slice(0, limit);
  }, [serverPayouts, limit]);
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
  /** True when derived in the browser (photoCount × price), not a server order sum. */
  implied: boolean;
}

export function useSalesByEvent(): SalesEventRow[] {
  const catalog = useAdminEvents().data ?? EMPTY_ADMIN_EVENTS;
  const disputes = useAdminDisputes().data ?? EMPTY_DISPUTES;

  const derived = useMemo(() => {
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
          implied: true,
        };
      });
  }, [catalog, disputes]);

  // Prefer the server's real order sums; fall back to derivation while it has no answer.
  const live = useAdminSalesByEventLive();
  const liveRows = useMemo(() => live?.map((r) => ({ ...r, implied: false })), [live]);
  return liveRows ?? derived;
}

export function useTopEvents(limit: number = 10): SalesEventRow[] {
  const all = useSalesByEvent();
  return useMemo(
    () => [...all].sort((a, b) => b.impliedGmv - a.impliedGmv).slice(0, limit),
    [all, limit],
  );
}

// Counts from the dispute list; the refund money comes from the server KPIs.
export interface RefundPulse {
  refundCount: number;
  openCount: number;
  escalatedCount: number;
}

export function useRefundPulse(): RefundPulse {
  const disputes = useAdminDisputes().data ?? EMPTY_DISPUTES;

  return useMemo(() => {
    let refundCount = 0;
    let openCount = 0;
    let escalatedCount = 0;
    for (const d of disputes) {
      if (d.refundAmount && d.refundAmount > 0) refundCount += 1;
      if (d.status === "open") openCount += 1;
      if (d.status === "escalated") escalatedCount += 1;
    }
    return { refundCount, openCount, escalatedCount };
  }, [disputes]);
}
