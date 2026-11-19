"use client";

import { useMemo } from "react";
import { PHOTO_PRICE_PHP, splitGmv } from "@/lib/platform-economics";
import { useAdminEvents, EMPTY_ADMIN_EVENTS } from "@/hooks/use-admin-events";
import { useAdminDisputes, EMPTY_DISPUTES } from "@/hooks/use-admin-disputes";
import { useAdminSalesByEventLive } from "@/hooks/use-admin-data";

// Admin sales — derivation hooks. Pure read-side.
//
// The money tiles, weekly trend and top photographers read the server only
// (`use-admin-data.ts`). `useSalesByEvent` prefers the server's real order
// sums and marks rows it had to derive in the browser as `implied`.

export const SALES_TREND_WEEKS = 12;

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
