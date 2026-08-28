"use client";

import { useQuery, useInfiniteQuery } from "@tanstack/react-query";
import { dedupeItems, useInfiniteList } from "@/hooks/use-infinite-list";
import { PAGE_SIZE } from "@/lib/pagination-config";
import { ApiError } from "@/lib/api";
import {
  fetchPhotographerEvents,
  fetchPhotographerEventDetail,
  fetchPhotographerEventPhotos,
  type PhotographerEventDetail,
} from "@/lib/api-photographer";
import {
  fetchPhotographerEarnings,
  fetchPerEventEarnings,
  fetchPhotographerPayouts,
  fetchPhotographerPayoutBalance,
  fetchPhotographerPayoutReports,
  fetchPhotographerTransactions,
  type PayoutBalanceResponse,
  type PerEventEarning,
} from "@/lib/api-photographer-earnings";
import { fetchPublicPhotographer } from "@/lib/api-photographer-public";
import {
  fetchPlatformFees,
  PLATFORM_FEES_FALLBACK,
  type PlatformFees,
} from "@/lib/api-platform";
import type {
  PhotographerEarnings,
  PhotographerEventSummary,
  PhotographerLibraryPhoto,
  PhotographerPayout,
} from "@/lib/photographer-mock";
import type {
  PayoutReport,
  PayoutReportStatus,
} from "@/lib/admin-payout-reports";
import type { PhotographerProfile } from "@/lib/photographer-registry";
import type { PaginatedResponse } from "@/types/api";

// React Query hooks for photographer reads.
// Keys: ["photographer", <domain>, ...]
// Stale times: events 60s, earnings 5min, transactions 60s, public 5min.

const EVENTS_STALE_MS = 60_000;
const EARNINGS_STALE_MS = 5 * 60_000;
// Public profile + platform fees move on settings saves / config changes, not
// by the minute. 30 min stays well under the 1 h presigned-URL TTL on covers.
const PUBLIC_STALE_MS = 30 * 60_000;

// ───────────────────────────────────────────── Covered events

// Covered events. The list surfaces (/dashboard/events, /profile portfolio,
// dashboard "Next up") filter by date/search CLIENT-SIDE — the BE endpoint only
// supports `withUploads`, so true offset pagination would only ever filter the
// loaded page. Instead those surfaces pass `limit: COVERED_EVENTS_MAX` to pull
// the full set (the BE max, which covers every realistic photographer) and keep
// their own Load-more client-slice, so the rendered DOM stays bounded. Simple
// consumers (setup journey, action grid) omit it and take the fetcher default,
// which is also the BE max — a lower default silently truncated the
// dashboard's setup-mode fork and Next-up glance at 24 rows.
export const COVERED_EVENTS_MAX = 200;

export function usePhotographerEvents(
  args: { withUploads?: boolean; limit?: number } = {},
): PhotographerEventSummary[] | null {
  const query = useQuery<PhotographerEventSummary[]>({
    queryKey: ["photographer", "events", args],
    queryFn: () => fetchPhotographerEvents(args).then((r) => r.items),
    staleTime: EVENTS_STALE_MS,
  });
  return query.data ?? null;
}

export interface PhotographerEventDetailResult {
  /** null while the fetch is in flight — and also on failure. */
  detail: PhotographerEventDetail | null;
  /** BE returned 404: the photographer doesn't cover this event, or it's gone. */
  isMissing: boolean;
}

// Returns a result object rather than the bare detail because `null` alone
// can't distinguish "still loading" from "doesn't exist" — and the caller has
// to 404 on the second. Without `isMissing`, a bad id renders a skeleton
// forever, since api.get throws on 404 and query.data stays undefined.
export function usePhotographerEventDetail(
  eventId: string | null,
): PhotographerEventDetailResult {
  const query = useQuery<PhotographerEventDetail | null>({
    queryKey: ["photographer", "events", eventId],
    queryFn: () =>
      eventId ? fetchPhotographerEventDetail(eventId) : Promise.resolve(null),
    enabled: !!eventId,
    staleTime: EVENTS_STALE_MS,
    // A 404 is a verdict, not a blip. Retrying it just holds the skeleton up
    // for several seconds before the page can render its 404. 429 is excluded
    // for the opposite reason — retrying re-arms the empty bucket (this
    // override replaces the global predicate in providers.tsx, so it must
    // carry the same exclusion).
    retry: (failureCount, err) =>
      !(
        err instanceof ApiError &&
        (err.status === 404 || err.status === 429)
      ) && failureCount < 3,
  });
  return {
    detail: query.data ?? null,
    isMissing: query.error instanceof ApiError && query.error.status === 404,
  };
}

export function usePhotographerEventPhotos(eventId: string | null) {
  return useInfiniteList<PhotographerLibraryPhoto>({
    queryKey: ["photographer", "events", eventId, "photos"],
    fetchPage: (offset, limit) =>
      eventId
        ? fetchPhotographerEventPhotos(eventId, { offset, limit })
        : Promise.resolve({ items: [], total: 0, offset, limit }),
    limit: PAGE_SIZE.PHOTO_INCREMENT,
    enabled: !!eventId,
    staleTime: EVENTS_STALE_MS,
  });
}

// ───────────────────────────────────────────── Earnings

export function usePhotographerEarnings(): PhotographerEarnings | null {
  const query = useQuery<PhotographerEarnings | null>({
    queryKey: ["photographer", "earnings"],
    queryFn: () => fetchPhotographerEarnings(),
    staleTime: EARNINGS_STALE_MS,
  });
  return query.data ?? null;
}

// Single capped fetch (not offset pagination): the slab sorts client-side by
// revenueKept desc and the BE has no revenue-sort param, so paging would only
// rank the loaded page. The fetcher defaults to the BE max, and the slab
// client-slices the render. Envelope kept so the page can say "N of M".
export function usePhotographerPerEventEarnings(): PaginatedResponse<PerEventEarning> | null {
  const query = useQuery<PaginatedResponse<PerEventEarning>>({
    queryKey: ["photographer", "earnings", "per-event"],
    queryFn: () => fetchPerEventEarnings(),
    staleTime: EARNINGS_STALE_MS,
  });
  return query.data ?? null;
}

// ───────────────────────────────────────────── Payouts

export function usePhotographerPayouts() {
  return useInfiniteList<PhotographerPayout>({
    queryKey: ["photographer", "payouts"],
    fetchPage: (offset, limit) => fetchPhotographerPayouts({ offset, limit }),
    limit: PAGE_SIZE.PAYOUT_INCREMENT,
    staleTime: EARNINGS_STALE_MS,
  });
}

// Unpaid balance + open-request state for the Request Payout hero on
// /dashboard/billing. Short stale time so the hero reacts quickly after a
// request is submitted or admin acts on a cycle.
export function usePhotographerPayoutBalance(): PayoutBalanceResponse | null {
  const query = useQuery<PayoutBalanceResponse | null>({
    queryKey: ["photographer", "payouts", "balance"],
    queryFn: () => fetchPhotographerPayoutBalance(),
    staleTime: 30_000,
  });
  return query.data ?? null;
}

export function usePhotographerPayoutReports(args: {
  status?: PayoutReportStatus;
} = {}): PayoutReport[] | null {
  const query = useQuery<PayoutReport[] | null>({
    queryKey: ["photographer", "payouts", "reports", args],
    queryFn: () => fetchPhotographerPayoutReports(args),
    staleTime: EARNINGS_STALE_MS,
  });
  return query.data ?? null;
}

// ───────────────────────────────────────────── Transactions

// Uses useInfiniteQuery directly (not useInfiniteList) to surface the extra
// `monthTotals` envelope field. The server computes monthTotals over ALL rows,
// so it is identical on every page — read it from page 0, never accumulate.
export function usePhotographerTransactions() {
  const query = useInfiniteQuery({
    queryKey: ["photographer", "transactions"],
    queryFn: ({ pageParam }) =>
      fetchPhotographerTransactions({
        offset: pageParam,
        limit: PAGE_SIZE.TRANSACTION_INCREMENT,
      }),
    initialPageParam: 0,
    getNextPageParam: (last) => {
      const next = last.offset + last.limit;
      return next < last.total ? next : undefined;
    },
    staleTime: EARNINGS_STALE_MS,
  });
  const pages = query.data?.pages ?? [];
  return {
    items: dedupeItems(pages),
    total: pages.at(-1)?.total ?? 0,
    monthTotals: pages[0]?.monthTotals ?? {},
    isLoading: query.isPending,
    isFetchingNextPage: query.isFetchingNextPage,
    hasNextPage: query.hasNextPage,
    fetchNextPage: () => void query.fetchNextPage(),
    error: query.error,
  };
}

// ───────────────────────────────────────────── Public profile

export function usePublicPhotographer(
  handle: string | null,
): PhotographerProfile | null {
  const query = useQuery<PhotographerProfile | null>({
    queryKey: ["photographer", "public", handle],
    queryFn: () =>
      handle ? fetchPublicPhotographer(handle) : Promise.resolve(null),
    enabled: !!handle,
    staleTime: PUBLIC_STALE_MS,
  });
  return query.data ?? null;
}

// ───────────────────────────────────────────── Platform fees

export function usePlatformFees(): PlatformFees {
  const query = useQuery<PlatformFees | null>({
    queryKey: ["platform", "fees"],
    queryFn: () => fetchPlatformFees(),
    staleTime: PUBLIC_STALE_MS,
  });
  return query.data ?? PLATFORM_FEES_FALLBACK;
}
