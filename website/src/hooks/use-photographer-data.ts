"use client";

import { usePrivateQuerySession } from "@/hooks/use-private-query-session";

import { useQuery, useInfiniteQuery } from "@tanstack/react-query";
import { dedupeItems, useInfiniteList } from "@/hooks/use-infinite-list";
import { PAGE_SIZE } from "@/lib/pagination-config";
import { ApiError } from "@/lib/api";
import {
  fetchPhotographerEvents,
  fetchPhotographerEventDetail,
  fetchPhotographerEventPhotos,
  getEventMetadataVisibility,
  type EventMetadataVisibilityDto,
  type PhotographerEventDetail,
} from "@/lib/api-photographer";
import {
  fetchPhotographerEarnings,
  fetchPerEventEarnings,
  fetchPhotographerPayouts,
  fetchPhotographerPayoutBalance,
  fetchPhotographerTransactions,
  type PayoutBalanceResponse,
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
import type { PhotographerProfile } from "@/lib/photographer-registry";

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
  const session = usePrivateQuerySession("PHOTOGRAPHER");
  const query = useQuery<PhotographerEventSummary[]>({
    queryKey: ["photographer", "events", args, ...session.key],
    queryFn: ({ signal }) => fetchPhotographerEvents(args, signal).then((r) => r.items),
    enabled: session.enabled,
    staleTime: EVENTS_STALE_MS,
  });
  return query.data ?? null;
}

export interface PhotographerEventDetailResult {
  /** null while the fetch is in flight — and also on failure. */
  detail: PhotographerEventDetail | null;
  /** BE returned 404: the photographer doesn't cover this event, or it's gone. */
  isMissing: boolean;
  error: Error | null;
  isLoading: boolean;
  refetch: () => void;
}

// Returns a result object rather than the bare detail because `null` alone
// can't distinguish "still loading" from "doesn't exist" — and the caller has
// to 404 on the second. Without `isMissing`, a bad id renders a skeleton
// forever, since api.get throws on 404 and query.data stays undefined.
export function usePhotographerEventDetail(
  eventId: string | null,
  { includeRemoved = false }: { includeRemoved?: boolean } = {},
): PhotographerEventDetailResult {
  const session = usePrivateQuerySession("PHOTOGRAPHER");
  const query = useQuery<PhotographerEventDetail | null>({
    queryKey: ["photographer", "events", eventId, includeRemoved, ...session.key],
    queryFn: ({ signal }) =>
      eventId ? fetchPhotographerEventDetail(eventId, signal, includeRemoved) : Promise.resolve(null),
    enabled: session.enabled && !!eventId,
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
    error: query.error,
    isLoading: query.isPending && session.enabled,
    refetch: () => void query.refetch(),
  };
}

export function usePhotographerEventPhotos(
  eventId: string | null,
  opts: { deleted?: boolean; enabled?: boolean } = {},
) {
  const deleted = opts.deleted ?? false;
  const session = usePrivateQuerySession("PHOTOGRAPHER");
  return useInfiniteList<PhotographerLibraryPhoto>({
    // Active and Deleted (trash) are separate cached lists for the same event.
    queryKey: [
      "photographer",
      "events",
      eventId,
      "photos",
      deleted ? "deleted" : "active",
      ...session.key,
    ],
    fetchPage: (offset, limit, signal) =>
      eventId
        ? fetchPhotographerEventPhotos(eventId, { offset, limit, deleted }, signal)
        : Promise.resolve({ items: [], total: 0, offset, limit }),
    limit: PAGE_SIZE.PHOTO_INCREMENT,
    enabled: session.enabled && !!eventId && (opts.enabled ?? true),
    staleTime: EVENTS_STALE_MS,
  });
}

export function usePhotographerEventMetadataVisibility(eventId: string | null) {
  const session = usePrivateQuerySession("PHOTOGRAPHER");
  return useQuery<EventMetadataVisibilityDto>({
    queryKey: ["photographer", "events", eventId, "metadata-visibility", ...session.key],
    queryFn: ({ signal }) => getEventMetadataVisibility(eventId!, signal),
    enabled: session.enabled && !!eventId,
    staleTime: EVENTS_STALE_MS,
  });
}

// ───────────────────────────────────────────── Earnings

export function usePhotographerEarnings(): PhotographerEarnings | null {
  const session = usePrivateQuerySession("PHOTOGRAPHER");
  const query = useQuery<PhotographerEarnings | null>({
    queryKey: ["photographer", "earnings", ...session.key],
    queryFn: ({ signal }) => fetchPhotographerEarnings(signal),
    enabled: session.enabled,
    staleTime: EARNINGS_STALE_MS,
  });
  return query.data ?? null;
}

// The per-event list follows server order and uses raw server offsets, so
// events past the former first-200 cap remain reachable.
export function usePhotographerPerEventEarnings() {
  const session = usePrivateQuerySession("PHOTOGRAPHER");
  const query = useInfiniteQuery({
    queryKey: ["photographer", "earnings", "per-event", ...session.key],
    queryFn: ({ pageParam, signal }) => fetchPerEventEarnings({ offset: pageParam, limit: pageParam === 0 ? PAGE_SIZE.EARNINGS_INITIAL : PAGE_SIZE.EARNINGS_INCREMENT }, signal),
    initialPageParam: 0,
    getNextPageParam: (last) => last.items.length && last.offset + last.limit < last.total ? last.offset + last.limit : undefined,
    enabled: session.enabled,
    staleTime: EARNINGS_STALE_MS,
  });
  const pages = query.data?.pages ?? [];
  return {
    items: pages.flatMap((page) => page.items),
    total: pages.at(-1)?.total ?? 0,
    isLoading: query.isPending && session.enabled,
    error: query.isFetchNextPageError ? null : query.error,
    appendError: query.isFetchNextPageError ? query.error : null,
    refetch: () => void query.refetch(),
    fetchNextPage: () => void query.fetchNextPage(),
    hasNextPage: query.hasNextPage,
    isFetchingNextPage: query.isFetchingNextPage,
  };
}

// ───────────────────────────────────────────── Payouts

export function usePhotographerPayouts() {
  const session = usePrivateQuerySession("PHOTOGRAPHER");
  return useInfiniteList<PhotographerPayout>({
    queryKey: ["photographer", "payouts", ...session.key],
    fetchPage: (offset, limit, signal) => fetchPhotographerPayouts({ offset, limit }, signal),
    limit: PAGE_SIZE.PAYOUT_INCREMENT,
    enabled: session.enabled,
    staleTime: EARNINGS_STALE_MS,
  });
}

// Unpaid balance + open-request state for the Request Payout hero on
// /dashboard/billing. Short stale time so the hero reacts quickly after a
// request is submitted or admin acts on a cycle.
export function usePhotographerPayoutBalance(): PayoutBalanceResponse | null {
  const session = usePrivateQuerySession("PHOTOGRAPHER");
  const query = useQuery<PayoutBalanceResponse | null>({
    queryKey: ["photographer", "payouts", "balance", ...session.key],
    queryFn: ({ signal }) => fetchPhotographerPayoutBalance(signal),
    enabled: session.enabled,
    staleTime: 30_000,
  });
  return query.data ?? null;
}

// ───────────────────────────────────────────── Transactions

// Uses useInfiniteQuery directly (not useInfiniteList) to surface the extra
// `monthTotals` envelope field. The server computes monthTotals over ALL rows,
// so it is identical on every page — read it from page 0, never accumulate.
export function usePhotographerTransactions() {
  const session = usePrivateQuerySession("PHOTOGRAPHER");
  const query = useInfiniteQuery({
    queryKey: ["photographer", "transactions", ...session.key],
    queryFn: ({ pageParam, signal }) =>
      fetchPhotographerTransactions({
        offset: pageParam,
        limit: PAGE_SIZE.TRANSACTION_INCREMENT,
      }, signal),
    initialPageParam: 0,
    getNextPageParam: (last) => {
      const next = last.offset + last.limit;
      return next < last.total ? next : undefined;
    },
    enabled: session.enabled,
    staleTime: EARNINGS_STALE_MS,
  });
  const pages = query.data?.pages ?? [];
  return {
    items: dedupeItems(pages),
    total: pages.at(-1)?.total ?? 0,
    monthTotals: pages[0]?.monthTotals ?? {},
    isLoading: query.isPending && session.enabled,
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
