"use client";

import { useQuery } from "@tanstack/react-query";
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
  type TransactionsResponse,
} from "@/lib/api-photographer-earnings";
import {
  fetchPublicPhotographer,
  fetchPublicPhotographerEventPhotos,
  type PublicPhotographerPhotosArgs,
} from "@/lib/api-photographer-public";
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
import type { MockPhoto } from "@/types/photo";
import type { EventDetail } from "@/types/event";
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

export function usePhotographerEvents(
  args: { withUploads?: boolean } = {},
): PhotographerEventSummary[] | null {
  const query = useQuery<PhotographerEventSummary[]>({
    queryKey: ["photographer", "events", args],
    queryFn: () => fetchPhotographerEvents(args),
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

export function usePhotographerEventPhotos(
  eventId: string | null,
): PhotographerLibraryPhoto[] | null {
  const query = useQuery<PhotographerLibraryPhoto[]>({
    queryKey: ["photographer", "events", eventId, "photos"],
    queryFn: () =>
      eventId
        ? fetchPhotographerEventPhotos(eventId)
        : Promise.resolve([]),
    enabled: !!eventId,
    staleTime: EVENTS_STALE_MS,
  });
  return query.data ?? null;
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

// Envelope, not a bare array — the page needs `total` to say "N of M" instead
// of hedging with "the 200 most recent".
export function usePhotographerPerEventEarnings(): PaginatedResponse<PerEventEarning> | null {
  const query = useQuery<PaginatedResponse<PerEventEarning>>({
    queryKey: ["photographer", "earnings", "per-event"],
    queryFn: () => fetchPerEventEarnings(),
    staleTime: EARNINGS_STALE_MS,
  });
  return query.data ?? null;
}

// ───────────────────────────────────────────── Payouts

export function usePhotographerPayouts(): PaginatedResponse<PhotographerPayout> | null {
  const query = useQuery<PaginatedResponse<PhotographerPayout>>({
    queryKey: ["photographer", "payouts"],
    queryFn: () => fetchPhotographerPayouts(),
    staleTime: EARNINGS_STALE_MS,
  });
  return query.data ?? null;
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

export function usePhotographerTransactions(): TransactionsResponse | null {
  const query = useQuery<TransactionsResponse>({
    queryKey: ["photographer", "transactions"],
    queryFn: () => fetchPhotographerTransactions(),
    staleTime: EARNINGS_STALE_MS,
  });
  return query.data ?? null;
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

export function usePublicPhotographerPhotos(
  handle: string | null,
  eventSlug: string | null,
  event: EventDetail | null,
  expectedCount: number,
  args: PublicPhotographerPhotosArgs = {},
): MockPhoto[] | null {
  const query = useQuery<MockPhoto[]>({
    queryKey: ["photographer", "public", handle, eventSlug, "photos", args],
    queryFn: () =>
      handle && eventSlug && event
        ? fetchPublicPhotographerEventPhotos(
            handle,
            eventSlug,
            event,
            expectedCount,
            args,
          )
        : Promise.resolve([]),
    enabled: !!handle && !!eventSlug && !!event,
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
