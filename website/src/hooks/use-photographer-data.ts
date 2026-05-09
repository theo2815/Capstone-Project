"use client";

import { useQuery } from "@tanstack/react-query";
import { BACKEND_LIVE } from "@/lib/backend-flag";
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
  fetchPhotographerPayoutReports,
  fetchPhotographerTransactions,
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
import type { MockPhoto } from "@/app/events/[slug]/mock-photos";
import type { EventDetail } from "@/types/event";

// Hybrid hooks for photographer reads. Each hook returns canonical server
// data when `BACKEND_LIVE=true`; in mock mode the hook returns `null` so
// callers fall back to existing store/mock derivations.
//
// React Query keys: ["photographer", <domain>, ...]
// Stale times: events 60s, earnings 5min, transactions 60s, public 5min.

const EVENTS_STALE_MS = 60_000;
const EARNINGS_STALE_MS = 5 * 60_000;
const PUBLIC_STALE_MS = 5 * 60_000;

// ───────────────────────────────────────────── Covered events

export function usePhotographerEvents(
  args: { withUploads?: boolean } = {},
): PhotographerEventSummary[] | null {
  const query = useQuery<PhotographerEventSummary[]>({
    queryKey: ["photographer", "events", args],
    queryFn: () => fetchPhotographerEvents(args),
    enabled: BACKEND_LIVE,
    staleTime: EVENTS_STALE_MS,
  });
  return BACKEND_LIVE ? query.data ?? null : null;
}

export function usePhotographerEventDetail(
  eventId: string | null,
): PhotographerEventDetail | null {
  const query = useQuery<PhotographerEventDetail | null>({
    queryKey: ["photographer", "events", eventId],
    queryFn: () =>
      eventId ? fetchPhotographerEventDetail(eventId) : Promise.resolve(null),
    enabled: BACKEND_LIVE && !!eventId,
    staleTime: EVENTS_STALE_MS,
  });
  return BACKEND_LIVE ? query.data ?? null : null;
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
    enabled: BACKEND_LIVE && !!eventId,
    staleTime: EVENTS_STALE_MS,
  });
  return BACKEND_LIVE ? query.data ?? null : null;
}

// ───────────────────────────────────────────── Earnings

export function usePhotographerEarnings(): PhotographerEarnings | null {
  const query = useQuery<PhotographerEarnings | null>({
    queryKey: ["photographer", "earnings"],
    queryFn: () => fetchPhotographerEarnings(),
    enabled: BACKEND_LIVE,
    staleTime: EARNINGS_STALE_MS,
  });
  return BACKEND_LIVE ? query.data ?? null : null;
}

export function usePhotographerPerEventEarnings(): PerEventEarning[] | null {
  const query = useQuery<PerEventEarning[]>({
    queryKey: ["photographer", "earnings", "per-event"],
    queryFn: () => fetchPerEventEarnings(),
    enabled: BACKEND_LIVE,
    staleTime: EARNINGS_STALE_MS,
  });
  return BACKEND_LIVE ? query.data ?? null : null;
}

// ───────────────────────────────────────────── Payouts

export function usePhotographerPayouts(): PhotographerPayout[] | null {
  const query = useQuery<PhotographerPayout[]>({
    queryKey: ["photographer", "payouts"],
    queryFn: () => fetchPhotographerPayouts(),
    enabled: BACKEND_LIVE,
    staleTime: EARNINGS_STALE_MS,
  });
  return BACKEND_LIVE ? query.data ?? null : null;
}

export function usePhotographerPayoutReports(args: {
  status?: PayoutReportStatus;
} = {}): PayoutReport[] | null {
  const query = useQuery<PayoutReport[] | null>({
    queryKey: ["photographer", "payouts", "reports", args],
    queryFn: () => fetchPhotographerPayoutReports(args),
    enabled: BACKEND_LIVE,
    staleTime: EARNINGS_STALE_MS,
  });
  return BACKEND_LIVE ? query.data ?? null : null;
}

// ───────────────────────────────────────────── Transactions

export function usePhotographerTransactions(): TransactionsResponse | null {
  const query = useQuery<TransactionsResponse>({
    queryKey: ["photographer", "transactions"],
    queryFn: () => fetchPhotographerTransactions(),
    enabled: BACKEND_LIVE,
    staleTime: EARNINGS_STALE_MS,
  });
  return BACKEND_LIVE ? query.data ?? null : null;
}

// ───────────────────────────────────────────── Public profile

export function usePublicPhotographer(
  handle: string | null,
): PhotographerProfile | null {
  const query = useQuery<PhotographerProfile | null>({
    queryKey: ["photographer", "public", handle],
    queryFn: () =>
      handle ? fetchPublicPhotographer(handle) : Promise.resolve(null),
    enabled: BACKEND_LIVE && !!handle,
    staleTime: PUBLIC_STALE_MS,
  });
  return BACKEND_LIVE ? query.data ?? null : null;
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
    enabled: BACKEND_LIVE && !!handle && !!eventSlug && !!event,
    staleTime: PUBLIC_STALE_MS,
  });
  return BACKEND_LIVE ? query.data ?? null : null;
}

// ───────────────────────────────────────────── Platform fees

export function usePlatformFees(): PlatformFees {
  const query = useQuery<PlatformFees | null>({
    queryKey: ["platform", "fees"],
    queryFn: () => fetchPlatformFees(),
    enabled: BACKEND_LIVE,
    staleTime: PUBLIC_STALE_MS,
  });
  if (BACKEND_LIVE && query.data) return query.data;
  return PLATFORM_FEES_FALLBACK;
}
