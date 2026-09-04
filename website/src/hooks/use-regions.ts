"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchRegions } from "@/lib/api-photographer-settings";
import type { PHRegion } from "@/lib/ph-regions";

// `GET /api/v1/regions` is the canonical Philippine region + province list.
// Until 2026-08-16 this app rendered a hardcoded copy from `lib/ph-regions.ts`
// while the mobile app carried a third copy of its own — so a region added
// backend-side reached nobody. The backend owns the list; both clients read it.
//
// Public endpoint, and the response carries `Cache-Control: max-age=1d`, so a
// long staleTime here just avoids re-parsing rather than saving a round trip.
// Not gated on auth: the admin review surfaces render region labels for signed-
// in staff, and the settings dropdown needs it the moment the page mounts.
export function useRegions(): {
  regions: readonly PHRegion[];
  isLoading: boolean;
  error: unknown;
} {
  const query = useQuery<PHRegion[]>({
    queryKey: ["regions"],
    // fetchRegions is typed nullable because ApiClient unwraps an empty
    // envelope to null; an absent list is an empty list to every consumer.
    queryFn: async () => ((await fetchRegions()) ?? []) as PHRegion[],
    staleTime: 24 * 60 * 60 * 1000,
  });

  return {
    regions: query.data ?? [],
    isLoading: query.isPending,
    error: query.error,
  };
}
