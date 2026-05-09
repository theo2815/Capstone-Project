"use client";

import { useQuery } from "@tanstack/react-query";
import { BACKEND_LIVE } from "@/lib/backend-flag";
import { fetchSelfies } from "@/lib/api-selfies";
import { useUserMediaStore, type SelfieRef } from "@/store/user-media-store";

// React Query for live mode, Zustand subscription for mock mode. Same hybrid
// pattern as use-orders.ts so mock-mode mutations through useUserMediaStore
// propagate without manual cache invalidation.

export function useSelfiesList(): {
  selfies: SelfieRef[];
  isLoading: boolean;
  error: unknown;
} {
  const storeSelfies = useUserMediaStore((s) => s.selfies);

  const query = useQuery<SelfieRef[]>({
    queryKey: ["me", "selfies"],
    queryFn: () => fetchSelfies(),
    enabled: BACKEND_LIVE,
    staleTime: 60_000,
  });

  if (BACKEND_LIVE) {
    return {
      selfies: query.data ?? [],
      isLoading: query.isPending,
      error: query.error,
    };
  }
  return { selfies: storeSelfies, isLoading: false, error: null };
}
