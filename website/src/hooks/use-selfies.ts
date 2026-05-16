"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchSelfies } from "@/lib/api-selfies";
import type { SelfieRef } from "@/store/user-media-store";

export function useSelfiesList(): {
  selfies: SelfieRef[];
  isLoading: boolean;
  error: unknown;
} {
  const query = useQuery<SelfieRef[]>({
    queryKey: ["me", "selfies"],
    queryFn: () => fetchSelfies(),
    staleTime: 60_000,
  });

  return {
    selfies: query.data ?? [],
    isLoading: query.isPending,
    error: query.error,
  };
}
