"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchRecentAdminDecisions } from "@/lib/api-admin";
import { useAuthStore } from "@/store/auth-store";

export function useAdminDecisions(limit: number, enabled = true) {
  const user = useAuthStore((s) => s.user);
  return useQuery({
    queryKey: ["admin", "decisions", user?.id ?? null, limit] as const,
    queryFn: ({ signal }) => fetchRecentAdminDecisions(limit, signal),
    enabled: enabled && user?.role === "ADMIN",
    staleTime: 30_000,
  });
}
