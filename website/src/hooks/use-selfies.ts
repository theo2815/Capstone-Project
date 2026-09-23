"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchSelfies } from "@/lib/api-selfies";
import { usePrivateQuerySession } from "@/hooks/use-private-query-session";
import type { SelfieRef } from "@/store/user-media-store";

// Gated on auth: `GET /me/selfies` is auth-required, so firing it for a
// guest would 401 → ApiClient catches it → redirects to /login. That broke
// guest face search on /events/[slug] — the SelfieSearchPanel mounts this
// hook unconditionally to detect a library, but only AUTHED runners have
// one. For guests we skip the call and return [].
export function useSelfiesList(): {
  selfies: SelfieRef[];
  isLoading: boolean;
  error: unknown;
} {
  const session = usePrivateQuerySession();
  const query = useQuery<SelfieRef[]>({
    queryKey: ["me", "selfies", ...session.key],
    queryFn: ({ signal }) => fetchSelfies(signal),
    staleTime: 60_000,
    enabled: session.enabled,
  });

  return {
    selfies: query.data ?? [],
    isLoading: query.isPending && session.enabled,
    error: query.error,
  };
}
