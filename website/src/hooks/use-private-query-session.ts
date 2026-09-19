"use client";

import { useAuthStore } from "@/store/auth-store";
import type { User } from "@/types/user";

// Append after the domain prefix so existing invalidation still reaches all
// its lists/details. A new login, even to the same account, gets fresh data.
export function usePrivateQuerySession(role?: User["role"]) {
  const user = useAuthStore((s) => s.user);
  const authenticated = useAuthStore((s) => s.isAuthenticated);
  const version = useAuthStore((s) => s.sessionVersion);
  const enabled = authenticated && !!user && (!role || user.role === role);
  return {
    enabled,
    key: [enabled ? user.id : null, enabled ? user.role : null, version] as const,
  };
}
