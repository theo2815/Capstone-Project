"use client";

import { useAuthStore } from "@/store/auth-store";
import { useViewModeStore } from "@/store/view-mode-store";
import type { Role } from "@/types/user";

// The role the UI should PRESENT as. Identical to the account's true role,
// except a PHOTOGRAPHER in runner view mode presents as RUNNER — so every
// role-branched SURFACE (profile, account, save button, face-search library,
// rail "More" list, photographer inbox bell) renders the runner variant and the
// experience is never mixed.
//
// This is for PRESENTATION only. Capability / data-access decisions must keep
// reading the true `user.role`, NOT this:
//   - route guards (`<ProtectedRoute allowedRoles>`, dashboard/admin layouts)
//   - `roleHome()` post-login/register/verify landing + switch-back target
//   - the RUNNER-gated cart + saved-events merge (auth-hydrator.tsx)
//   - photographer-settings / verification hydration
// A photographer in runner view is still a PHOTOGRAPHER server-side, so the
// runner data features that are RUNNER-restricted (persisted saves, the runner
// inbox) stay inert — they degrade to empty rather than 403-ing the session.
export function useEffectiveRole(): Role | null {
  const role = useAuthStore((s) => s.user?.role) ?? null;
  const viewMode = useViewModeStore((s) => s.viewMode);
  if (!role) return null;
  return role === "PHOTOGRAPHER" && viewMode === "runner" ? "RUNNER" : role;
}
