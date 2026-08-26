import { create } from "zustand";
import { persist } from "zustand/middleware";

// Which surface set a PHOTOGRAPHER account is currently looking at. The account
// role is singular and backend-owned (re-fetched from /auth/me on every mount),
// so "Switch to Runner" can't mutate user.role — a client-side flip of the role
// would be clobbered on the next reload. This client-only flag is that switch.
//
// Only meaningful for PHOTOGRAPHER accounts; runners + admins ignore it (a pure
// runner has no photographer dashboard to switch into). resetUserScopedStores()
// (auth-reset.ts) calls reset() on every auth transition, so a fresh login
// always lands in photographer view — logging in never auto-exposes the runner
// interface. The value persists across refreshes within a session until the
// user switches back or logs out.
export type ViewMode = "photographer" | "runner";

interface ViewModeState {
  viewMode: ViewMode;
  setViewMode: (mode: ViewMode) => void;
  reset: () => void;
}

export const useViewModeStore = create<ViewModeState>()(
  persist(
    (set) => ({
      viewMode: "photographer",
      setViewMode: (viewMode) => set({ viewMode }),
      reset: () => set({ viewMode: "photographer" }),
    }),
    { name: "quickpitik-view-mode" },
  ),
);
