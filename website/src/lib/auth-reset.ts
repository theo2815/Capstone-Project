import { useCartStore } from "@/store/cart-store";
import { useOrdersStore } from "@/store/orders-store";
import { usePhotographerSettingsStore } from "@/store/photographer-settings-store";
import { useSavedEventsStore } from "@/store/saved-events-store";
import { useUserMediaStore } from "@/store/user-media-store";
import { useAdminDisputeStore } from "@/store/admin-dispute-store";
import { useAdminEventOverridesStore } from "@/store/admin-event-overrides-store";
import { useAdminFlagStore } from "@/store/admin-flag-store";
import { useAdminLegendStore } from "@/store/admin-legend-store";
import { useAdminPaletteStore } from "@/store/admin-palette-store";
import { useAdminPayoutStore } from "@/store/admin-payout-store";
import { useAdminPayoutReportStore } from "@/store/admin-payout-report-store";
import { useAdminUserStore } from "@/store/admin-user-store";
import { useAdminWsStatusStore } from "@/store/admin-ws-status-store";
import { useAdminUsersServerStore } from "@/lib/admin-users-data";
import { useMyPhotographerMessagesStore } from "@/lib/me-photographer-messages-data";
import { useMyRunnerMessagesStore } from "@/lib/me-runner-messages-data";
import { resetPhotographerHydrationGuard } from "@/hooks/use-photographer-settings-hydration";
import { useViewModeStore } from "@/store/view-mode-store";

// Reset every Zustand store that carries user-scoped data + clears the
// underlying localStorage slot each one persists into. Call this on every
// auth transition (login, register, logout) so a fresh authenticated
// session never inherits the previous user's photographer settings,
// cart, saved events, orders, selfies, or admin decisions/queue caches.
//
// History: the 2026-05-18 audit found that User A's photographer-settings
// (Facebook URLs, payouts, brand) rehydrated into User B's /dashboard/settings
// in the same browser because every persisted store had no userId scoping
// and no clear-on-logout hook. Same family of bug as the disabled mock
// Google OAuth identity leak — different layer, same symptom ("all users
// share same profile"). See website/decisions.md 2026-05-18.
//
// The 2026-05-27 admin audit caught the same family of leak on the admin
// side: nine admin stores (decision logs, queue caches, palette recents,
// flag overrides) survived logout, so the next admin login saw stale data
// before React Query revalidated. All nine are now reset below.
//
// Cart + saved-events have a syncEnabled flag that, when true, mirrors
// mutations to /me/cart and /me/saved-events. Flip both off BEFORE calling
// clear() so a logout-time clear doesn't fire a spurious DELETE against a
// stale token (which would 401 silently but pollutes the network panel).
export function resetUserScopedStores(): void {
  useCartStore.getState().setSyncEnabled(false);
  useSavedEventsStore.getState().setSyncEnabled(false);

  // View mode is a client-only photographer/runner toggle — always reset to
  // photographer so a fresh login never auto-exposes the runner interface.
  useViewModeStore.getState().reset();

  usePhotographerSettingsStore.getState().reset();
  // The hydration guard lives at module level (it must survive DashboardShell
  // remounts) — re-arm it here so the next account hydrates fresh.
  resetPhotographerHydrationGuard();
  // PF-9 (2026-05-27): the photographer-messages inbox cache survived logout
  // — next photographer login saw stale messages + unread badge from the
  // previous account until the first refetch. Same class as the 9 admin-store
  // leaks closed 2026-05-27 (see admin-flow audit P0).
  useMyPhotographerMessagesStore.getState().reset();
  // Same leak, runner side (2026-08-28 audit): PF-9 closed the photographer
  // inbox only — the runner inbox store kept the previous account's messages
  // and unread badge until the first refetch.
  useMyRunnerMessagesStore.getState().reset();
  useUserMediaStore.getState().clear();
  useOrdersStore.getState().clear();
  useCartStore.getState().clear();
  useSavedEventsStore.getState().clear();

  // Admin stores — decision logs, queue caches, optimistic overrides, and
  // command-palette recents. The legend + WS status stores reset via their
  // existing setters since they don't expose a `clear()` action.
  useAdminUserStore.getState().clear();
  useAdminPayoutStore.getState().clear();
  useAdminPayoutReportStore.getState().clear();
  useAdminDisputeStore.getState().clear();
  useAdminFlagStore.getState().clear();
  useAdminEventOverridesStore.getState().clear();
  useAdminUsersServerStore.getState().clear();
  useAdminPaletteStore.getState().setOpen(false);
  useAdminPaletteStore.getState().clearRecents();
  useAdminLegendStore.getState().setOpen(false);
  useAdminWsStatusStore.getState().setStatus("degraded");
}

// The guest buffer is the one thing the wipe above must not eat.
//
// resetUserScopedStores() runs inside useAuth.login/register, BEFORE
// <AuthHydrator> reads the cart and saved-event ids for the guest→authed merge
// (Q-003). So the merge was handed `[]` on every login: a guest who filled a
// cart, hit Checkout, and signed in to pay arrived at an empty checkout, and a
// guest's bookmarks never reached /me/saved-events/merge at all.
//
// Capture before the wipe, restore after it, then let the merge run. Safe
// because every other teardown wipes too — logout (useAuth.logout) and the
// revoked-session path in <AuthHydrator> — so whatever survives in these two
// stores at login time is the guest's own, not a previous user's residue.
// That invariant is why this can be a straight restore instead of scoping the
// persist keys by userId.
export function captureGuestBuffer(): () => void {
  const items = useCartStore.getState().items;
  const ids = useSavedEventsStore.getState().ids;

  return () => {
    if (items.length > 0) useCartStore.getState().setItems(items);
    if (ids.length > 0) useSavedEventsStore.getState().setIds(ids);
  };
}
