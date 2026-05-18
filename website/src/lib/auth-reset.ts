import { useCartStore } from "@/store/cart-store";
import { useOrdersStore } from "@/store/orders-store";
import { usePhotographerSettingsStore } from "@/store/photographer-settings-store";
import { useSavedEventsStore } from "@/store/saved-events-store";
import { useUserMediaStore } from "@/store/user-media-store";

// Reset every Zustand store that carries user-scoped data + clears the
// underlying localStorage slot each one persists into. Call this on every
// auth transition (login, register, logout) so a fresh authenticated
// session never inherits the previous user's photographer settings,
// cart, saved events, orders, or selfies.
//
// History: the 2026-05-18 audit found that User A's photographer-settings
// (Facebook URLs, payouts, brand) rehydrated into User B's /dashboard/settings
// in the same browser because every persisted store had no userId scoping
// and no clear-on-logout hook. Same family of bug as the disabled mock
// Google OAuth identity leak — different layer, same symptom ("all users
// share same profile"). See website/decisions.md 2026-05-18.
//
// Cart + saved-events have a syncEnabled flag that, when true, mirrors
// mutations to /me/cart and /me/saved-events. Flip both off BEFORE calling
// clear() so a logout-time clear doesn't fire a spurious DELETE against a
// stale token (which would 401 silently but pollutes the network panel).
export function resetUserScopedStores(): void {
  useCartStore.getState().setSyncEnabled(false);
  useSavedEventsStore.getState().setSyncEnabled(false);

  usePhotographerSettingsStore.getState().reset();
  useUserMediaStore.getState().clear();
  useOrdersStore.getState().clear();
  useCartStore.getState().clear();
  useSavedEventsStore.getState().clear();
}
