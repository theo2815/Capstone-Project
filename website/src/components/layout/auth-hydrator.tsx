"use client";

import { useEffect, useRef } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useAuthStore } from "@/store/auth-store";
import { useCartStore } from "@/store/cart-store";
import { useSavedEventsStore } from "@/store/saved-events-store";
import { clearTokens, getAccessToken, getRefreshToken } from "@/lib/auth";
import { resetUserScopedStores } from "@/lib/auth-reset";
import { refreshAccessToken } from "@/lib/api";
import { API_BASE_URL } from "@/lib/constants";
import { mergeCart } from "@/lib/api-cart";
import { mergeSavedEvents } from "@/lib/api-saved-events";
import type { ApiResponse } from "@/types/api";
import type { User } from "@/types/user";

export function AuthHydrator() {
  const queryClient = useQueryClient();
  const setUser = useAuthStore((s) => s.setUser);
  const setLoading = useAuthStore((s) => s.setLoading);
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const userId = useAuthStore((s) => s.user?.id);
  const role = useAuthStore((s) => s.user?.role);

  // One-shot guard so React strict-mode double-mount can't fire merge twice.
  // Reset on logout (user transition to null) so re-login re-fires.
  const mergedForUserRef = useRef<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function hydrate() {
      const accessToken = getAccessToken();
      const refreshToken = getRefreshToken();

      if (!accessToken && !refreshToken) {
        if (!cancelled) setLoading(false);
        return;
      }

      let user = await fetchMe(accessToken);
      if (cancelled) return;

      if (!user && refreshToken) {
        // Shared single-flight with ApiClient — this hook and the page's
        // React Query hooks all wake on the same expired token, and the BE
        // revokes on rotate, so independent refreshes would kill the session.
        const refreshedAccess = await refreshAccessToken();
        if (cancelled) return;
        if (refreshedAccess) {
          user = await fetchMe(refreshedAccess);
          if (cancelled) return;
        }
      }

      if (user) {
        setUser(user);
      } else {
        // A refresh token that can't be redeemed is an auth transition like
        // login or logout, and owes the same wipe. clearTokens() alone left
        // every persisted store intact, so the tab carried on as a guest still
        // holding the previous user's saved events and cart — visible
        // immediately as filled bookmark hearts on events they never saved.
        //
        // This is also what lets useAuth.login() trust the guest buffer: once
        // every teardown wipes, whatever sits in those stores at login time
        // belongs to the guest sitting there now. See captureGuestBuffer().
        clearTokens();
        resetUserScopedStores();
        // The React Query cache is the same kind of residue — clear it with
        // the stores (see the matching clears in useAuth).
        queryClient.clear();
        setLoading(false);
      }
    }

    hydrate();
    return () => {
      cancelled = true;
    };
  }, [setUser, setLoading, queryClient]);

  // Guest → authed merge per Q-003. Runs exactly once per signed-in user.
  // Failure leaves local state intact (do NOT clear) — retried on next mount.
  useEffect(() => {
    if (!isAuthenticated || !userId) {
      // Logout transition — flip stores back to local-only mode and reset
      // the one-shot guard so the next login can re-merge.
      useCartStore.getState().setSyncEnabled(false);
      useSavedEventsStore.getState().setSyncEnabled(false);
      mergedForUserRef.current = null;
      return;
    }
    if (mergedForUserRef.current === userId) return;
    mergedForUserRef.current = userId;

    // Both endpoints are @PreAuthorize("hasRole('RUNNER')") backend-side, but
    // this hydrator is mounted globally in providers.tsx — so every
    // photographer and admin login was firing two calls that could only 403.
    // Harmless (api.ts reacts to 401 only, and the catch below swallows it)
    // but noisy in the network log and in the BE's access log. Claim the
    // one-shot ref above regardless, so a non-runner session doesn't re-try on
    // every mount.
    if (role !== "RUNNER") return;

    const localItems = useCartStore.getState().items;
    const localIds = useSavedEventsStore.getState().ids;

    Promise.all([mergeCart(localItems), mergeSavedEvents(localIds)])
      .then(([mergedItems, mergedSummaries]) => {
        useCartStore.getState().setItems(mergedItems);
        useSavedEventsStore.getState().setSummaries(mergedSummaries);
      })
      .catch(() => {
        // Spec rule: do NOT clear local on merge failure. Retry on next load.
      })
      .finally(() => {
        // Flip sync ON regardless of merge outcome — subsequent toggles are
        // best-effort against the server, and the next mount will retry merge
        // if this one failed.
        useCartStore.getState().setSyncEnabled(true);
        useSavedEventsStore.getState().setSyncEnabled(true);
      });
  }, [isAuthenticated, userId, role]);

  return null;
}

async function fetchMe(accessToken: string | null): Promise<User | null> {
  if (!accessToken) return null;
  try {
    const res = await fetch(`${API_BASE_URL}/auth/me`, {
      headers: { Authorization: `Bearer ${accessToken}` },
      signal: AbortSignal.timeout(30_000),
    });
    if (!res.ok) return null;
    const body: ApiResponse<User> = await res.json();
    return body.success ? body.data : null;
  } catch {
    return null;
  }
}
