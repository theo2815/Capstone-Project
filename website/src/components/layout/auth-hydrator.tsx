"use client";

import { useEffect, useRef } from "react";
import { useAuthStore } from "@/store/auth-store";
import { useCartStore } from "@/store/cart-store";
import { useSavedEventsStore } from "@/store/saved-events-store";
import {
  clearTokens,
  getAccessToken,
  getRefreshToken,
  setTokens,
} from "@/lib/auth";
import { API_BASE_URL } from "@/lib/constants";
import { mergeCart } from "@/lib/api-cart";
import { mergeSavedEvents } from "@/lib/api-saved-events";
import type { ApiResponse } from "@/types/api";
import type { User } from "@/types/user";

export function AuthHydrator() {
  const setUser = useAuthStore((s) => s.setUser);
  const setLoading = useAuthStore((s) => s.setLoading);
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const userId = useAuthStore((s) => s.user?.id);

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
        const refreshedAccess = await refreshAccess(refreshToken);
        if (cancelled) return;
        if (refreshedAccess) {
          user = await fetchMe(refreshedAccess);
          if (cancelled) return;
        }
      }

      if (user) {
        setUser(user);
      } else {
        clearTokens();
        setLoading(false);
      }
    }

    hydrate();
    return () => {
      cancelled = true;
    };
  }, [setUser, setLoading]);

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
  }, [isAuthenticated, userId]);

  return null;
}

async function fetchMe(accessToken: string | null): Promise<User | null> {
  if (!accessToken) return null;
  try {
    const res = await fetch(`${API_BASE_URL}/auth/me`, {
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    if (!res.ok) return null;
    const body: ApiResponse<User> = await res.json();
    return body.success ? body.data : null;
  } catch {
    return null;
  }
}

async function refreshAccess(refreshToken: string): Promise<string | null> {
  try {
    const res = await fetch(`${API_BASE_URL}/auth/refresh`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refreshToken }),
    });
    if (!res.ok) return null;
    const body: ApiResponse<{ accessToken: string; refreshToken: string }> =
      await res.json();
    if (!body.success) return null;
    setTokens(body.data.accessToken, body.data.refreshToken);
    return body.data.accessToken;
  } catch {
    return null;
  }
}
