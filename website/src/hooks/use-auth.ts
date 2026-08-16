"use client";

import { useCallback } from "react";
import { useAuthStore } from "@/store/auth-store";
import { api } from "@/lib/api";
import { getRefreshToken, setTokens, clearTokens } from "@/lib/auth";
import { captureGuestBuffer, resetUserScopedStores } from "@/lib/auth-reset";
import type {
  AuthResponse,
  LoginRequest,
  OAuthIdentity,
  RegisterRequest,
  Role,
} from "@/types/user";

export function useAuth() {
  const {
    user,
    isAuthenticated,
    isLoading,
    pendingOAuth,
    setUser,
    setPendingOAuth,
    completeOnboarding: completeOnboardingInStore,
    logout: clearUser,
  } = useAuthStore();

  // Returns the freshly-authed user so callers can route by role without
  // racing the React render of `useAuthStore.user`. Zustand's setState is
  // synchronous, so getState() would also work — but returning the value
  // keeps the dataflow explicit at the call site.
  const login = useCallback(
    async (credentials: LoginRequest) => {
      const data = await api.post<AuthResponse>("/auth/login", credentials);
      // Clear any leftover state from a previous user in this browser BEFORE
      // setting the new user — prevents User A's photographer settings,
      // cart, etc. from bleeding into User B's session. The guest's own cart
      // and bookmarks are carried across the wipe so <AuthHydrator>'s merge
      // still has something to merge — see captureGuestBuffer().
      const restoreGuestBuffer = captureGuestBuffer();
      resetUserScopedStores();
      restoreGuestBuffer();
      setTokens(data.accessToken, data.refreshToken);
      setUser(data.user);
      return data.user;
    },
    [setUser],
  );

  const register = useCallback(
    async (payload: RegisterRequest) => {
      const data = await api.post<AuthResponse>("/auth/register", payload);
      const restoreGuestBuffer = captureGuestBuffer();
      resetUserScopedStores();
      restoreGuestBuffer();
      setTokens(data.accessToken, data.refreshToken);
      setUser(data.user);
      return data.user;
    },
    [setUser],
  );

  const mockGoogleLogin = useCallback(async () => {
    await new Promise((resolve) => setTimeout(resolve, 700));
    const identity: OAuthIdentity = {
      provider: "GOOGLE",
      sub: `google-${Date.now()}`,
      email: "juan.delacruz@gmail.com",
      name: "Juan dela Cruz",
    };
    setPendingOAuth(identity);
    return identity;
  }, [setPendingOAuth]);

  // Third auth entry point alongside login/register, and it authenticates a
  // user the same way — so it owes the same wipe. Unreachable while
  // OAUTH_ENABLED is false (google-button.tsx), but the reset belongs with the
  // transition, not with whoever wires real OAuth later.
  const completeOnboarding = useCallback(
    (role: Role) => {
      resetUserScopedStores();
      return completeOnboardingInStore(role);
    },
    [completeOnboardingInStore],
  );

  const cancelOnboarding = useCallback(() => {
    setPendingOAuth(null);
  }, [setPendingOAuth]);

  const logout = useCallback(() => {
    // Best-effort refresh-token revocation per Q-009 ADR (2026-05-09). Fire
    // and forget — the FE clears state regardless of network outcome so
    // logout stays fast and offline-friendly.
    const refreshToken = getRefreshToken();
    if (refreshToken) {
      api.post("/auth/logout", { refreshToken }).catch(() => {
        /* best-effort: server-side revoke is nice-to-have, not blocking */
      });
    }
    clearTokens();
    // Reset BEFORE clearUser so cart/saved-events still see syncEnabled state
    // correctly when their clear() runs (resetUserScopedStores flips
    // syncEnabled false first to skip the spurious BE clear call).
    resetUserScopedStores();
    clearUser();
  }, [clearUser]);

  return {
    user,
    isAuthenticated,
    isLoading,
    pendingOAuth,
    login,
    register,
    mockGoogleLogin,
    completeOnboarding,
    cancelOnboarding,
    logout,
  };
}
