"use client";

import { useCallback } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useAuthStore } from "@/store/auth-store";
import { api, ApiError } from "@/lib/api";
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
  const queryClient = useQueryClient();
  const {
    user,
    isAuthenticated,
    isLoading,
    pendingOAuth,
    setUser,
    setPendingOAuth,
    logout: clearUser,
  } = useAuthStore();

  // The one place a fresh AuthResponse becomes a session — every auth entry
  // point (password login, register, Google) funnels through it.
  //
  // Clear any leftover state from a previous user in this browser BEFORE
  // setting the new user — prevents User A's photographer settings, cart,
  // etc. from bleeding into User B's session. The guest's own cart and
  // bookmarks are carried across the wipe so <AuthHydrator>'s merge still
  // has something to merge — see captureGuestBuffer(). The React Query cache
  // gets the same treatment: it outlives the Zustand wipe (the client is
  // created once in providers.tsx), so with a 60s staleTime User B's first
  // render after a same-tab account switch was served User A's cached
  // /me/orders, selfies, admin queues.
  const establishSession = useCallback(
    (data: AuthResponse) => {
      const restoreGuestBuffer = captureGuestBuffer();
      resetUserScopedStores();
      queryClient.clear();
      restoreGuestBuffer();
      setTokens(data.accessToken, data.refreshToken);
      setUser(data.user);
      return data.user;
    },
    [setUser, queryClient],
  );

  // Returns the freshly-authed user so callers can route by role without
  // racing the React render of `useAuthStore.user`. Zustand's setState is
  // synchronous, so getState() would also work — but returning the value
  // keeps the dataflow explicit at the call site.
  const login = useCallback(
    async (credentials: LoginRequest) => {
      const data = await api.post<AuthResponse>("/auth/login", credentials);
      return establishSession(data);
    },
    [establishSession],
  );

  const register = useCallback(
    async (payload: RegisterRequest) => {
      const data = await api.post<AuthResponse>("/auth/register", payload);
      return establishSession(data);
    },
    [establishSession],
  );

  // Exchange a Google ID token (from the GIS button) for a session. Returns
  // the user, or null when the backend answered 422 ROLE_REQUIRED — a brand
  // new Google account that must pick RUNNER/PHOTOGRAPHER on /onboarding
  // first. In that case the identity (plus the raw token, for the re-POST)
  // is parked in pendingOAuth and <OnboardingGate> takes over routing.
  const googleLogin = useCallback(
    async (idToken: string) => {
      try {
        const data = await api.post<AuthResponse>("/auth/google", { idToken });
        return establishSession(data);
      } catch (err) {
        if (
          err instanceof ApiError &&
          err.errors.some((e) => e.code === "ROLE_REQUIRED")
        ) {
          setPendingOAuth({
            provider: "GOOGLE",
            idToken,
            ...decodeGoogleIdentity(idToken),
          });
          return null;
        }
        throw err;
      }
    },
    [establishSession, setPendingOAuth],
  );

  // Second leg of the new-Google-user flow: re-POST the parked ID token with
  // the picked role. The backend creates the account and answers with the
  // normal pair, so this authenticates exactly like login/register and owes
  // the same establishSession wipe.
  const completeOnboarding = useCallback(
    async (role: Role) => {
      const pending = pendingOAuth;
      if (!pending) {
        throw new Error("completeOnboarding called without pendingOAuth");
      }
      const data = await api.post<AuthResponse>("/auth/google", {
        idToken: pending.idToken,
        role,
      });
      const authedUser = establishSession(data);
      setPendingOAuth(null);
      return authedUser;
    },
    [pendingOAuth, establishSession, setPendingOAuth],
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
    queryClient.clear();
    clearUser();
  }, [clearUser, queryClient]);

  return {
    user,
    isAuthenticated,
    isLoading,
    pendingOAuth,
    login,
    register,
    googleLogin,
    completeOnboarding,
    cancelOnboarding,
    logout,
  };
}

// Display-only parse of the ID token's payload (base64url, not plain base64)
// so /onboarding can show who is signing up. Trust is not a concern here: the
// backend has already verified this exact token's signature — ROLE_REQUIRED
// is only reachable through a verified token — and verifies it again on the
// completing POST before any account is created.
function decodeGoogleIdentity(
  idToken: string,
): Pick<OAuthIdentity, "sub" | "email" | "name" | "avatarUrl"> {
  try {
    const payload = JSON.parse(
      atob(idToken.split(".")[1].replace(/-/g, "+").replace(/_/g, "/")),
    );
    return {
      sub: payload.sub ?? "",
      email: payload.email ?? "",
      name: payload.name ?? payload.email?.split("@")[0] ?? "Google user",
      avatarUrl: payload.picture,
    };
  } catch {
    return { sub: "", email: "", name: "Google user", avatarUrl: undefined };
  }
}
