import { create } from "zustand";
import type { OAuthIdentity, User } from "@/types/user";

interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  // A Google identity that still needs a role pick on /onboarding. Held in
  // memory only — a hard refresh loses it, and the onboarding page bounces
  // back to /login where one more button click restores it. Completing the
  // pick goes through useAuth.completeOnboarding (a real backend call), so
  // this store never fabricates a session.
  pendingOAuth: OAuthIdentity | null;
  setUser: (user: User | null) => void;
  setLoading: (loading: boolean) => void;
  setPendingOAuth: (identity: OAuthIdentity | null) => void;
  logout: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  isAuthenticated: false,
  isLoading: true,
  pendingOAuth: null,
  setUser: (user) =>
    set({ user, isAuthenticated: !!user, isLoading: false }),
  setLoading: (isLoading) => set({ isLoading }),
  setPendingOAuth: (pendingOAuth) => set({ pendingOAuth, isLoading: false }),
  logout: () =>
    set({
      user: null,
      isAuthenticated: false,
      isLoading: false,
      pendingOAuth: null,
    }),
}));
