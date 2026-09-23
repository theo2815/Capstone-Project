import { create } from "zustand";
import type { OAuthIdentity, User } from "@/types/user";

interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  // Changes on every fresh authentication and identity transition.
  // Async mutations use it to discard responses from an earlier session.
  sessionVersion: number;
  // A Google identity that still needs a role pick on /onboarding. Held in
  // memory only — a hard refresh loses it, and the onboarding page bounces
  // back to /login where one more button click restores it. Completing the
  // pick goes through useAuth.completeOnboarding (a real backend call), so
  // this store never fabricates a session.
  pendingOAuth: OAuthIdentity | null;
  pendingMfa: string | null;
  setUser: (user: User | null) => void;
  startSession: (user: User) => void;
  setLoading: (loading: boolean) => void;
  setPendingOAuth: (identity: OAuthIdentity | null) => void;
  setPendingMfa: (token: string | null) => void;
  logout: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  isAuthenticated: false,
  isLoading: true,
  sessionVersion: 0,
  pendingOAuth: null,
  pendingMfa: null,
  setUser: (user) =>
    set((s) => ({
      user, isAuthenticated: !!user, isLoading: false, pendingMfa: null,
      sessionVersion: s.sessionVersion + (s.user?.id !== user?.id ? 1 : 0),
    })),
  startSession: (user) =>
    set((s) => ({
      user, isAuthenticated: true, isLoading: false, pendingMfa: null,
      sessionVersion: s.sessionVersion + 1,
    })),
  setLoading: (isLoading) => set({ isLoading }),
  setPendingOAuth: (pendingOAuth) => set({ pendingOAuth, isLoading: false }),
  setPendingMfa: (pendingMfa) => set({ pendingMfa }),
  logout: () =>
    set((s) => ({
      user: null,
      isAuthenticated: false,
      isLoading: false,
      pendingOAuth: null,
      pendingMfa: null,
      sessionVersion: s.sessionVersion + 1,
    })),
}));
