import { create } from "zustand";
import type { OAuthIdentity, Role, User } from "@/types/user";

interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  pendingOAuth: OAuthIdentity | null;
  setUser: (user: User | null) => void;
  setLoading: (loading: boolean) => void;
  setPendingOAuth: (identity: OAuthIdentity | null) => void;
  completeOnboarding: (role: Role) => User;
  logout: () => void;
}

export const useAuthStore = create<AuthState>((set, get) => ({
  user: null,
  isAuthenticated: false,
  isLoading: true,
  pendingOAuth: null,
  setUser: (user) =>
    set({ user, isAuthenticated: !!user, isLoading: false }),
  setLoading: (isLoading) => set({ isLoading }),
  setPendingOAuth: (pendingOAuth) => set({ pendingOAuth, isLoading: false }),
  completeOnboarding: (role) => {
    const pending = get().pendingOAuth;
    if (!pending) {
      throw new Error("completeOnboarding called without pendingOAuth");
    }
    const user: User = {
      id: `mock-${pending.sub}`,
      email: pending.email,
      name: pending.name,
      role,
      avatarUrl: pending.avatarUrl,
      createdAt: new Date().toISOString(),
    };
    set({
      user,
      isAuthenticated: true,
      isLoading: false,
      pendingOAuth: null,
    });
    return user;
  },
  logout: () =>
    set({
      user: null,
      isAuthenticated: false,
      isLoading: false,
      pendingOAuth: null,
    }),
}));
