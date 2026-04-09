"use client";

import { useCallback } from "react";
import { useAuthStore } from "@/store/auth-store";
import { api } from "@/lib/api";
import { setTokens, clearTokens } from "@/lib/auth";
import type { AuthResponse, LoginRequest, RegisterRequest } from "@/types/user";

export function useAuth() {
  const { user, isAuthenticated, isLoading, setUser, logout: clearUser } =
    useAuthStore();

  const login = useCallback(
    async (credentials: LoginRequest) => {
      const data = await api.post<AuthResponse>("/auth/login", credentials);
      setTokens(data.accessToken, data.refreshToken);
      setUser(data.user);
    },
    [setUser],
  );

  const register = useCallback(
    async (payload: RegisterRequest) => {
      const data = await api.post<AuthResponse>("/auth/register", payload);
      setTokens(data.accessToken, data.refreshToken);
      setUser(data.user);
    },
    [setUser],
  );

  const logout = useCallback(() => {
    clearTokens();
    clearUser();
  }, [clearUser]);

  return { user, isAuthenticated, isLoading, login, register, logout };
}
