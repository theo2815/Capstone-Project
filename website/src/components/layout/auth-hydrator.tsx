"use client";

import { useEffect } from "react";
import { useAuthStore } from "@/store/auth-store";
import {
  clearTokens,
  getAccessToken,
  getRefreshToken,
  setTokens,
} from "@/lib/auth";
import { API_BASE_URL } from "@/lib/constants";
import type { ApiResponse } from "@/types/api";
import type { User } from "@/types/user";

export function AuthHydrator() {
  const setUser = useAuthStore((s) => s.setUser);
  const setLoading = useAuthStore((s) => s.setLoading);

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
