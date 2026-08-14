import { getAccessToken, getRefreshToken, setTokens, clearTokens } from "./auth";
import { API_BASE_URL } from "./constants";
import { buildLoginRedirect } from "./redirect";
import type { ApiResponse } from "@/types/api";

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  async fetch<T>(path: string, options?: RequestInit): Promise<T> {
    const token = getAccessToken();
    const headers: HeadersInit = {
      ...(options?.body instanceof FormData
        ? {}
        : { "Content-Type": "application/json" }),
      ...(token && { Authorization: `Bearer ${token}` }),
      ...options?.headers,
    };

    const res = await fetch(`${this.baseUrl}${path}`, {
      ...options,
      headers,
    });

    if (res.status === 401 && !isPublicAuthEndpoint(path)) {
      const refreshed = await refreshAccessToken();
      if (refreshed) return this.fetch<T>(path, options);
      this.redirectToLogin();
      throw new Error("Unauthorized");
    }

    const data: ApiResponse<T> = await res.json();
    if (!data.success) {
      throw new ApiError(
        data.errors ?? [{ code: "UNKNOWN", message: "Request failed" }],
        res.status,
      );
    }
    return data.data;
  }

  async get<T>(path: string): Promise<T> {
    return this.fetch<T>(path, { method: "GET" });
  }

  async post<T>(
    path: string,
    body?: unknown,
    init?: { headers?: HeadersInit },
  ): Promise<T> {
    return this.fetch<T>(path, {
      method: "POST",
      body: body instanceof FormData ? body : JSON.stringify(body),
      headers: init?.headers,
    });
  }

  async put<T>(path: string, body: unknown): Promise<T> {
    return this.fetch<T>(path, {
      method: "PUT",
      body: JSON.stringify(body),
    });
  }

  async delete<T>(path: string): Promise<T> {
    return this.fetch<T>(path, { method: "DELETE" });
  }

  private redirectToLogin(): void {
    if (typeof window !== "undefined") {
      clearTokens();
      const currentUrl = window.location.pathname + window.location.search;
      window.location.href = buildLoginRedirect(currentUrl);
    }
  }
}

let inFlightRefresh: Promise<string | null> | null = null;

// Single-flight refresh, shared by every caller in this JS context.
//
// The backend ROTATES on refresh — `RefreshTokenService.validateAndRotate`
// revokes the presented token before issuing the replacement. So two
// concurrent refreshes carrying the same plaintext mean the second one is
// rejected as already-revoked, and its caller tears down a perfectly valid
// session (ApiClient bounces to /login; AuthHydrator clears tokens).
//
// The common trigger is NOT two tabs — it is one page load with an expired
// access token: <AuthHydrator> refreshes while the page's React Query hooks
// each 401 and refresh alongside it. Funnelling every caller through one
// promise means exactly one POST /auth/refresh per expiry.
export function refreshAccessToken(): Promise<string | null> {
  if (inFlightRefresh) return inFlightRefresh;
  inFlightRefresh = doRefresh().finally(() => {
    inFlightRefresh = null;
  });
  return inFlightRefresh;
}

async function doRefresh(): Promise<string | null> {
  const refreshToken = getRefreshToken();
  if (!refreshToken) return null;

  const accessToken = await postRefresh(refreshToken);
  if (accessToken) return accessToken;

  // Cross-tab fallback. The single-flight promise above is per-JS-context, so
  // another tab can still rotate out from under us. If storage now holds a
  // different token than the one we just sent, that tab already succeeded —
  // retry once with the fresh value rather than bouncing a live session.
  const current = getRefreshToken();
  if (current && current !== refreshToken) return postRefresh(current);
  return null;
}

async function postRefresh(refreshToken: string): Promise<string | null> {
  try {
    const res = await fetch(`${API_BASE_URL}/auth/refresh`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refreshToken }),
    });

    if (!res.ok) return null;

    const data: ApiResponse<{
      accessToken: string;
      refreshToken: string;
    }> = await res.json();

    if (!data.success) return null;
    setTokens(data.data.accessToken, data.data.refreshToken);
    return data.data.accessToken;
  } catch {
    return null;
  }
}

const PUBLIC_AUTH_ENDPOINTS = new Set([
  "/auth/login",
  "/auth/register",
  "/auth/refresh",
  "/auth/forgot-password",
  "/auth/reset-password",
  "/auth/logout",
]);

function isPublicAuthEndpoint(path: string): boolean {
  const [pathname] = path.split("?");
  return PUBLIC_AUTH_ENDPOINTS.has(pathname);
}

export class ApiError extends Error {
  errors: { code: string; message: string; field?: string }[];
  status?: number;

  constructor(
    errors: { code: string; message: string; field?: string }[],
    status?: number,
  ) {
    super(errors[0]?.message ?? "Unknown error");
    this.errors = errors;
    this.status = status;
  }
}

export const api = new ApiClient(API_BASE_URL);
