import { getAccessToken, getRefreshToken, setTokens, clearTokens } from "./auth";
import { API_BASE_URL, ROUTES } from "./constants";
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

    if (res.status === 401) {
      const refreshed = await this.refreshToken();
      if (refreshed) return this.fetch<T>(path, options);
      this.redirectToLogin();
      throw new Error("Unauthorized");
    }

    const data: ApiResponse<T> = await res.json();
    if (!data.success) {
      throw new ApiError(
        data.errors ?? [{ code: "UNKNOWN", message: "Request failed" }],
      );
    }
    return data.data;
  }

  async get<T>(path: string): Promise<T> {
    return this.fetch<T>(path, { method: "GET" });
  }

  async post<T>(path: string, body?: unknown): Promise<T> {
    return this.fetch<T>(path, {
      method: "POST",
      body: body instanceof FormData ? body : JSON.stringify(body),
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

  private async refreshToken(): Promise<boolean> {
    const refreshToken = getRefreshToken();
    if (!refreshToken) return false;

    try {
      const res = await fetch(`${this.baseUrl}/auth/refresh`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refreshToken }),
      });

      if (!res.ok) return false;

      const data: ApiResponse<{
        accessToken: string;
        refreshToken: string;
      }> = await res.json();

      if (data.success) {
        setTokens(data.data.accessToken, data.data.refreshToken);
        return true;
      }
      return false;
    } catch {
      return false;
    }
  }

  private redirectToLogin(): void {
    if (typeof window !== "undefined") {
      clearTokens();
      window.location.href = ROUTES.LOGIN;
    }
  }
}

export class ApiError extends Error {
  errors: { code: string; message: string; field?: string }[];

  constructor(errors: { code: string; message: string; field?: string }[]) {
    super(errors[0]?.message ?? "Unknown error");
    this.errors = errors;
  }
}

export const api = new ApiClient(API_BASE_URL);
