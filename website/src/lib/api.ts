import { getAccessToken, getRefreshToken, setTokens, clearTokens } from "./auth";
import { API_BASE_URL } from "./constants";
import { buildLoginRedirect, currentUrlForRedirect } from "./redirect";
import type { ApiResponse } from "@/types/api";

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  // Full request path (auth header, 30s/120s timeout, single-flight 401 refresh
  // + one retry). Returns the parsed envelope data AND the raw Response so
  // callers needing a header (getWithTotal → X-Total-Count) can read it.
  async fetchRaw<T>(
    path: string,
    options?: RequestInit,
    // Set on the one retry we allow after a token refresh. A second refresh
    // can never help: if a freshly-minted token still 401s, a newer one will
    // too — and unbounded recursion here re-submits the request forever.
    retried = false,
  ): Promise<{ data: T; res: Response }> {
    const token = getAccessToken();
    const isFormData = options?.body instanceof FormData;
    const headers: HeadersInit = {
      ...(isFormData ? {} : { "Content-Type": "application/json" }),
      ...(token && { Authorization: `Bearer ${token}` }),
      ...options?.headers,
    };

    // Nothing here carried a signal, so a stalled connection hung forever —
    // React Query never saw the promise settle and skeletons held
    // indefinitely. FormData gets the long budget: a 10 MB cover on venue
    // Wi-Fi is slow, not dead.
    let res: Response;
    try {
      res = await fetch(`${this.baseUrl}${path}`, {
        ...options,
        headers,
        signal:
          options?.signal ??
          AbortSignal.timeout(isFormData ? 120_000 : 30_000),
      });
    } catch (e) {
      if (
        e instanceof DOMException &&
        (e.name === "TimeoutError" || e.name === "AbortError")
      ) {
        throw new ApiError([
          {
            code: "TIMEOUT",
            message: "Request timed out. Check your connection and try again.",
          },
        ]);
      }
      throw e;
    }

    // Parsed before the 401 branch, because an expired session and a rejected
    // password both arrive as 401 and only the envelope's code tells them
    // apart. Guarded: a 401 raised outside the app (proxy, gateway) may carry
    // no JSON at all, and that still has to reach the refresh path.
    let data: ApiResponse<T> | null = null;
    try {
      data = (await res.json()) as ApiResponse<T>;
    } catch {
      data = null;
    }

    if (
      res.status === 401 &&
      !isPublicAuthEndpoint(path) &&
      !isCredentialRejection(data)
    ) {
      const refreshed = retried ? false : await refreshAccessToken();
      if (refreshed) return this.fetchRaw<T>(path, options, true);
      this.redirectToLogin();
      throw new Error("Unauthorized");
    }

    if (!data) {
      throw new ApiError(
        [{ code: "UNKNOWN", message: "Request failed" }],
        res.status,
        retryAfterSeconds(res),
      );
    }
    if (!data.success) {
      throw new ApiError(
        data.errors ?? [{ code: "UNKNOWN", message: "Request failed" }],
        res.status,
        retryAfterSeconds(res),
      );
    }
    return { data: data.data, res };
  }

  async fetch<T>(path: string, options?: RequestInit): Promise<T> {
    return (await this.fetchRaw<T>(path, options)).data;
  }

  async get<T>(path: string): Promise<T> {
    return this.fetch<T>(path, { method: "GET" });
  }

  // GET that also reads the X-Total-Count response header (CORS-exposed) for
  // endpoints whose body stays a bare array for mobile parity (message inboxes).
  // `total` is null when the header is absent or unparseable.
  async getWithTotal<T>(
    path: string,
  ): Promise<{ data: T; total: number | null }> {
    const { data, res } = await this.fetchRaw<T>(path, { method: "GET" });
    const raw = res.headers.get("X-Total-Count");
    const parsed = raw == null ? NaN : Number(raw);
    return { data, total: Number.isFinite(parsed) ? parsed : null };
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
      window.location.href = buildLoginRedirect(currentUrlForRedirect());
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
      signal: AbortSignal.timeout(30_000),
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
  // Opened from a link in the NEW inbox, so the caller is frequently signed
  // out — or signed in with a stale token, since confirming revokes every
  // session. Either way a 401 here must surface as an error on the page, not
  // bounce the user to /login and swallow their confirmation.
  "/auth/confirm-email-change",
  // Opened from the registration mail, often on a different device than the
  // one that registered. A signed-out caller must see the page's own error,
  // not get bounced to /login with the token silently spent.
  "/auth/verify-email",
  "/auth/logout",
]);

function isPublicAuthEndpoint(path: string): boolean {
  const [pathname] = path.split("?");
  return PUBLIC_AUTH_ENDPOINTS.has(pathname);
}

// A 401 carrying INVALID_CREDENTIALS is the server rejecting a password the
// user just typed — `PUT /me/password` and `PUT /me/email` both re-verify the
// current password and throw UnauthorizedException on a mismatch. That is not
// an expired session, and treating it as one is a trap: the refresh SUCCEEDS
// (the session is fine), the retry re-submits the same wrong password, and the
// pair loops forever. Observed live at 58 requests before the tab was closed.
// Letting it fall through as an ApiError is also what makes the forms' own
// INVALID_CREDENTIALS branches reachable — they never were.
function isCredentialRejection<T>(data: ApiResponse<T> | null): boolean {
  return data?.errors?.some((e) => e.code === "INVALID_CREDENTIALS") ?? false;
}

// Every backend 429 — rate-limit bucket and account lockout alike — carries
// Retry-After in seconds, and CorsConfig exposes it to browser JS. Parse it
// once here so a catch site can say "try again in ~N s" instead of guessing.
function retryAfterSeconds(res: Response): number | undefined {
  const raw = res.headers.get("Retry-After");
  if (!raw) return undefined;
  const n = Number(raw);
  return Number.isFinite(n) && n >= 0 ? n : undefined;
}

// Human copy for ApiError.retryAfterSeconds ("42 seconds" / "3 minutes").
export function formatRetryWait(seconds: number): string {
  if (seconds < 90) return `${Math.max(1, Math.ceil(seconds))} seconds`;
  return `${Math.ceil(seconds / 60)} minutes`;
}

export class ApiError extends Error {
  errors: { code: string; message: string; field?: string }[];
  status?: number;
  // Seconds from a 429's Retry-After header; undefined on every other status.
  retryAfterSeconds?: number;

  constructor(
    errors: { code: string; message: string; field?: string }[],
    status?: number,
    retryAfterSeconds?: number,
  ) {
    super(errors[0]?.message ?? "Unknown error");
    this.errors = errors;
    this.status = status;
    this.retryAfterSeconds = retryAfterSeconds;
  }
}

export const api = new ApiClient(API_BASE_URL);
