const ACCESS_TOKEN_KEY = "quickpitik_access_token";
const REFRESH_TOKEN_KEY = "quickpitik_refresh_token";

export function getAccessToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(ACCESS_TOKEN_KEY);
}

export function getRefreshToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(REFRESH_TOKEN_KEY);
}

export function setTokens(accessToken: string, refreshToken: string): void {
  localStorage.setItem(ACCESS_TOKEN_KEY, accessToken);
  localStorage.setItem(REFRESH_TOKEN_KEY, refreshToken);
}

export function clearTokens(): void {
  localStorage.removeItem(ACCESS_TOKEN_KEY);
  localStorage.removeItem(REFRESH_TOKEN_KEY);
}

export function decodeBase64Url(input: string): string {
  let base64 = input.replace(/-/g, "+").replace(/_/g, "/");
  const pad = base64.length % 4;
  if (pad === 2) {
    base64 += "==";
  } else if (pad === 3) {
    base64 += "=";
  } else if (pad === 1) {
    throw new Error("Invalid base64url string length");
  }
  return atob(base64);
}

export function isTokenExpired(token: string): boolean {
  try {
    const parts = token.split(".");
    if (parts.length < 2) return true;
    const json = decodeBase64Url(parts[1]);
    const payload = JSON.parse(json);
    if (typeof payload.exp !== "number") return true;
    return payload.exp * 1000 < Date.now();
  } catch {
    return true;
  }
}
