import { API_BASE_URL } from "@/lib/constants";

// Build a WebSocket URL from an absolute server path (e.g. "/ws/admin/notifications").
//
// Prefer NEXT_PUBLIC_WS_URL when set (already carries the ws:// protocol
// and the /ws suffix per .env.local). Otherwise derive from API_BASE_URL's
// origin only — API_BASE_URL carries the REST /api/v1 prefix which the
// BE WebSocket handlers do NOT live under, so we must strip the path
// component before appending the WS path.
export function buildWsUrl(path: string): string {
  if (!path.startsWith("/")) {
    throw new Error(`buildWsUrl path must be absolute, got: ${path}`);
  }
  const wsEnv = process.env.NEXT_PUBLIC_WS_URL;
  if (wsEnv) {
    // Env value ends in /ws; strip it so we can re-append the full path
    // (which itself starts with /ws/...).
    return wsEnv.replace(/\/ws\/?$/, "") + path;
  }
  const apiUrl = new URL(API_BASE_URL);
  const protocol = apiUrl.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${apiUrl.host}${path}`;
}
