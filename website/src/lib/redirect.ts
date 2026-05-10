import { ROUTES } from "@/lib/constants";

// Pure helpers — server-safe. Hook variant lives at
// `hooks/use-redirect-target.ts` so importing `useSearchParams` doesn't leak
// into the server bundle via `lib/api.ts`.

export function isSafeRedirect(value: string | null | undefined): value is string {
  if (!value) return false;
  if (!value.startsWith("/")) return false;
  if (value.startsWith("//") || value.startsWith("/\\")) return false;
  return true;
}

// Builds the `/login?redirect=...` URL used by `<ProtectedRoute>` and the
// ApiClient 401 handler. `currentUrl` should be `pathname + search` so the
// user returns to the exact view they were on. Skips the redirect param
// entirely when already on `/login` (defense against loops).
export function buildLoginRedirect(currentUrl: string): string {
  if (!currentUrl || currentUrl.startsWith(ROUTES.LOGIN)) return ROUTES.LOGIN;
  return `${ROUTES.LOGIN}?redirect=${encodeURIComponent(currentUrl)}`;
}
