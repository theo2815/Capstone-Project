import { ROUTES } from "@/lib/constants";
import type { Role } from "@/types/user";

// Pure helpers — server-safe. Hook variant lives at
// `hooks/use-redirect-target.ts` so importing `useSearchParams` doesn't leak
// into the server bundle via `lib/api.ts`.

export function isSafeRedirect(value: string | null | undefined): value is string {
  if (!value) return false;
  if (!value.startsWith("/")) return false;
  if (value.startsWith("//") || value.startsWith("/\\")) return false;
  // The prefix checks alone are bypassable: the WHATWG parser strips
  // TAB/CR/LF before parsing and folds `\` into `/`, so `/%09/evil.com`
  // decodes to `/<TAB>/evil.com`, passes both guards above, and navigates
  // to `//evil.com`. Resolve against a fixed base — anything that escapes
  // that base's origin is not a relative path.
  try {
    return new URL(value, "http://x").origin === "http://x";
  } catch {
    return false;
  }
}

// The current view, in the form `buildLoginRedirect` wants. The hash is part
// of "the exact view they were on": every slab deep-link carries one
// (`/account#password` from the IdentityRail, `/dashboard/settings#payout`
// from the billing CTA, `/profile#selfies`), and dropping it landed the user
// back at the top of a long page after signing in.
//
// Guarded rather than assumed — `lib/api.ts` imports this module, and that
// import must not pull anything browser-only into the server bundle. Returns
// "" server-side, which `buildLoginRedirect` already treats as "no redirect".
export function currentUrlForRedirect(): string {
  if (typeof window === "undefined") return "";
  const { pathname, search, hash } = window.location;
  return pathname + search + hash;
}

// Builds the `/login?redirect=...` URL used by `<ProtectedRoute>` and the
// ApiClient 401 handler. `currentUrl` should come from
// `currentUrlForRedirect()` so the user returns to the exact view they were
// on. Skips the redirect param entirely when already on `/login` (defense
// against loops).
export function buildLoginRedirect(currentUrl: string): string {
  if (!currentUrl || currentUrl.startsWith(ROUTES.LOGIN)) return ROUTES.LOGIN;
  return `${ROUTES.LOGIN}?redirect=${encodeURIComponent(currentUrl)}`;
}

// Default landing route per role. Used as the FALLBACK by login + register
// forms when no preserved `?redirect=…` is present. The preserved redirect
// wins when set so a guest-bounced-from-protected-link flow lands back on
// the original page, not the role default.
//
//   RUNNER        → /events       (browse races)
//   PHOTOGRAPHER  → /dashboard    (workspace overview)
//   ADMIN         → /admin/overview (KPI dashboard)
export function roleHome(role: Role): string {
  switch (role) {
    case "ADMIN":
      return ROUTES.ADMIN_OVERVIEW;
    case "PHOTOGRAPHER":
      return ROUTES.DASHBOARD;
    case "RUNNER":
      return ROUTES.EVENTS;
  }
}
