"use client";

import { useSearchParams } from "next/navigation";
import { isSafeRedirect } from "@/lib/redirect";
import { ROUTES } from "@/lib/constants";

// Reads `?redirect=<path>` off the current URL and returns it (after safety
// check) or `fallback`. Lives in `hooks/` rather than `lib/redirect.ts` so
// the redirect helpers stay server-importable — `lib/api.ts` imports
// `buildLoginRedirect` and that file must NOT pull `useSearchParams` into
// the server bundle.
export function useRedirectTarget(fallback: string = ROUTES.HOME): string {
  const params = useSearchParams();
  const raw = params?.get("redirect");
  return isSafeRedirect(raw) ? raw : fallback;
}

// Reads `?next=<path>` — the "come back here when you're done" convention the
// event page uses when it sends a runner off to build a selfie library
// (`/profile?next=/events/<slug>#selfies` from PhotoAlertToggle and the
// selfie-search tip). Null when absent or unsafe.
export function useNextTarget(): string | null {
  const params = useSearchParams();
  const raw = params?.get("next");
  return isSafeRedirect(raw) ? raw : null;
}
