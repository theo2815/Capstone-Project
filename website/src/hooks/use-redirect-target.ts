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
