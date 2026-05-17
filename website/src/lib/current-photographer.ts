import type { User } from "@/types/user";

// Resolves the session user's photographer identity for surfaces that
// filter content by photographerId — notification bell, billing rail,
// file-report modal. Before the 2026-05-17 BE migration this hardcoded
// "photog-cebustride" for every photographer so the mock seed data lit
// up the bell; that mapping silently broke notifications for real BE
// users (their photographerId was their real UUID but the bell filtered
// by the mock id and never matched).
//
// Now: returns the session user's real id + name. Handle is best-effort
// from the photographer-settings store when the caller passes it in (the
// store reflects the photographer's most recent /dashboard/settings put);
// when the caller doesn't have it on hand, leave it null and the surface
// falls back to the brand/name display.

export interface CurrentPhotographer {
  id: string;
  name: string;
  handle: string | null;
}

export function resolveCurrentPhotographer(
  user: User | null,
  handle: string | null = null,
): CurrentPhotographer | null {
  if (!user || user.role !== "PHOTOGRAPHER") return null;
  return {
    id: user.id,
    name: user.name,
    handle: handle?.trim() || null,
  };
}
