import { api } from "@/lib/api";
import type { PhotographerProfile } from "@/lib/photographer-registry";
import type { MockPhoto } from "@/types/photo";
import type { PaginatedResponse } from "@/types/api";

// Phase F.2 photographer-public backend contract
//   Q-016 (public gallery + OG metadata) RESOLVED 2026-05-10. See vault decisions.
//
//   GET /api/v1/public/photographers/{handle}                              → PhotographerProfile
//   GET /api/v1/public/photographers/{handle}/events/{slug}/photos?offset=&limit=
//       → PaginatedResponse<MockPhoto>
//
// No Bearer JWT required (public). Server enforces:
//   • Rate limit: 60 req/min per IP
//   • Reserved-handle check (returns RESERVED_HANDLE on collision attempts)

export async function fetchPublicPhotographer(
  handle: string,
): Promise<PhotographerProfile | null> {
  return api.get<PhotographerProfile>(
    `/public/photographers/${encodeURIComponent(handle)}`,
  );
}

export interface PublicPhotographerPhotosArgs {
  offset?: number;
  limit?: number;
}

export async function fetchPublicPhotographerEventPhotos(
  handle: string,
  eventSlug: string,
  args: PublicPhotographerPhotosArgs = {},
): Promise<PaginatedResponse<MockPhoto>> {
  const p = new URLSearchParams();
  p.set("offset", String(args.offset ?? 0));
  p.set("limit", String(args.limit ?? 24));
  return api.get<PaginatedResponse<MockPhoto>>(
    `/public/photographers/${encodeURIComponent(handle)}/events/${encodeURIComponent(eventSlug)}/photos?${p.toString()}`,
  );
}
