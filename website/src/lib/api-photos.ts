import { api } from "@/lib/api";
import { BACKEND_LIVE } from "@/lib/backend-flag";
import { getEventBySlug } from "@/app/events/mock-events";
import { generateMockPhotos, type MockPhoto } from "@/app/events/[slug]/mock-photos";
import type { PaginatedResponse } from "@/types/api";

export type Photo = MockPhoto;

export interface EventPhotosQuery {
  bib?: string;
  offset?: number;
  limit?: number;
}

export interface EventPhotosResult {
  items: Photo[];
  total: number;
  offset: number;
  limit: number;
}

function normalizeBib(raw: string | undefined): string {
  if (!raw) return "";
  return raw.replace(/^B-/i, "").trim().toUpperCase();
}

export async function fetchEventPhotos(
  slug: string,
  query: EventPhotosQuery = {},
): Promise<EventPhotosResult> {
  const offset = query.offset ?? 0;
  const limit = query.limit ?? 200;

  if (BACKEND_LIVE) {
    const params = new URLSearchParams();
    if (query.bib) params.set("bib", query.bib);
    params.set("offset", String(offset));
    params.set("limit", String(limit));
    const res = await api.get<PaginatedResponse<Photo>>(
      `/events/${encodeURIComponent(slug)}/photos?${params.toString()}`,
    );
    return res;
  }

  // Mock fallback — generate from event seed and apply bib filter client-side.
  const event = getEventBySlug(slug);
  if (!event) {
    return { items: [], total: 0, offset, limit };
  }
  const all = generateMockPhotos(event);
  const cleaned = normalizeBib(query.bib);
  const filtered = cleaned
    ? all.filter((p) => p.bib && p.bib.replace(/^B-/, "").includes(cleaned))
    : all;
  const slice = filtered.slice(offset, offset + limit);
  return { items: slice, total: filtered.length, offset, limit };
}

export interface SearchByFaceArgs {
  selfieId?: string;
  selfieFile?: File;
  offset?: number;
  limit?: number;
}

export async function searchEventByFace(
  slug: string,
  args: SearchByFaceArgs,
): Promise<EventPhotosResult> {
  const offset = args.offset ?? 0;
  const limit = args.limit ?? 200;

  if (BACKEND_LIVE) {
    const path = `/events/${encodeURIComponent(slug)}/photos/search-by-face`;
    if (args.selfieFile) {
      const form = new FormData();
      form.append("selfie", args.selfieFile);
      form.append("offset", String(offset));
      form.append("limit", String(limit));
      const res = await api.post<PaginatedResponse<Photo>>(path, form);
      return res;
    }
    if (!args.selfieId) {
      throw new Error("searchEventByFace requires selfieId or selfieFile");
    }
    const res = await api.post<PaginatedResponse<Photo>>(path, {
      selfieId: args.selfieId,
      offset,
      limit,
    });
    return res;
  }

  // Mock fallback — pretend the user's primary selfie matches DEMO_BIB photos.
  const event = getEventBySlug(slug);
  if (!event) return { items: [], total: 0, offset, limit };
  const all = generateMockPhotos(event);
  const matched = all.filter((p) => p.bib === "B-4082");
  const slice = matched.slice(offset, offset + limit);
  return { items: slice, total: matched.length, offset, limit };
}
