import { api } from "@/lib/api";
import type { MockPhoto } from "@/types/photo";
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

export async function fetchEventPhotos(
  slug: string,
  query: EventPhotosQuery = {},
): Promise<EventPhotosResult> {
  const offset = query.offset ?? 0;
  const limit = query.limit ?? 200;

  const params = new URLSearchParams();
  if (query.bib) params.set("bib", query.bib);
  params.set("offset", String(offset));
  params.set("limit", String(limit));
  const res = await api.get<PaginatedResponse<Photo>>(
    `/events/${encodeURIComponent(slug)}/photos?${params.toString()}`,
  );
  return res;
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

  const path = `/events/${encodeURIComponent(slug)}/photos/search-by-face`;
  if (args.selfieFile) {
    const form = new FormData();
    form.append("selfie", args.selfieFile);
    form.append("offset", String(offset));
    form.append("limit", String(limit));
    return api.post<PaginatedResponse<Photo>>(path, form);
  }
  if (!args.selfieId) {
    throw new Error("searchEventByFace requires selfieId or selfieFile");
  }
  return api.post<PaginatedResponse<Photo>>(path, {
    selfieId: args.selfieId,
    offset,
    limit,
  });
}
