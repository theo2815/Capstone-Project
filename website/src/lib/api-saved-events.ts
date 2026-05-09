import { api } from "@/lib/api";
import { BACKEND_LIVE } from "@/lib/backend-flag";

// Saved-events backend contract (Q-003 RESOLVED 2026-05-09).
//   GET    /api/v1/me/saved-events            → string[] (event IDs)
//   POST   /api/v1/me/saved-events            { eventId } → { savedAt }
//   DELETE /api/v1/me/saved-events/{eventId}  → { removed }
//   POST   /api/v1/me/saved-events/merge      { eventIds } → string[]
//
// Mock-mode is a no-op for save/unsave (the local store is the source of
// truth). `merge` returns the union so the AuthHydrator path works without
// the backend.

export async function fetchSavedEvents(): Promise<string[]> {
  if (BACKEND_LIVE) {
    return api.get<string[]>("/me/saved-events");
  }
  return [];
}

export async function postSaveEvent(eventId: string): Promise<void> {
  if (!BACKEND_LIVE) return;
  await api.post<{ savedAt: string }>("/me/saved-events", { eventId });
}

export async function postUnsaveEvent(eventId: string): Promise<void> {
  if (!BACKEND_LIVE) return;
  await api.delete<{ removed: boolean }>(
    `/me/saved-events/${encodeURIComponent(eventId)}`,
  );
}

export async function mergeSavedEvents(
  localIds: string[],
): Promise<string[]> {
  if (BACKEND_LIVE) {
    if (localIds.length === 0) return api.get<string[]>("/me/saved-events");
    return api.post<string[]>("/me/saved-events/merge", { eventIds: localIds });
  }
  // Mock fallback — local is canonical; nothing to dedupe against.
  return localIds;
}
