import { api } from "@/lib/api";

// Saved-events backend contract (Q-003 RESOLVED 2026-05-09).
//   GET    /api/v1/me/saved-events            → string[] (event IDs)
//   POST   /api/v1/me/saved-events            { eventId } → { savedAt }
//   DELETE /api/v1/me/saved-events/{eventId}  → { removed }
//   POST   /api/v1/me/saved-events/merge      { eventIds } → string[]

export async function fetchSavedEvents(): Promise<string[]> {
  return api.get<string[]>("/me/saved-events");
}

export async function postSaveEvent(eventId: string): Promise<void> {
  await api.post<{ savedAt: string }>("/me/saved-events", { eventId });
}

export async function postUnsaveEvent(eventId: string): Promise<void> {
  await api.delete<{ removed: boolean }>(
    `/me/saved-events/${encodeURIComponent(eventId)}`,
  );
}

export async function mergeSavedEvents(
  localIds: string[],
): Promise<string[]> {
  if (localIds.length === 0) return api.get<string[]>("/me/saved-events");
  return api.post<string[]>("/me/saved-events/merge", { eventIds: localIds });
}
