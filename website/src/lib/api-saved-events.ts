import { api } from "@/lib/api";

// Saved-events backend contract (Q-003 RESOLVED 2026-05-09, expanded
// 2026-05-19 PM to carry full event summaries so /profile race log can
// render without a second round-trip).
//   GET    /api/v1/me/saved-events            → SavedEventSummary[]
//   POST   /api/v1/me/saved-events            { eventId } → SavedEventSummary
//   DELETE /api/v1/me/saved-events/{eventId}  → { removed }
//   POST   /api/v1/me/saved-events/merge      { eventIds } → SavedEventSummary[]

export type SavedEventState = "upcoming" | "live" | "open" | "past";

export interface SavedEventSummary {
  id: string;
  slug: string;
  name: string;
  date: string;
  state: SavedEventState;
  bannerUrl: string | null;
  location: string;
  savedAt: string;
}

export async function fetchSavedEvents(): Promise<SavedEventSummary[]> {
  return api.get<SavedEventSummary[]>("/me/saved-events");
}

export async function postSaveEvent(
  eventId: string,
): Promise<SavedEventSummary> {
  return api.post<SavedEventSummary>("/me/saved-events", { eventId });
}

export async function postUnsaveEvent(eventId: string): Promise<void> {
  await api.delete<{ removed: boolean }>(
    `/me/saved-events/${encodeURIComponent(eventId)}`,
  );
}

export async function mergeSavedEvents(
  localIds: string[],
): Promise<SavedEventSummary[]> {
  if (localIds.length === 0) return api.get<SavedEventSummary[]>("/me/saved-events");
  return api.post<SavedEventSummary[]>("/me/saved-events/merge", {
    eventIds: localIds,
  });
}
