import { api } from "@/lib/api";
import type { PhotographerMessage } from "@/lib/photographer-messages";

// Photographer-facing inbox API. Mirrors the BE
// MePhotographerMessagesController endpoints. All requests carry the JWT
// bearer; BE filters by the principal's userId so a runner / unauth
// caller never reaches this surface (403 via @PreAuthorize).

export async function fetchMyPhotographerMessages(): Promise<PhotographerMessage[]> {
  return api.get<PhotographerMessage[]>("/me/photographer/messages");
}

export async function markMyPhotographerMessageRead(
  id: string,
): Promise<PhotographerMessage> {
  return api.fetch<PhotographerMessage>(
    `/me/photographer/messages/${encodeURIComponent(id)}/read`,
    { method: "PATCH" },
  );
}

export interface MarkAllReadResponse {
  markedRead: number;
}

export async function markAllMyPhotographerMessagesRead(): Promise<MarkAllReadResponse> {
  return api.fetch<MarkAllReadResponse>(
    "/me/photographer/messages/read-all",
    { method: "PATCH" },
  );
}

export interface MessageRemovedResponse {
  removed: boolean;
}

export async function removeMyPhotographerMessage(
  id: string,
): Promise<MessageRemovedResponse> {
  return api.delete<MessageRemovedResponse>(
    `/me/photographer/messages/${encodeURIComponent(id)}`,
  );
}
