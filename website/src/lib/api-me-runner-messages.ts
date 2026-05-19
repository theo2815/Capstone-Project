import { api } from "@/lib/api";
import type { RunnerMessage } from "@/lib/runner-messages";

// Runner-facing inbox API. Mirrors the BE MeRunnerMessagesController
// endpoints. All requests carry the JWT bearer; BE filters by the
// principal's userId and gates on hasRole('RUNNER') so a photographer /
// unauth caller never reaches this surface (403).

export async function fetchMyRunnerMessages(): Promise<RunnerMessage[]> {
  return api.get<RunnerMessage[]>("/me/runner/messages");
}

export async function markMyRunnerMessageRead(
  id: string,
): Promise<RunnerMessage> {
  return api.fetch<RunnerMessage>(
    `/me/runner/messages/${encodeURIComponent(id)}/read`,
    { method: "PATCH" },
  );
}

export interface MarkAllReadResponse {
  markedRead: number;
}

export async function markAllMyRunnerMessagesRead(): Promise<MarkAllReadResponse> {
  return api.fetch<MarkAllReadResponse>(
    "/me/runner/messages/read-all",
    { method: "PATCH" },
  );
}

export interface MessageRemovedResponse {
  removed: boolean;
}

export async function removeMyRunnerMessage(
  id: string,
): Promise<MessageRemovedResponse> {
  return api.delete<MessageRemovedResponse>(
    `/me/runner/messages/${encodeURIComponent(id)}`,
  );
}
