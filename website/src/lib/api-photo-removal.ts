import { api } from "./api";
import type {
  RemovalPage, RemovalPhotoDto, PhotoRemovalRequestDto, PhotographerRemovalRequestDto, RemovalHistoryDto,
  RemovalMatchResult, RemovalRequestFilters, RemovalPhotoFilters, RemovalPageFilters,
  CreateRemovalRequestCommand, MatchRemovalCommand, VerifyRemovalCommand, CancelRemovalCommand,
  DecideRemovalCommand, RespondToRemovalCommand, RetryRemovalCleanupCommand, RemovalSupportUpdateCommand,
} from "@/types/photo-removal";

const adminPath = "/admin/photo-removal-requests";
const photographerPath = "/me/photographer/photo-removal-requests";
const casePath = (id: string) => `${adminPath}/${encodeURIComponent(id)}`;
function filtered(path: string, filters: RemovalRequestFilters | RemovalPhotoFilters | RemovalPageFilters) {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(filters)) if (value !== undefined) params.set(key, String(value));
  return params.size ? `${path}?${params}` : path;
}

export function listAdminRemovalRequests(filters: RemovalRequestFilters = {}, signal?: AbortSignal): Promise<RemovalPage<PhotoRemovalRequestDto>> {
  return api.get(filtered(adminPath, filters), { signal });
}
export function getAdminRemovalRequest(id: string, signal?: AbortSignal): Promise<PhotoRemovalRequestDto> {
  return api.get(casePath(id), { signal });
}
export function createRemovalRequest(command: CreateRemovalRequestCommand): Promise<PhotoRemovalRequestDto> {
  return api.post(adminPath, command);
}
export function replaceRemovalEvidence(id: string, file: File, commandId: string, expectedVersion: number): Promise<PhotoRemovalRequestDto> {
  const body = new FormData();
  body.append("selfie", file); body.append("commandId", commandId); body.append("expectedVersion", String(expectedVersion));
  return api.fetch(`${casePath(id)}/evidence`, { method: "PUT", body });
}
export function getRemovalEvidence(id: string, signal?: AbortSignal): Promise<Blob> {
  return api.getBlob(`${casePath(id)}/evidence`, { signal });
}
export function listRemovalPhotos(id: string, filters: RemovalPhotoFilters = {}, signal?: AbortSignal): Promise<RemovalPage<RemovalPhotoDto>> {
  return api.get(filtered(`${casePath(id)}/photos`, filters), { signal });
}
export function listRemovalHistory(id: string, filters: RemovalPageFilters = {}, signal?: AbortSignal): Promise<RemovalPage<RemovalHistoryDto>> {
  return api.get(filtered(`${casePath(id)}/history`, filters), { signal });
}
export function findRemovalMatches(id: string, command: MatchRemovalCommand): Promise<RemovalMatchResult> {
  return api.post(`${casePath(id)}/matches`, command);
}
export function verifyRemovalRequest(id: string, command: VerifyRemovalCommand): Promise<PhotoRemovalRequestDto> {
  return api.post(`${casePath(id)}/verify`, command);
}
export function cancelRemovalRequest(id: string, command: CancelRemovalCommand): Promise<PhotoRemovalRequestDto> {
  return api.post(`${casePath(id)}/cancel`, command);
}
export function decideRemovalRequest(id: string, command: DecideRemovalCommand): Promise<PhotoRemovalRequestDto> {
  return api.post(`${casePath(id)}/decisions`, command);
}
export function retryRemovalCleanup(id: string, command: RetryRemovalCleanupCommand): Promise<PhotoRemovalRequestDto> {
  return api.post(`${casePath(id)}/retry-cleanup`, command);
}
export function addRemovalSupportUpdate(id: string, command: RemovalSupportUpdateCommand): Promise<RemovalHistoryDto> {
  return api.post(`${casePath(id)}/support-updates`, command);
}
export function listPhotographerRemovalRequests(filters: RemovalRequestFilters = {}, signal?: AbortSignal): Promise<RemovalPage<PhotographerRemovalRequestDto>> {
  return api.get(filtered(photographerPath, filters), { signal });
}
export function getPhotographerRemovalRequest(id: string, signal?: AbortSignal): Promise<PhotographerRemovalRequestDto> {
  return api.get(`${photographerPath}/${encodeURIComponent(id)}`, { signal });
}
export function respondToRemovalRequest(id: string, command: RespondToRemovalCommand): Promise<PhotographerRemovalRequestDto> {
  return api.post(`${photographerPath}/${encodeURIComponent(id)}/responses`, command);
}
