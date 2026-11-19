import type { RemovalActorType, RemovalMatchSource, RemovalRequestStatus } from "@/types/photo-removal";

// Admin display labels for photo-removal codes; codes stay codes on the client.
export const REMOVAL_STATUS_LABEL: Record<RemovalRequestStatus, string> = {
  DRAFT: "Draft", OPEN: "Open", RESOLVING: "Closing", RESOLVED: "Resolved", CANCELLED: "Cancelled",
};
export const REMOVAL_SOURCE_LABEL: Record<RemovalMatchSource, string> = { SELFIE: "Selfie match", BIB: "Bib search", MANUAL: "Picked by hand" };
const ACTOR_LABEL: Record<RemovalActorType, string> = {
  ADMIN: "Admin", PHOTOGRAPHER: "Photographer", SYSTEM: "System", SUPPORT: "Support", RUNNER: "Runner",
};
// History actions, from RemovalHistoryActions in the backend.
const ACTION_LABEL: Record<string, string> = {
  CREATED: "Case created", EVIDENCE_REPLACED: "Selfie replaced", VERIFIED: "Photos verified", NOTICE_STORED: "Owner notice stored",
  OWNER_UNAVAILABLE: "Owner unavailable", OWNER_RESPONDED: "Owner responded", REVIEW_OVERDUE: "Owner review overdue",
  ADMIN_DECIDED: "Admin decision", ERASURE_PROGRESS: "Erasure progress", EVIDENCE_PURGED: "Evidence deleted",
  SUPPORT_UPDATED: "Support update", CANCEL_REQUESTED: "Cancel requested", RESOLVED: "Case resolved", CANCELLED: "Case cancelled",
  SUGGESTED: "Suggested", SUGGESTION_DISMISSED: "Suggestion dismissed",
};
// A history state change is a case status or a photo state.
const STATE_LABEL: Record<string, string> = {
  ...REMOVAL_STATUS_LABEL, CANDIDATE: "Candidate", VERIFIED: "Verified", EXCLUDED: "Excluded", REMOVAL_AUTHORIZED: "Removal authorized",
  REMOVED: "Removed", RETAINED: "Retained", RETAINED_RESTRICTED: "Retained restricted",
};
// A code added later reads as words, never as a raw code.
const label = (map: Record<string, string>, code: string) =>
  Object.hasOwn(map, code) ? map[code] : code.charAt(0) + code.slice(1).toLowerCase().replaceAll("_", " ");
export const removalActionLabel = (code: string) => label(ACTION_LABEL, code);
export const removalActorLabel = (code: string) => label(ACTOR_LABEL, code);
export const removalStateLabel = (code: string) => label(STATE_LABEL, code);
