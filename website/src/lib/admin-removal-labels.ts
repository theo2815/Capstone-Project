import { formatSalesDate } from "@/lib/format";
import type { PhotoRemovalRequestDto, RemovalActorType, RemovalMatchSource, RemovalRequestStatus } from "@/types/photo-removal";

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

// FT-32 (C-74): the one line under a request's header that names its step and who it waits for.
export function removalStepLine(request: PhotoRemovalRequestDto, now = Date.now()): string | null {
  switch (request.status) {
    case "DRAFT": return "Step 1 of 4. Choose the photos this request is about, then verify them.";
    case "OPEN": {
      // An owner still to answer: a reachable owner, not overdue, with a hidden photo they have not answered for.
      const unanswered = request.reviews.filter((review) => !review.overdue && !review.unavailableReason &&
        request.items.some((item) => item.reviewId === review.id && item.state === "VERIFIED" && !item.ownerResponse));
      // An owner with no deadline yet has not been told; the sheet has no line for that, so none shows.
      if (unanswered.some((review) => !review.deadline)) return null;
      const waiting = unanswered.filter((review) => Date.parse(review.deadline!) > now);
      if (waiting.length === 0) return "Step 3 of 4. Ready for your decision on each photo.";
      const until = waiting.map((review) => review.deadline!).reduce((a, b) => (Date.parse(b) > Date.parse(a) ? b : a));
      return `Step 2 of 4. Waiting for ${waiting.length} photographer${waiting.length === 1 ? "" : "s"} to answer, until ${formatSalesDate(until)}. You can decide earlier with an urgent reason.`;
    }
    // A cancel also passes through Closing; only an authorized removal has files being deleted.
    case "RESOLVING": return request.items.some((item) => item.state === "REMOVAL_AUTHORIZED")
      ? "Step 4 of 4. Removal authorized. The files are being deleted." : null;
    case "RESOLVED": return "Finished. Every photo was decided.";
    // W32.1's "Nothing was hidden or removed." is held: a request cancelled while open had its photos hidden.
    default: return null;
  }
}
