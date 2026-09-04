// Content moderation queue. Rows come from GET /api/v1/admin/flags
// (AdminFlagDto); `photoSnapshot` is hydrated server-side so the UI never
// cross-references a photo store. Runner-side flag filing is not built yet —
// nothing constructs a flag row today.

export type FlagStatus = "open" | "resolved" | "hidden" | "dismissed" | "escalated";
export type FlagReason =
  | "wrong_runner"
  | "low_quality"
  | "inappropriate"
  | "watermark_bypass"
  | "duplicate"
  | "other";

export interface FlagPhotoSnapshot {
  alt: string;
  kmMark: number | null;
  bib: string | null;
  thumbnailUrl: string | null;
}

export interface Flag {
  id: string;
  photoId: string | null;
  eventId: string | null;
  eventName: string | null;
  photographerHandle: string;
  reportedBy: string; // handle or "system"
  /** Free text on the wire (flags.reason VARCHAR(40)); usually a FlagReason. */
  reason: string;
  note: string;
  status: FlagStatus;
  reportedAt: string;
  reviewedAt: string | null;
  reviewedBy: string | null;
  reviewerNote: string | null;
  photoSnapshot: FlagPhotoSnapshot;
}

export const FLAG_REASON_LABEL: Record<FlagReason, string> = {
  wrong_runner: "Wrong runner / face mismatch",
  low_quality: "Quality below threshold",
  inappropriate: "Inappropriate or NSFW",
  watermark_bypass: "Watermark removed or bypassed",
  duplicate: "Duplicate of another listing",
  other: "Other (see note)",
};

export function flagReasonLabel(reason: string): string {
  return (FLAG_REASON_LABEL as Record<string, string>)[reason] ?? reason;
}

export function flagEventName(flag: Flag): string {
  return flag.eventName ?? "Unknown event";
}

// Transition table mirrors AdminFlagService: hide from open/escalated,
// escalate from open, dismiss from open/escalated/hidden (dismissing a
// hidden flag restores the photo unless another hidden flag still targets it).
export function canHide(flag: Flag): boolean {
  return flag.status === "open" || flag.status === "escalated";
}
export function canEscalate(flag: Flag): boolean {
  return flag.status === "open";
}
export function canDismiss(flag: Flag): boolean {
  return canHide(flag) || flag.status === "hidden";
}
