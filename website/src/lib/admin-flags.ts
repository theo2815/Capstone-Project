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
  | "other"
  // Written by the backend's SystemFlagService (SystemFlagReason, lowercase).
  | "profile_image_flagged"
  | "profile_image_unchecked"
  | "payout_name_changed"
  | "repeat_account_match"
  | "race_check_miss"
  | "race_edit_held"
  | "possible_duplicate"
  | "same_bib_duplicate"
  | "no_proof_hidden"
  | "no_proof_unindexed"
  | "report_threshold_hidden"
  | "user_report"
  | "runner_removal_held";

/** What a flag points at (flags.target_kind). Only photo flags can be hidden. */
export type FlagTargetKind = "photo" | "user" | "event";

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
  targetKind: FlagTargetKind;
  targetId: string;
}

export const FLAG_REASON_LABEL: Record<FlagReason, string> = {
  wrong_runner: "Wrong runner / face mismatch",
  low_quality: "Quality below threshold",
  inappropriate: "Inappropriate or NSFW",
  watermark_bypass: "Watermark removed or bypassed",
  duplicate: "Duplicate of another listing",
  other: "Other (see note)",
  profile_image_flagged: "Profile image failed the automatic check",
  profile_image_unchecked: "Profile image could not be checked",
  payout_name_changed: "Payout name changed",
  repeat_account_match: "Details match another account",
  race_check_miss: "Race failed an automatic check",
  race_edit_held: "Race edit waiting for review",
  possible_duplicate: "Possible duplicate of another race",
  same_bib_duplicate: "Possible duplicate: same bibs as another race",
  no_proof_hidden: "Race hidden: no proof it took place",
  no_proof_unindexed: "Race has no indexed photos yet",
  report_threshold_hidden: "Race hidden after several reports",
  user_report: "Reported by a user",
  runner_removal_held: "Runner deletion held",
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
// Hide hides a photo, so it exists only for photo flags (AdminFlagService.hide).
function isOpenOrEscalated(flag: Flag): boolean {
  return flag.status === "open" || flag.status === "escalated";
}
export function canHide(flag: Flag): boolean {
  return flag.targetKind === "photo" && isOpenOrEscalated(flag);
}
export function canEscalate(flag: Flag): boolean {
  return flag.status === "open";
}
export function canDismiss(flag: Flag): boolean {
  return isOpenOrEscalated(flag) || flag.status === "hidden";
}
