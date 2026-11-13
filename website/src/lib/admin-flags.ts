// Content moderation queue. Rows come from GET /api/v1/admin/flags
// (AdminFlagDto); `photoSnapshot` is hydrated server-side so the UI never
// cross-references a photo store. Runner-side flag filing is not built yet —
// nothing constructs a flag row today.

import { ApiError } from "@/lib/api";

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
  | "runner_removal_held"
  // PE-2: a photographer's batch of uploads that may not belong to the event.
  | "batch_relevance_low";

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
  /** PE-2.2: when an admin confirmed a payout hold on this relevance flag. */
  payoutHoldAt: string | null;
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
  batch_relevance_low: "Uploads may not belong to the event",
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

// PE-2.7: relevance flags (reason batch_relevance_low on a photographer). They are
// held, hidden by batch and resolved here; the single-photo Hide never applies.
export function isRelevanceFlag(flag: Flag): boolean {
  return flag.targetKind === "user" && flag.reason === "batch_relevance_low";
}
export function canConfirmHold(flag: Flag): boolean {
  return isRelevanceFlag(flag) && isOpenOrEscalated(flag) && !flag.payoutHoldAt;
}
export function canHideBatch(flag: Flag): boolean {
  return isRelevanceFlag(flag) && isOpenOrEscalated(flag);
}
export function canResolve(flag: Flag): boolean {
  return canHideBatch(flag);
}

export type FlagConfirmKind = "hold" | "resolve" | "dismiss";

/** The server's own refusal text (ApiError.errors[0].message), or null for any other failure. */
export function serverMessage(error: unknown): string | null {
  return error instanceof ApiError ? (error.errors[0]?.message ?? null) : null;
}

/** What the inline confirm says before the click (R2-03 for resolve and dismiss). */
export function relevanceConfirmText(kind: FlagConfirmKind): string {
  switch (kind) {
    case "hold":
      return "Holds this photographer's payouts until you resolve or dismiss this flag. They are told in their inbox.";
    case "resolve":
      return "Ends the payout hold. Photos this flag hid stay hidden; you can restore them one by one.";
    case "dismiss":
      return "False alarm. Ends the payout hold and puts back on sale every photo this flag hid.";
  }
}

export const COPY_PAIR_NOTICE =
  "The earlier upload is presumed to be the original. Check both photographers' upload histories before you decide.";

// Mirrors dto/admin/RelevanceEvidenceDto.kt. Admin-only: copy samples name the other photographer.
export interface RelevanceEvidence {
  flagId: string;
  payoutHoldAt: string | null;
  batches: RelevanceBatch[];
}
export interface RelevanceBatch {
  eventId: string;
  eventName: string;
  eventDate: string;
  sharedRace: boolean;
  level: string | null;
  copyLow: boolean | null;
  dateLow: boolean | null;
  photosLive: number;
  lookups: number;
  tightHits: number;
  weakHits: number;
  exactMatches: number;
  xmpMatches: number;
  datedPhotos: number | null;
  datedOutside: number | null;
  metadataPending: number | null;
  bibbedPhotos: number | null;
  bibOverlapPhotos: number | null;
  scoredAt: string | null;
  copySamples: CopySample[];
  dateSamples: DateSample[];
  hiddenPhotos: number;
}
export interface CopySample {
  photoId: string;
  thumbnailUrl: string | null;
  matchPhotoId: string | null;
  matchThumbnailUrl: string | null;
  matchPhotographerHandle: string | null;
  distance: number | null;
  pair: string | null;
  exact: boolean;
  exactPhotoId: string | null;
  exactThumbnailUrl: string | null;
  exactPhotographerHandle: string | null;
  exactEventId: string | null;
}
export interface DateSample {
  photoId: string;
  thumbnailUrl: string | null;
  takenOn: string;
}
// Mirrors dto/admin/HiddenPhotoDto.kt. A restored row stays in the list with restoredAt set.
export interface HiddenPhoto {
  photoId: string;
  eventId: string;
  thumbnailUrl: string | null;
  hiddenAt: string;
  restoredAt: string | null;
}
