// Phase 2b — content moderation queue. Flags are reported either by
// runners (via a future button on the runner photo grid; not built yet)
// or surfaced by admin during review. Each flag carries a `photoSnapshot`
// inline so the moderation UI doesn't need to cross-reference a photo
// store. When backend lands in Phase F, snapshots stay — they're set
// when the flag is filed.

export type FlagStatus = "open" | "hidden" | "dismissed" | "escalated";
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
  thumbnailUrl?: string;
}

export interface Flag {
  id: string;
  photoId: string;
  eventId: string;
  photographerHandle: string;
  reportedBy: string; // handle or "system"
  reason: FlagReason;
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

export const ADMIN_FLAGS: ReadonlyArray<Flag> = [];

export function getOpenFlags(): Flag[] {
  return ADMIN_FLAGS.filter((f) => f.status === "open");
}

export function getFlagById(id: string): Flag | undefined {
  return ADMIN_FLAGS.find((f) => f.id === id);
}
