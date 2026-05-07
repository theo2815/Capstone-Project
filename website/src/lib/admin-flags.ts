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

export const ADMIN_FLAGS: ReadonlyArray<Flag> = [
  // 5 open
  {
    id: "FLG-001",
    photoId: "photo-bay-run-672",
    eventId: "1",
    photographerHandle: "paksitphotos",
    reportedBy: "juan",
    reason: "wrong_runner",
    note: "This bib number is mine but the runner in the photo isn't me. The face overlay confidence is way off.",
    status: "open",
    reportedAt: "2026-05-06T08:20:00.000Z",
    reviewedAt: null,
    reviewedBy: null,
    reviewerNote: null,
    photoSnapshot: {
      alt: "Runner crossing the bridge — disputed bib match",
      kmMark: 8,
      bib: "1247",
    },
  },
  {
    id: "FLG-002",
    photoId: "photo-stride-204",
    eventId: "u3",
    photographerHandle: "cebustride",
    reportedBy: "system",
    reason: "low_quality",
    note: "AI cull score 0.31 — below 0.50 platform threshold. Surfaced for manual review.",
    status: "open",
    reportedAt: "2026-05-06T16:45:00.000Z",
    reviewedAt: null,
    reviewedBy: null,
    reviewerNote: null,
    photoSnapshot: {
      alt: "Out-of-focus mid-stride frame",
      kmMark: 15,
      bib: "0712",
    },
  },
  {
    id: "FLG-003",
    photoId: "photo-bay-run-803",
    eventId: "1",
    photographerHandle: "paksitphotos",
    reportedBy: "thea",
    reason: "inappropriate",
    note: "Frame appears to crop the runner inappropriately. Should be hidden until photographer re-edits.",
    status: "open",
    reportedAt: "2026-05-07T02:14:00.000Z",
    reviewedAt: null,
    reviewedBy: null,
    reviewerNote: null,
    photoSnapshot: {
      alt: "Runner near finish — composition concern",
      kmMark: 41,
      bib: "0863",
    },
  },
  {
    id: "FLG-004",
    photoId: "photo-stride-318",
    eventId: "u2",
    photographerHandle: "cebustride",
    reportedBy: "nico",
    reason: "watermark_bypass",
    note: "Watermark looks edited out — there's a visible smudge where it should be. I think someone reposted from a paid copy.",
    status: "open",
    reportedAt: "2026-05-07T04:50:00.000Z",
    reviewedAt: null,
    reviewedBy: null,
    reviewerNote: null,
    photoSnapshot: {
      alt: "Runner at sunrise — watermark integrity concern",
      kmMark: 5,
      bib: "0421",
    },
  },
  {
    id: "FLG-005",
    photoId: "photo-bay-run-944",
    eventId: "1",
    photographerHandle: "paksitphotos",
    reportedBy: "system",
    reason: "duplicate",
    note: "Burst-mode duplicate of photo-bay-run-943 (similarity score 0.97). Recommend hide one.",
    status: "open",
    reportedAt: "2026-05-07T06:00:00.000Z",
    reviewedAt: null,
    reviewedBy: null,
    reviewerNote: null,
    photoSnapshot: {
      alt: "Mid-stride duplicate frame",
      kmMark: 18,
      bib: "1102",
    },
  },

  // 1 hidden (resolved against photographer)
  {
    id: "FLG-006",
    photoId: "photo-bay-run-118",
    eventId: "1",
    photographerHandle: "cebustride",
    reportedBy: "maria",
    reason: "wrong_runner",
    note: "Bib was tagged correctly but a different person is in the frame. Confirmed by runner email.",
    status: "hidden",
    reportedAt: "2026-04-18T10:00:00.000Z",
    reviewedAt: "2026-04-19T11:30:00.000Z",
    reviewedBy: "admin",
    reviewerNote: "Confirmed misidentification — photo hidden, photographer notified.",
    photoSnapshot: {
      alt: "Runner crossing the boulevard — disputed identity",
      kmMark: 12,
      bib: "2089",
    },
  },

  // 1 dismissed (false alarm)
  {
    id: "FLG-007",
    photoId: "photo-stride-022",
    eventId: "u3",
    photographerHandle: "cebustride",
    reportedBy: "jp",
    reason: "low_quality",
    note: "Photo looked blurry on my phone.",
    status: "dismissed",
    reportedAt: "2026-04-25T14:11:00.000Z",
    reviewedAt: "2026-04-26T09:02:00.000Z",
    reviewedBy: "admin",
    reviewerNote: "Resolution acceptable on desktop — display issue, not photo quality.",
    photoSnapshot: {
      alt: "Runner approaching aid station",
      kmMark: 7,
      bib: "0517",
    },
  },

  // 1 escalated
  {
    id: "FLG-008",
    photoId: "photo-bay-run-552",
    eventId: "1",
    photographerHandle: "paksitphotos",
    reportedBy: "thea",
    reason: "other",
    note: "Possible copyright concern — photo appears to include a sponsor logo arrangement that may need clearance.",
    status: "escalated",
    reportedAt: "2026-04-30T16:33:00.000Z",
    reviewedAt: null,
    reviewedBy: null,
    reviewerNote: null,
    photoSnapshot: {
      alt: "Runner with sponsor banners visible in background",
      kmMark: 30,
      bib: "0863",
    },
  },
];

export function getOpenFlags(): Flag[] {
  return ADMIN_FLAGS.filter((f) => f.status === "open");
}

export function getFlagById(id: string): Flag | undefined {
  return ADMIN_FLAGS.find((f) => f.id === id);
}
