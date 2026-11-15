// RV-1.11a: the admin Photo Relevance page. Types mirror backend `dto/admin/RelevanceCaseDtos.kt`;
// every label the page shows for a verdict, note, action or alert kind lives here.

export interface RelevanceCaseRow {
  caseId: string;
  eventId: string;
  eventName: string;
  photographerHandle: string | null;
  photographerName: string;
  verdict: string | null;
  verdictNote: string | null;
  action: string | null;
  appliedMode: string | null;
  wouldHave: boolean;
  batchPhotos: number;
  junkPercent: number | null;
  decidedAt: string | null;
}

export interface RelevanceWord {
  word: string;
  photos: number;
}

export interface RelevanceSample {
  photoId: string;
  thumbnailUrl: string | null;
  fgSharpness: number | null;
  fgBrightness: number | null;
  sharpness: number | null;
  brightness: number | null;
  blurry: boolean | null;
  dark: boolean | null;
  notRace: boolean | null;
}

export interface RelevanceAlert {
  alertId: string;
  caseId: string | null;
  eventId: string;
  eventName: string;
  photographerHandle: string | null;
  photographerName: string;
  kind: string;
  wouldHave: boolean;
  photos: number | null;
  createdAt: string | null;
  seenAt: string | null;
}

export interface RelevanceCaseDetail {
  case: RelevanceCaseRow;
  stage: string;
  sampleSize: number | null;
  measured: number | null;
  blurry: number | null;
  dark: number | null;
  notRace: number | null;
  junk: number | null;
  band: string | null;
  modelVerdict: string | null;
  adminReason: string | null;
  evidenceUsed: string[];
  summarySent: string | null;
  model: string | null;
  promptVersion: string | null;
  wordsPhotos: number | null;
  topWords: RelevanceWord[];
  junkSamples: RelevanceSample[];
  cleanSamples: RelevanceSample[];
  alerts: RelevanceAlert[];
}

export type RelevanceTab = "hidden" | "appeals" | "cant_tell" | "all";

export const RELEVANCE_TABS: readonly { value: RelevanceTab; label: string; empty: string }[] = [
  { value: "hidden", label: "Hidden", empty: "No batch is hidden, or would have been hidden." },
  { value: "appeals", label: "Appeals", empty: "No appeals yet. Appeals arrive when photographers can answer a decision." },
  { value: "cant_tell", label: "Can't tell", empty: "No batch is waiting on a person to judge it." },
  { value: "all", label: "All", empty: "No batch has been checked yet." },
];

export const parseRelevanceTab = (raw: string): RelevanceTab =>
  RELEVANCE_TABS.some((t) => t.value === raw) ? (raw as RelevanceTab) : "hidden";

const VERDICT_LABEL: Record<string, string> = {
  FITS: "Fits",
  MIXED: "Mixed",
  JUNK_DUMP: "Junk dump",
  WRONG_EVENT: "Wrong event",
  COPIED: "Copied",
  CANT_TELL: "Can't tell",
};

// Why code replaced the model's answer with "Can't tell" (plan 5.9), or applied a case as record-only (5.14).
const NOTE_LABEL: Record<string, string> = {
  CALL_FAILED: "The call to the model failed after every try.",
  CUT_OFF: "The model's answer was cut off or refused.",
  BAD_JSON: "The model's answer was not the expected JSON.",
  UNKNOWN_VALUE: "The model answered with a verdict that does not exist.",
  CONTRADICTS: "The model's verdict contradicts the measured numbers or the copy rule.",
  THIN_EVIDENCE: "The model said wrong event without enough evidence.",
  SMALL_BATCH: "The batch is too small for anything but Fits or Can't tell.",
  SUSPICIOUS_WORDS: "Words read off the photos seemed to tell the model what to answer.",
  NO_CONSENT: "The photographer has not accepted the Terms that allow automatic hiding.",
};

const ACTION_LABEL: Record<string, string> = {
  NONE: "Nothing",
  HIDE_JUNK: "Hide the junk photos",
  HIDE_ALL: "Hide the whole batch",
};

const ALERT_LABEL: Record<string, string> = {
  BIG_HIDE: "Big hide",
  COPIED: "Copied photos",
  APPEAL: "New appeal",
  THIRD_CASE: "Third confirmed case",
};

const BAND_LABEL: Record<string, string> = {
  UNDER: "Under 10% junk",
  MIXED: "10% to 70% junk",
  OVER: "Over 70% junk",
  SMALL: "Batch too small",
  UNKNOWN: "Unknown (a cut-off is unset or too few photos were measured)",
};

// The evidence the model said it used (plan 5.9, `evidence_used`).
const EVIDENCE_LABEL: Record<string, string> = {
  junk_share: "Junk share",
  copy_rule: "Copy rule",
  date_rule: "Date rule",
  batch_words: "Words on the batch",
  event_words: "Words on other photographers' batches",
  words_suspicious: "Words that looked like instructions",
};

// An unknown value from a newer server shows as its raw text, never a crash.
const label = (map: Record<string, string>, value: string | null): string | null =>
  value === null ? null : (map[value] ?? value);

export const verdictLabel = (v: string | null) => label(VERDICT_LABEL, v) ?? "Not decided";
export const verdictNoteLabel = (n: string | null) => label(NOTE_LABEL, n);
export const actionLabel = (a: string | null) => label(ACTION_LABEL, a);
// RV-1.11b: the inbox's alert list. A record-only alert reads as on the page.
export const WOULD_HAVE_LABEL = "Would have hidden";
export const ALERTS_EMPTY = "No unseen relevance alerts.";
export const ALERT_SEEN_FAILED = "Could not mark the alert seen. It stays in the list; try again.";

export const alertKindLabel = (k: string) => label(ALERT_LABEL, k) ?? k;
export const evidenceLabel = (e: string) => label(EVIDENCE_LABEL, e) ?? e;
export const bandLabel = (b: string | null) => label(BAND_LABEL, b) ?? "Not worked out";

/** Applied in record-only with a hide recorded: nothing was hidden. The server works it out. */
export const isWouldHave = (row: Pick<RelevanceCaseRow, "wouldHave">) => row.wouldHave;

// RV-2.6: the actions on a case. Types mirror `RelevanceHiddenPhotoDto`, `RelevanceRestoreDto`, `RelevancePhotoRestoreDto`.
export interface RelevanceHiddenPhoto {
  photoId: string;
  heldBy: string;
  hiddenAt: string | null;
  restoredAt: string | null;
  restoredBy: string | null;
  status: string;
  thumbnailUrl: string | null;
}
export interface RelevanceRestoreResult {
  caseId: string;
  restoredPhotos: number;
}
export interface RelevancePhotoRestoreResult {
  photoId: string;
  status: string;
}

export type RelevanceCaseAction = "restore" | "keep_hidden" | "looks_fine" | "hide_batch";

export const RELEVANCE_ACTIONS: Record<RelevanceCaseAction, { label: string; title: string; confirm: string; confirmLabel: string; done: string; failed: string }> = {
  restore: {
    label: "Restore batch…",
    title: "Restore batch",
    confirm: "Puts back on sale every photo this case hid in the batch. The case closes as restored by an admin and does not count. Photos the photographer hid stay hidden.",
    confirmLabel: "Confirm restore",
    done: "Batch restored",
    failed: "Could not restore the batch — check the case before retrying.",
  },
  keep_hidden: {
    label: "Keep hidden…",
    title: "Keep hidden",
    confirm: "Keeps the photos hidden and marks this batch's alerts seen. Nothing else changes.",
    confirmLabel: "Confirm keep hidden",
    done: "Kept hidden",
    failed: "Could not keep the batch hidden — check the case before retrying.",
  },
  looks_fine: {
    label: "Looks fine…",
    title: "Looks fine",
    confirm: "Closes this Can't tell case as looking fine. Nothing is hidden or held.",
    confirmLabel: "Confirm looks fine",
    done: "Marked looks fine",
    failed: "Could not mark the case looks fine — check the case before retrying.",
  },
  hide_batch: {
    label: "Hide batch…",
    title: "Hide batch",
    confirm: "Takes every photo of this batch off sale, a chunk at a time over the next minutes. Works only while the relevance mode is auto-hide. You can restore the batch or photos one by one.",
    confirmLabel: "Confirm hide batch",
    done: "Batch hide started",
    failed: "Could not hide the batch — check the case before retrying.",
  },
};

export const HIDDEN_PHOTOS_EMPTY = "No photo of this batch is held.";
export const PHOTO_RESTORED = "Photo restored";
export const PHOTO_STILL_HIDDEN = "Released here, but another hold or flag still keeps the photo hidden.";
export const PHOTO_RESTORE_FAILED = "Could not restore the photo — check the case before retrying.";

const HELD_BY_LABEL: Record<string, string> = { AUTO: "Automatic", ADMIN: "Admin", PHOTOGRAPHER: "Photographer" };
export const heldByLabel = (h: string) => label(HELD_BY_LABEL, h) ?? h;

/** A hold these routes can release: not yet restored, and not the photographer's own hide. */
export const isReleasable = (p: RelevanceHiddenPhoto) => p.restoredAt === null && p.heldBy !== "PHOTOGRAPHER";

/**
 * The actions a case offers. Record-only never acts (P18): it offers only the batch restore, and only while a
 * releasable hold exists (seeded rows, or what a cut-off restore left). Can't tell actions need action NONE:
 * a Can't tell an admin hid offers the restores instead. The mode itself is not readable here; the server refuses.
 */
export function caseActions(detail: Pick<RelevanceCaseDetail, "case" | "stage">, hasReleasable: boolean): RelevanceCaseAction[] {
  const row = detail.case;
  const acting = detail.stage === "APPLIED" && row.appliedMode !== "RECORD_ONLY";
  const hiding = row.action !== null && row.action !== "NONE";
  if (acting && hiding) return ["restore", "keep_hidden"];
  if (acting && row.verdict === "CANT_TELL" && row.action === "NONE") return ["looks_fine", "hide_batch"];
  return hasReleasable ? ["restore"] : [];
}

/** Why a sample counts as junk, or that it was not judged while a cut-off is unset. */
export function sampleLabel(s: RelevanceSample): string {
  const reasons = [s.blurry && "Blurry", s.dark && "Dark", s.notRace && "Not a race"].filter(Boolean);
  if (reasons.length > 0) return reasons.join(" · ");
  // cleanSamples means "not known to be junk": an unset cut-off leaves a rule unanswered.
  return s.blurry === null || s.dark === null || s.notRace === null ? "Not judged" : "Clean";
}
