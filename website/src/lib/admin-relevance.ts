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
  // RV-4.1/RV-4.2: null = not confirmed; the photographer's confirmed cases in the window; null = no live payout hold;
  // null while the case is current.
  confirmedAt: string | null;
  photographerStrikes: number;
  payoutHoldAt: string | null;
  // RV-4.5a: a payout hold was placed on this case at some time (an ended one included).
  payoutHoldPlaced: boolean;
  closeReason: string | null;
}

export type RelevanceTab = "hidden" | "appeals" | "cant_tell" | "all";

export const RELEVANCE_TABS: readonly { value: RelevanceTab; label: string; empty: string }[] = [
  { value: "hidden", label: "Hidden", empty: "No batch is hidden, or would have been hidden." },
  { value: "appeals", label: "Appeals", empty: "No open appeals." },
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
  ADMIN_HIDE: "An admin hid this batch from this page.",
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

export type RelevanceCaseAction = "restore" | "keep_hidden" | "looks_fine" | "hide_batch" | "end_hold" | "hold_payouts";

export const RELEVANCE_ACTIONS: Record<RelevanceCaseAction, { label: string; title: string; confirm: string; confirmLabel: string; done: string; failed: string }> = {
  restore: {
    label: "Restore batch…",
    title: "Restore batch",
    confirm: "Puts back on sale every photo this case hid in the batch. The case closes as restored by an admin and does not count. A Copied case's payout hold ends too. Photos the photographer hid stay hidden.",
    confirmLabel: "Confirm restore",
    done: "Batch restored",
    // RV-4.3: a restore that broke off can now be finished from the page (caseActions offers it on the closed case).
    failed: "Could not restore the batch. The case was read again: if photos are still held, press Restore batch again to finish the rest.",
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
  // RV-4.3 (A.8): the case stays as it is; AdminFlagService tells the photographer only once no hold is left.
  end_hold: {
    label: "End hold",
    title: "End hold",
    confirm: "Ends the payout hold this case placed. The case itself stays as it is. The photographer is told their payout hold was lifted only when no other Photo Relevance hold is left.",
    confirmLabel: "Confirm end hold",
    done: "Hold ended",
    failed: "Could not end the hold. The case was read again: if the payout hold still shows, press End hold again.",
  },
  // RV-4.5b: a no-consent Copied case; the server sends the photographer the approved "Payouts on hold" message.
  hold_payouts: {
    label: "Hold payouts",
    title: "Hold payouts",
    confirm: "Places the payout hold an automatic Copied case would have placed. The photographer is told their payouts are on hold. End hold or a won appeal lifts it. Works only while the relevance mode is auto-hide.",
    confirmLabel: "Confirm hold payouts",
    done: "Payouts held",
    failed: "Could not hold the payouts — check the case before retrying.",
  },
};

// RV-4.3: the consequences of a confirmed case (A.7, A.8).
export const CONSEQUENCES_TEXT = {
  title: "Consequences",
  confirmedOn: "Confirmed on",
  cases: "Cases in the last 6 months",
  holdSince: "Payouts on hold since",
  endHoldHint: "One photographer can have several holds, one per Copied case. Payouts resume only when the last one ends.",
  openPhotographer: "Open photographer →",
  repeat: "Repeat cases",
  repeatEmpty: "No photographer has a repeat case.",
} as const;

export const HIDDEN_PHOTOS_EMPTY = "No photo of this batch is held.";
export const PHOTO_RESTORED = "Photo restored";
export const PHOTO_STILL_HIDDEN = "Released here, but another hold or flag still keeps the photo hidden.";
export const photoRestoreLabel = (position: number, hiddenOn: string) => `Restore photo ${position}, hidden ${hiddenOn}`;
export const PHOTO_RESTORE_FAILED = "Could not restore the photo — check the case before retrying.";

const HELD_BY_LABEL: Record<string, string> = { AUTO: "Automatic", ADMIN: "Admin", PHOTOGRAPHER: "Photographer" };
export const heldByLabel = (h: string) => label(HELD_BY_LABEL, h) ?? h;

/** A hold these routes can release: not yet restored, and not the photographer's own hide. */
export const isReleasable = (p: RelevanceHiddenPhoto) => p.restoredAt === null && p.heldBy !== "PHOTOGRAPHER";

/**
 * The actions a case offers. Record-only never acts (P18): it offers only the batch restore, and only while it is
 * current and a releasable hold exists (seeded rows). Can't tell actions need action NONE:
 * a Can't tell an admin hid offers the restores instead. The mode itself is not readable here; the server refuses.
 */
export function caseActions(
  detail: Pick<RelevanceCaseDetail, "case" | "stage" | "payoutHoldAt" | "payoutHoldPlaced" | "closeReason">,
  hasReleasable: boolean,
): RelevanceCaseAction[] {
  // RV-4.3: a live payout hold can be ended whatever the case's stage; the server answers 409 when none is left.
  const endHold: RelevanceCaseAction[] = detail.payoutHoldAt ? ["end_hold"] : [];
  const row = detail.case;
  const acting = detail.stage === "APPLIED" && row.appliedMode !== "RECORD_ONLY";
  const hiding = row.action !== null && row.action !== "NONE";
  // RV-4.5b: a no-consent case whose verdict would have hidden the whole batch; an admin may hide it, and hold a
  // Copied one's payouts (also after hiding it) once. An admin's hide of it reads AUTO_HIDE + ADMIN_HIDE.
  const noConsent = detail.stage === "APPLIED" && row.appliedMode === "RECORD_ONLY" && row.verdictNote === "NO_CONSENT"
    && ["JUNK_DUMP", "WRONG_EVENT", "COPIED"].includes(row.verdict ?? "");
  const adminHid = acting && row.verdictNote === "ADMIN_HIDE";
  const holdPayouts: RelevanceCaseAction[] =
    (noConsent || adminHid) && row.verdict === "COPIED" && !detail.payoutHoldPlaced ? ["hold_payouts"] : [];
  if (acting && hiding) return ["restore", "keep_hidden", ...holdPayouts, ...endHold];
  if (acting && row.verdict === "CANT_TELL" && row.action === "NONE") return ["looks_fine", "hide_batch", ...endHold];
  // A restore that broke off leaves the case closed RESTORED_BY_ADMIN with holds live: the server takes the repeat
  // (and refuses it once a newer case is current).
  const restorable = detail.stage === "APPLIED" || (detail.stage === "CLOSED" && detail.closeReason === "RESTORED_BY_ADMIN");
  const noConsentActions: RelevanceCaseAction[] = noConsent ? ["hide_batch", ...holdPayouts] : [];
  return [...(restorable && hasReleasable ? (["restore"] as const) : []), ...noConsentActions, ...endHold];
}

/** Why a sample counts as junk, or that it was not judged while a cut-off is unset. */
export function sampleLabel(s: RelevanceSample): string {
  const reasons = [s.blurry && "Blurry", s.dark && "Dark", s.notRace && "Not a race"].filter(Boolean);
  if (reasons.length > 0) return reasons.join(" · ");
  // cleanSamples means "not known to be junk": an unset cut-off leaves a rule unanswered.
  return s.blurry === null || s.dark === null || s.notRace === null ? "Not judged" : "Clean";
}

// RV-3.5: the Appeals tab. Types mirror `RelevanceAppealRowDto`. The photographer's reason is their own free text.
export type RelevanceAppealStatus = "OPEN" | "WON" | "LOST";
export interface RelevanceAppealRow {
  appealId: string;
  caseId: string;
  batchId: string;
  eventId: string;
  eventName: string | null;
  photographerName: string | null;
  photographerHandle: string | null;
  verdict: string | null;
  verdictNote: string | null;
  action: string | null;
  caseStage: string;
  status: RelevanceAppealStatus;
  reason: string;
  createdAt: string | null;
  evidenceOpenedAt: string | null;
  evidenceOpenedBy: string | null;
  decidedAt: string | null;
  decidedBy: string | null;
  decisionNote: string | null;
}

/** Open appeals first, then the decided ones. */
export const APPEAL_SECTIONS: readonly { status: RelevanceAppealStatus; label: string; empty: string }[] = [
  { status: "OPEN", label: "Open", empty: RELEVANCE_TABS.find((t) => t.value === "appeals")!.empty },
  { status: "WON", label: "Won", empty: "No appeal has been won." },
  { status: "LOST", label: "Lost", empty: "No appeal has been lost." },
];

export const APPEAL_NOTE_MAX = 1000;
export const APPEAL_TEXT = {
  reason: "Photographer's reason",
  samples: "Samples",
  openSamples: "Open the samples",
  openSamplesHint: "Opening the samples records that you looked at them. Won and Lost unlock after that.",
  openFailed: "Could not record that you opened the samples. Try again.",
  note: "Note (admins only, optional)",
  noteHint: "The photographer never sees this note.",
  won: "Won",
  lost: "Lost",
  done: { WON: "Appeal won", LOST: "Appeal lost" } as Record<"WON" | "LOST", string>,
  decideFailed: "Could not decide the appeal — check the appeal before retrying.",
  decisionNote: "Decision note",
  filed: "Filed",
  decided: "Decided",
  // The decision is final for the photographer: each confirm says what it does (RelevanceAppealService.decide).
  confirm: {
    WON: {
      title: "Appeal won",
      confirmLabel: "Confirm won",
      text: "Marks the appeal won and tells the photographer. A whole-batch hide becomes a junk-only hide: the full check puts the good photos back on sale as it measures them. A junk-only hide is lifted. A Copied case's payout hold ends; payouts resume once the photographer has no other hold. The case does not count against the photographer. This cannot be undone.",
    },
    LOST: {
      title: "Appeal lost",
      confirmLabel: "Confirm lost",
      text: "Marks the appeal lost and tells the photographer. The photos stay off sale. A junk dump, wrong event or copied finding is confirmed at once and counts against the photographer. This cannot be undone.",
    },
  } as Record<"WON" | "LOST", { title: string; confirmLabel: string; text: string }>,
  // A case closed meanwhile (an admin restored the batch): the server records the decision only, sends no note.
  confirmClosed: "The case is already closed, so this only records your decision. The photographer is not told. This cannot be undone.",
} as const;

// RV-L7 part c: the photographer page's control for a withdrawn consent to automatic actions (legal 1.7).
export const CONSENT_WITHDRAWAL_TEXT = {
  title: "Automatic actions",
  notWithdrawn: "Consent to automatic hides and payout holds: not withdrawn.",
  record: "Record withdrawal",
  recordConfirmButton: "Record",
  recordTitle: "Record that this photographer withdrew consent?",
  recordConfirm: 'Use this when the photographer asks by email to withdraw consent. From now on nothing of theirs is taken off sale and no payout is held automatically. Their cases are shown to admins as "would have hidden". Photos already off sale and payouts already on hold stay as they are.',
  recording: "Recording…",
  recorded: "Withdrawal recorded",
  recordFailed: "Couldn't record the withdrawal — please try again.",
  withdrawnOn: "Consent withdrawn on",
  withdrawnAfter: "Automatic hides and payout holds are off for this photographer; admins review each case.",
  clear: "Clear withdrawal",
  clearConfirmButton: "Clear",
  clearTitle: "Clear the withdrawal?",
  clearConfirm: "Use this only when the photographer asks to give consent again. Automatic hides and payout holds can then apply to them again, if they accepted the current Terms.",
  clearing: "Clearing…",
  cleared: "Withdrawal cleared",
  clearFailed: "Couldn't clear the withdrawal — please try again.",
} as const;
