import { api } from "@/lib/api";
import type { PaginatedResponse } from "@/types/api";

// Seller details (SC-1 backend, plan § 3.4): required before new uploads.
// E1 GET / E2 POST /api/v1/me/photographer/seller-details. No links or storage
// keys come back; only admins open the files.

export type SellerDetailsStatus = "missing" | "submitted" | "approved" | "rejected";

// The detailed state (AA-I1). New code reads this, never `status`, which keeps
// today's values for old clients.
export type ReviewState = "missing" | "manual_review" | "auto_approved" | "approved" | "rejected";
const REVIEW_STATES: readonly string[] = ["missing", "manual_review", "auto_approved", "approved", "rejected"];

/** An unknown value (a newer backend) is treated like manual review. */
export function reviewStateOf(details: Pick<SellerDetails, "reviewState">): ReviewState {
  return REVIEW_STATES.includes(details.reviewState) ? details.reviewState : "manual_review";
}

const manilaToday = new Intl.DateTimeFormat("en-CA", { timeZone: "Asia/Manila" }); // YYYY-MM-DD

/**
 * The announced identity-gate date (AA-I22), while it is today or later in Manila and identity has not
 * passed; else null. After the date the identity gate state speaks instead (a slipped date would be wrong).
 */
export function identityNoticeDate(details: SellerDetails | undefined, now: Date = new Date()): string | null {
  const startsOn = details?.identityGateStartsOn;
  if (!details || !startsOn) return null;
  const state = reviewStateOf(details);
  if (state === "auto_approved" || state === "approved") return null;
  return startsOn >= manilaToday.format(now) ? startsOn : null;
}

/** "Monday 7 December 2026", as the notice email writes it. */
export function formatGateDate(isoDate: string): string {
  return new Intl.DateTimeFormat("en-GB", { weekday: "long", day: "numeric", month: "long", year: "numeric", timeZone: "UTC" })
    .format(new Date(isoDate))
    .replace(",", "");
}

export const SELLER_ID_TYPES = [
  { value: "philsys", label: "PhilSys National ID (card or ePhilID)" },
  { value: "drivers_license", label: "Driver's licence" },
  { value: "passport", label: "Passport" },
  { value: "umid", label: "UMID" },
  { value: "prc", label: "PRC ID" },
  { value: "postal", label: "Postal ID" },
  { value: "voters", label: "Voter's ID" },
] as const;

export type SellerIdType = (typeof SELLER_ID_TYPES)[number]["value"];

export interface SellerDetails {
  status: SellerDetailsStatus;
  reviewState: ReviewState;
  /** The server's upload-gate answer (AA-I6); clients never re-derive it. */
  uploadsAllowed?: boolean;
  identityGateStartsOn?: string | null;
  legalName?: string | null;
  idType?: SellerIdType | null;
  address?: string | null;
  phone?: string | null;
  hasIdBack?: boolean | null;
  hasBirForm?: boolean | null;
  rejectionReason?: string | null;
  submittedAt?: string | null;
  reviewedAt?: string | null;
}

export interface SellerDetailsInput {
  legalName: string;
  idType: SellerIdType;
  address: string;
  phone: string;
  idFront: File;
  idBack?: File | null;
  selfie: File;
  birForm?: File | null;
  idNumber: string;
  /** The finished live check. Left out = "Have an admin check instead". */
  livenessSessionId?: string;
}

// C1 (AA-I2). The STS keys work only for StartFaceLivenessSession, and the
// session expires about 3 minutes after this call (AA-0.3b): start it on press.
export interface LivenessSession {
  sessionId: string;
  region: string;
  attemptsLeftToday: number;
  credentials: { accessKeyId: string; secretAccessKey: string; sessionToken: string; expiration: string } | null;
}

/** Same rule as the server's first check (IdentityCheckService); the server still decides. */
export const normaliseIdNumber = (raw: string) => raw.replace(/[\s-]/g, "");
export const isValidIdNumber = (raw: string) => /^[A-Za-z0-9]{4,30}$/.test(normaliseIdNumber(raw));

// Mirrors SellerDetailsService on the server; the server still decides.
export const SELLER_FILE_MAX_BYTES = 5 * 1024 * 1024;
export const SELLER_FILE_TYPES = ["image/jpeg", "image/png", "image/webp"];

export function isValidPhone(raw: string): boolean {
  const typed = raw.trim();
  const compact = typed.replace(/[\s.()-]/g, "");
  const digits = compact.replace(/\D/g, "").length;
  return typed.length <= 30 && /^\+?\d+$/.test(compact)
    && (compact.startsWith("+63") || compact.startsWith("0")) && digits >= 10 && digits <= 13;
}

export function getSellerDetails(signal?: AbortSignal): Promise<SellerDetails> {
  return api.get<SellerDetails>("/me/photographer/seller-details", { signal });
}

export function submitSellerDetails(input: SellerDetailsInput): Promise<SellerDetails> {
  const fd = new FormData();
  fd.append("legalName", input.legalName);
  fd.append("idType", input.idType);
  fd.append("address", input.address);
  fd.append("phone", input.phone);
  fd.append("idFront", input.idFront);
  if (input.idBack) fd.append("idBack", input.idBack);
  fd.append("selfie", input.selfie);
  if (input.birForm) fd.append("birForm", input.birForm);
  fd.append("idNumber", input.idNumber);
  if (input.livenessSessionId) fd.append("livenessSessionId", input.livenessSessionId);
  return api.post<SellerDetails>("/me/photographer/seller-details", fd);
}

export function startLivenessSession(): Promise<LivenessSession> {
  return api.post<LivenessSession>("/me/photographer/seller-details/liveness-session", { biometricConsent: true });
}

/** C2b: a new live check for a manual-review record; the stored files are reused. */
export function submitLiveCheck(input: { livenessSessionId: string; idNumber: string }): Promise<SellerDetails> {
  return api.post<SellerDetails>("/me/photographer/seller-details/live-check", input);
}

// ─── Admin review (SC-1b backend: E3 to E6; AA-I8 adds the automatic results) ───

/** The admin list filter: the review queue, the spot check, or everything (null). */
export type AdminSellerFilter = "manual_review" | "auto_approved" | null;

export interface AdminSellerDetailsRow {
  userId: string;
  email: string;
  legalName: string;
  idType: SellerIdType;
  status: Exclude<SellerDetailsStatus, "missing">;
  reviewState: Exclude<ReviewState, "missing">;
  submittedAt: string;
  reviewedAt: string | null;
  /** UPPERCASE SellerReasonCode names; [] with autoCheckedAt null = sent before automatic checks. */
  reasonCodes: string[];
  /** 0-100, null when not run. */
  livenessConfidence: number | null;
  faceSimilarity: number | null;
  autoCheckedAt: string | null;
}

export interface RepeatMatch {
  userId: string;
  email: string;
  /** ID_NUMBER | PAYOUT_NUMBER | PHONE */
  matchedOn: string[];
  suspended: boolean;
}

export interface AdminSellerDetails extends AdminSellerDetailsRow {
  address: string;
  phone: string;
  hasIdBack: boolean;
  hasBirForm: boolean;
  rejectionReason: string | null;
  reviewedBy: string | null;
  revision: number;
  biometricConsentAt: string | null;
  hasIdNumber: boolean;
  repeatMatches: RepeatMatch[];
  // Links that expire after five minutes; only on E4, and each fetch logs a view.
  files: { idFront: string; idBack: string | null; selfie: string; birForm: string | null; livenessReference: string | null } | null;
  linksExpireAt: string | null;
}

export function listSellerDetails(
  status: AdminSellerFilter,
  limit: number,
  signal?: AbortSignal,
): Promise<PaginatedResponse<AdminSellerDetailsRow>> {
  const query = new URLSearchParams({ ...(status ? { status } : {}), offset: "0", limit: String(limit) });
  return api.get(`/admin/seller-details?${query}`, { signal });
}

export function getSellerDetailsRecord(userId: string, signal?: AbortSignal): Promise<AdminSellerDetails> {
  return api.get(`/admin/seller-details/${encodeURIComponent(userId)}`, { signal });
}

export function approveSellerDetails(userId: string, revision: number): Promise<AdminSellerDetails> {
  return api.post(`/admin/seller-details/${encodeURIComponent(userId)}/approve`, { revision });
}

export function rejectSellerDetails(userId: string, revision: number, reason: string): Promise<AdminSellerDetails> {
  return api.post(`/admin/seller-details/${encodeURIComponent(userId)}/reject`, { revision, reason });
}
