import { api } from "@/lib/api";
import type {
  BrandColor,
  PayoutAccount,
  PayoutMethod,
  PhotographerRegion,
  SocialLink,
  SocialPlatform,
  VerificationStatus,
} from "@/store/photographer-settings-store";

// Phase F.2 photographer-settings backend contract — see notes/photographer-settings-backend.md
//
// ─── Reference data (regions) ─────────────────────────────────────────────
//   GET    /api/v1/regions                                          → ApiRegion[]
//
// ─── Profile-shaped fields ────────────────────────────────────────────────
//   PUT    /api/v1/me/photographer/brand     { brandName, brandColor, bio } → ok
//   PUT    /api/v1/me/photographer/handle    { handle } (409 RESERVED_HANDLE / HANDLE_TAKEN)
//   PUT    /api/v1/me/photographer/region    { regionCode, provinceCode }
//   POST   /api/v1/me/photographer/cover     (multipart, file) → { url, uploadedAt }
//   POST   /api/v1/me/photographer/watermark (multipart, file) → { url, uploadedAt }
//
// ─── Socials CRUD ─────────────────────────────────────────────────────────
//   GET    /api/v1/me/photographer/socials                          → SocialLink[]
//   POST   /api/v1/me/photographer/socials   { platform, url }      → SocialLink
//   PATCH  /api/v1/me/photographer/socials/{id}   { url }           → SocialLink
//   DELETE /api/v1/me/photographer/socials/{id}                     → ok
//
// ─── Payouts CRUD + primary toggle (Q-E1 + Q-E2 RESOLVED) ─────────────────
//   GET    /api/v1/me/photographer/payouts                          → PayoutAccount[]
//   POST   /api/v1/me/photographer/payouts   { method, accountNumber, accountName }
//          → PayoutAccount   (first auto-primary)
//   PATCH  /api/v1/me/photographer/payouts/{id} { accountNumber?, accountName? }
//          → PayoutAccount
//   DELETE /api/v1/me/photographer/payouts/{id}                     → ok
//   PATCH  /api/v1/me/photographer/payouts/{id}/primary             → PayoutAccount[]
//   POST   /api/v1/me/photographer/payouts/{id}/qr (multipart)      → PayoutAccount
//
// ─── Verification submit ──────────────────────────────────────────────────
//   POST   /api/v1/me/photographer/verification                     → { status }
//
// ─── Event coupons ─────────────────────────────────────────────────────────
// Event coupon CRUD is event-scoped in api-coupons.ts.

// ───────────────────────────────────────────── Regions reference

export interface ApiProvince {
  code: string;
  name: string;
}

export interface ApiRegion {
  code: string;
  name: string;
  shortName: string;
  group: "luzon" | "visayas" | "mindanao";
  provinces: ApiProvince[];
}

export async function fetchRegions(): Promise<ApiRegion[] | null> {
  return api.get<ApiRegion[]>("/regions");
}

// ───────────────────────────────────────────── Brand / Handle / Region

export interface BrandPatch {
  brandName: string;
  brandColor: BrandColor;
  bio: string;
}

// Despite the URL, `GET /me/photographer/brand` returns the full profile-shaped
// settings state — brand + handle + region + presigned cover/watermark/avatar
// URLs. It's the only aggregated read on the photographer-settings surface; the
// settings page hydrates from this on mount so persisted-Zustand state cleared
// by resetUserScopedStores (lib/auth-reset.ts) is repopulated from the BE
// instead of rendering blank slabs for a verified photographer.
export interface PhotographerBrandResponse {
  brandName: string;
  brandColor: string;
  bio: string;
  handle: string;
  regionCode: string;
  provinceCode: string;
  coverUrl: string;
  watermarkUrl: string;
  avatarUrl: string;
}

export async function fetchBrand(): Promise<PhotographerBrandResponse | null> {
  return api.get<PhotographerBrandResponse>("/me/photographer/brand");
}

export async function putBrand(patch: BrandPatch): Promise<void> {
  await api.put<unknown>("/me/photographer/brand", patch);
}

export async function putHandle(handle: string): Promise<void> {
  await api.put<unknown>("/me/photographer/handle", { handle });
}

export async function putRegion(region: PhotographerRegion): Promise<void> {
  await api.put<unknown>("/me/photographer/region", region);
}

// ───────────────────────────────────────────── Cover / Watermark (multipart)

export interface MediaUploadResponse {
  url: string;
  uploadedAt: string;
}

async function postMultipart(
  path: string,
  file: File,
  fieldName: string = "file",
): Promise<MediaUploadResponse> {
  const fd = new FormData();
  fd.append(fieldName, file);
  return api.post<MediaUploadResponse>(path, fd);
}

export async function postCover(file: File): Promise<MediaUploadResponse> {
  return postMultipart("/me/photographer/cover", file);
}

export async function postWatermark(file: File): Promise<MediaUploadResponse> {
  return postMultipart("/me/photographer/watermark", file);
}

// ───────────────────────────────────────────── Coupon (V45)

// ───────────────────────────────────────────── Socials CRUD

export async function fetchSocials(): Promise<SocialLink[]> {
  return api.get<SocialLink[]>("/me/photographer/socials");
}

export async function postSocial(
  platform: SocialPlatform,
  url: string,
): Promise<SocialLink> {
  return api.post<SocialLink>("/me/photographer/socials", { platform, url });
}

export async function patchSocial(
  id: string,
  url: string,
): Promise<SocialLink> {
  return api.fetch<SocialLink>(
    `/me/photographer/socials/${encodeURIComponent(id)}`,
    { method: "PATCH", body: JSON.stringify({ url }) },
  );
}

export async function deleteSocial(id: string): Promise<void> {
  await api.delete<unknown>(`/me/photographer/socials/${encodeURIComponent(id)}`);
}

// ───────────────────────────────────────────── Payouts CRUD + primary

export async function fetchPayoutAccounts(): Promise<PayoutAccount[]> {
  return api.get<PayoutAccount[]>("/me/photographer/payouts");
}

export interface CreatePayoutAccountArgs {
  method: PayoutMethod;
  accountNumber: string;
  accountName: string;
}

export async function postPayoutAccount(
  args: CreatePayoutAccountArgs,
): Promise<PayoutAccount> {
  return api.post<PayoutAccount>("/me/photographer/payouts", args);
}

export interface PayoutAccountPatch {
  accountNumber?: string;
  accountName?: string;
}

export async function patchPayoutAccount(
  id: string,
  patch: PayoutAccountPatch,
): Promise<PayoutAccount> {
  return api.fetch<PayoutAccount>(
    `/me/photographer/payouts/${encodeURIComponent(id)}`,
    { method: "PATCH", body: JSON.stringify(patch) },
  );
}

export async function deletePayoutAccount(id: string): Promise<void> {
  await api.delete<unknown>(`/me/photographer/payouts/${encodeURIComponent(id)}`);
}

export async function patchPrimaryPayout(
  id: string,
): Promise<PayoutAccount[]> {
  return api.fetch<PayoutAccount[]>(
    `/me/photographer/payouts/${encodeURIComponent(id)}/primary`,
    { method: "PATCH" },
  );
}

export async function postPayoutQr(
  id: string,
  file: File,
): Promise<PayoutAccount> {
  const fd = new FormData();
  fd.append("file", file);
  return api.post<PayoutAccount>(
    `/me/photographer/payouts/${encodeURIComponent(id)}/qr`,
    fd,
  );
}

// ───────────────────────────────────────────── Verification submit + status

export interface VerificationSubmitResponse {
  status: VerificationStatus;
  missing?: string[] | null;
  // Populated when admin has suspended the account. Orthogonal to status
  // — an approved photographer can also be suspended, in which case the
  // FE prioritises the suspended state for gating + banner copy.
  suspendedAt?: string | null;
  suspensionReason?: string | null;
}

export async function submitVerification(): Promise<VerificationSubmitResponse> {
  return api.post<VerificationSubmitResponse>("/me/photographer/verification");
}

// Phase 4 — read-only status poll. Used by the photographer's settings page
// to detect admin approve/reject decisions across sessions without forcing
// a full page refresh. Symmetric with the POST above; never writes.
export async function fetchVerificationStatus(): Promise<VerificationSubmitResponse> {
  return api.get<VerificationSubmitResponse>("/me/photographer/verification");
}

// Phase 5 — photographer-side rescind. Flips a PENDING (or APPROVED) row
// back to INCOMPLETE so the photographer can edit + re-submit, or just
// remove themselves from the admin queue entirely. Idempotent.
export async function withdrawVerification(): Promise<VerificationSubmitResponse> {
  return api.post<VerificationSubmitResponse>(
    "/me/photographer/verification/withdraw",
  );
}
