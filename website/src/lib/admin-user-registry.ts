import type { VerificationStatus } from "@/store/photographer-settings-store";

// Admin-side user row type. Mirrors the BE's AdminUserRowDto returned from
// GET /api/v1/admin/users. After Phase 5 cleanup, this file holds types
// only — the seed array + dead helper functions were removed (BE is the
// source of truth via lib/admin-users-data.ts).
//
// Photographers carry a `settingsSnapshot` capturing the field-completeness
// state the admin reviews. Runners snapshot is null (no settings to gate on).

export type AdminRole = "PHOTOGRAPHER" | "RUNNER";

export interface PhotographerSettingsSnapshot {
  hasCover: boolean;
  hasBrandName: boolean;
  hasWatermark: boolean;
  hasHandle: boolean;
  hasRegion: boolean;
  socialCount: number;
  payoutCount: number;
}

export interface AdminUserRow {
  userId: string;
  role: AdminRole;
  email: string;
  name: string;
  /** Presigned avatar URL from the BE (via UserDtoMapper.resolveAvatarUrl).
   *  Null when the photographer hasn't uploaded one. */
  avatarUrl: string | null;
  brandName: string | null;
  handle: string | null;
  region: string | null;
  city: string;
  createdAt: string;
  verificationStatus: VerificationStatus;
  suspendedAt: string | null;
  suspensionReason: string | null;
  settingsSnapshot: PhotographerSettingsSnapshot | null;
}
