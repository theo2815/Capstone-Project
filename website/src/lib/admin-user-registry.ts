import type { VerificationStatus } from "@/store/photographer-settings-store";

// Mock admin-side user directory. Holds the seed for every photographer +
// runner the admin can see. Real backend ships in Phase F as
// `GET /admin/users?role=...`. The single logged-in user in `auth-store` is
// orthogonal — that's the session, this is the directory.
//
// Photographers carry a `settingsSnapshot` capturing the field-completeness
// state the admin reviews. Runners snapshot is null (no settings to gate
// on).

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

const fullSnapshot: PhotographerSettingsSnapshot = {
  hasCover: true,
  hasBrandName: true,
  hasWatermark: true,
  hasHandle: true,
  hasRegion: true,
  socialCount: 2,
  payoutCount: 1,
};

const partialSnapshot: PhotographerSettingsSnapshot = {
  hasCover: true,
  hasBrandName: true,
  hasWatermark: true,
  hasHandle: true,
  hasRegion: true,
  socialCount: 1,
  payoutCount: 1,
};

const incompleteSnapshot: PhotographerSettingsSnapshot = {
  hasCover: false,
  hasBrandName: true,
  hasWatermark: false,
  hasHandle: true,
  hasRegion: false,
  socialCount: 0,
  payoutCount: 0,
};

export const ADMIN_USER_SEED: ReadonlyArray<AdminUserRow> = [];

export function getUserById(
  userId: string,
  pool: ReadonlyArray<AdminUserRow> = ADMIN_USER_SEED,
): AdminUserRow | undefined {
  return pool.find((u) => u.userId === userId);
}

export function getPendingPhotographers(
  pool: ReadonlyArray<AdminUserRow> = ADMIN_USER_SEED,
): AdminUserRow[] {
  return pool.filter(
    (u) => u.role === "PHOTOGRAPHER" && u.verificationStatus === "pending",
  );
}

export function getIncompletePhotographers(
  pool: ReadonlyArray<AdminUserRow> = ADMIN_USER_SEED,
): AdminUserRow[] {
  return pool.filter(
    (u) => u.role === "PHOTOGRAPHER" && u.verificationStatus === "incomplete",
  );
}

export function getApprovedPhotographers(
  pool: ReadonlyArray<AdminUserRow> = ADMIN_USER_SEED,
): AdminUserRow[] {
  return pool.filter(
    (u) =>
      u.role === "PHOTOGRAPHER" &&
      u.verificationStatus === "approved" &&
      u.suspendedAt === null,
  );
}
