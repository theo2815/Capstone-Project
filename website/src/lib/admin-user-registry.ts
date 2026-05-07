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

export const ADMIN_USER_SEED: ReadonlyArray<AdminUserRow> = [
  // ─── Approved photographers (match PHOTOGRAPHER_REGISTRY) ────────────────
  {
    userId: "photog-paksit",
    role: "PHOTOGRAPHER",
    email: "team@paksitphotos.ph",
    name: "Paksit Studio",
    brandName: "Paksit Photos",
    handle: "paksitphotos",
    region: "Cebu · Central Visayas",
    city: "Cebu City",
    createdAt: "2024-08-15",
    verificationStatus: "approved",
    suspendedAt: null,
    suspensionReason: null,
    settingsSnapshot: fullSnapshot,
  },
  {
    userId: "photog-cebustride",
    role: "PHOTOGRAPHER",
    email: "hello@cebustride.com",
    name: "Cebu Stride",
    brandName: "Cebu Stride",
    handle: "cebustride",
    region: "Cebu · Central Visayas",
    city: "Mandaue",
    createdAt: "2024-11-02",
    verificationStatus: "approved",
    suspendedAt: null,
    suspensionReason: null,
    settingsSnapshot: fullSnapshot,
  },

  // ─── Pending photographers (queue for the admin to act on) ───────────────
  {
    userId: "photog-mango",
    role: "PHOTOGRAPHER",
    email: "mango.frames@gmail.com",
    name: "Maria Antonio",
    brandName: "Mango Frames",
    handle: "mangoframes",
    region: "Cebu · Central Visayas",
    city: "Lapu-Lapu",
    createdAt: "2026-04-22",
    verificationStatus: "pending",
    suspendedAt: null,
    suspensionReason: null,
    settingsSnapshot: partialSnapshot,
  },
  {
    userId: "photog-southshot",
    role: "PHOTOGRAPHER",
    email: "carlos@southshot.ph",
    name: "Carlos Reyes",
    brandName: "South Shot",
    handle: "southshot",
    region: "Cebu · Central Visayas",
    city: "Talisay",
    createdAt: "2026-05-01",
    verificationStatus: "pending",
    suspendedAt: null,
    suspensionReason: null,
    settingsSnapshot: fullSnapshot,
  },
  {
    userId: "photog-trailroots",
    role: "PHOTOGRAPHER",
    email: "ana@trailroots.ph",
    name: "Ana Villanueva",
    brandName: "Trail Roots",
    handle: "trailroots",
    region: "Cebu · Central Visayas",
    city: "Cordova",
    createdAt: "2026-05-04",
    verificationStatus: "pending",
    suspendedAt: null,
    suspensionReason: null,
    settingsSnapshot: partialSnapshot,
  },

  // ─── Incomplete photographers (settings still missing fields) ────────────
  {
    userId: "photog-newgen",
    role: "PHOTOGRAPHER",
    email: "kevin.ng@gmail.com",
    name: "Kevin Ng",
    brandName: "New Gen Photos",
    handle: "newgenphotos",
    region: null,
    city: "Cebu City",
    createdAt: "2026-04-30",
    verificationStatus: "incomplete",
    suspendedAt: null,
    suspensionReason: null,
    settingsSnapshot: incompleteSnapshot,
  },
  {
    userId: "photog-gridrun",
    role: "PHOTOGRAPHER",
    email: "team@gridrun.ph",
    name: "Maya Lim",
    brandName: "Grid Run",
    handle: "gridrun",
    region: null,
    city: "Mandaue",
    createdAt: "2026-05-03",
    verificationStatus: "incomplete",
    suspendedAt: null,
    suspensionReason: null,
    settingsSnapshot: { ...incompleteSnapshot, hasCover: true, socialCount: 1 },
  },

  // ─── Runners (no settings snapshot — they don't need verification) ───────
  {
    userId: "runner-juan",
    role: "RUNNER",
    email: "juan.delacruz@gmail.com",
    name: "Juan dela Cruz",
    brandName: null,
    handle: null,
    region: "Cebu · Central Visayas",
    city: "Cebu City",
    createdAt: "2026-01-12",
    verificationStatus: "approved",
    suspendedAt: null,
    suspensionReason: null,
    settingsSnapshot: null,
  },
  {
    userId: "runner-thea",
    role: "RUNNER",
    email: "thea.santos@gmail.com",
    name: "Thea Santos",
    brandName: null,
    handle: null,
    region: "Cebu · Central Visayas",
    city: "Mandaue",
    createdAt: "2026-02-04",
    verificationStatus: "approved",
    suspendedAt: null,
    suspensionReason: null,
    settingsSnapshot: null,
  },
  {
    userId: "runner-nico",
    role: "RUNNER",
    email: "nico.aquino@gmail.com",
    name: "Nico Aquino",
    brandName: null,
    handle: null,
    region: "Cebu · Central Visayas",
    city: "Lapu-Lapu",
    createdAt: "2026-03-18",
    verificationStatus: "approved",
    suspendedAt: null,
    suspensionReason: null,
    settingsSnapshot: null,
  },
  {
    userId: "runner-maria",
    role: "RUNNER",
    email: "maria.tan@gmail.com",
    name: "Maria Tan",
    brandName: null,
    handle: null,
    region: "Cebu · Central Visayas",
    city: "Cebu City",
    createdAt: "2026-04-01",
    verificationStatus: "approved",
    suspendedAt: null,
    suspensionReason: null,
    settingsSnapshot: null,
  },
  {
    userId: "runner-jp",
    role: "RUNNER",
    email: "jp.alvarez@gmail.com",
    name: "JP Alvarez",
    brandName: null,
    handle: null,
    region: "Cebu · Central Visayas",
    city: "Talisay",
    createdAt: "2026-04-22",
    verificationStatus: "approved",
    suspendedAt: null,
    suspensionReason: null,
    settingsSnapshot: null,
  },
];

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
