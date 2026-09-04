import { create } from "zustand";
import { persist } from "zustand/middleware";

// TODO(backend): swap localStorage persist for Spring Boot Phase F
// (`/me/photographer/settings`). State shape mirrors the eventual API
// contract — cover, brand, watermark are URL-bearing fields on the server.
// Until then, data URLs are stored client-side which has a ~5 MB total
// localStorage budget — `lib/image-utils.ts` downscales before writing.
//
// Backend ownership of new collections (added 2026-05-06 PM):
// - region: read-write via `PUT /me/photographer/region`. The list of regions
//   + provinces is owned by the backend (see `lib/ph-regions.ts` mock; replace
//   with `GET /api/regions`). FE only stores the selected codes.
// - socials: managed via `GET/POST/DELETE /me/photographer/socials`. The
//   server validates URLs match the platform domain and enforces ≥1 row.
// - payouts: managed via `GET/POST/DELETE /me/photographer/payouts` plus
//   `PATCH /me/photographer/payouts/{id}/primary` for primary-account
//   selection. The server enforces the "exactly one primary" invariant and
//   handles the auto-failover logic for unavailable primaries.

export type VerificationStatus = "incomplete" | "pending" | "approved";

// Locked palette of brand accent colors. `none` means the photographer
// inherits the global fresh accent on their public profile (default for
// runners; opt-in only). Other entries map to CSS color values used as
// thin bands on the cover banner and the brand-name dot on /{handle}.
export type BrandColor =
  | "none"
  | "fresh"
  | "amber"
  | "indigo"
  | "rose"
  | "ink";

export const BRAND_COLOR_HEX: Record<BrandColor, string> = {
  none: "transparent",
  fresh: "#2D9E5E",
  amber: "#D97706",
  indigo: "#4F46E5",
  rose: "#E11D48",
  ink: "#111111",
};

export const BRAND_COLOR_LABEL: Record<BrandColor, string> = {
  none: "No accent",
  fresh: "Fresh",
  amber: "Amber",
  indigo: "Indigo",
  rose: "Rose",
  ink: "Ink",
};

export interface CoverMedia {
  dataUrl: string;
  uploadedAt: string;
}

export interface WatermarkMedia {
  dataUrl: string;
  uploadedAt: string;
}

// Photographer coupon (V45). One per photographer; `expiresAt` is an ISO
// datetime or null. The discount is a percentage of the photographer's own
// share of each sale — the platform cut never moves. Managed via
// GET/PUT/DELETE /me/photographer/coupon.
export interface PhotographerCoupon {
  code: string;
  percentOff: number;
  active: boolean;
  expiresAt: string | null;
}

// ─── Region ──────────────────────────────────────────────────────────────
// Codes resolve against `lib/ph-regions.ts` (mock; backend will own the list
// via `GET /api/regions`). Storing codes — not display names — so the FE can
// re-render labels after a backend rename without a client migration.

export interface PhotographerRegion {
  regionCode: string;
  provinceCode: string;
}

// ─── Socials ─────────────────────────────────────────────────────────────
// Toggle-based list. Photographer adds one row per platform they're on.
// Minimum 1 required for verification. Multiple rows of the same platform
// are allowed (e.g., personal + business Facebook).

export const SOCIAL_PLATFORMS = [
  "facebook",
  "instagram",
  "tiktok",
  "x",
  "youtube",
  "website",
] as const;
export type SocialPlatform = (typeof SOCIAL_PLATFORMS)[number];

export const SOCIAL_PLATFORM_LABEL: Record<SocialPlatform, string> = {
  facebook: "Facebook",
  instagram: "Instagram",
  tiktok: "TikTok",
  x: "X",
  youtube: "YouTube",
  website: "Website",
};

/** Mono-cap two-letter tile shown next to each social row. */
export const SOCIAL_PLATFORM_TILE: Record<SocialPlatform, string> = {
  facebook: "FB",
  instagram: "IG",
  tiktok: "TT",
  x: "X",
  youtube: "YT",
  website: "WEB",
};

export interface SocialLink {
  id: string;
  platform: SocialPlatform;
  url: string;
}

// ─── Payouts ─────────────────────────────────────────────────────────────
// Toggle-based list of payout accounts. ≥1 required. Exactly one is the
// `primary` — the default destination for sales transfers. The rest are
// backups; if the primary is unavailable (closed account, frozen number),
// the backend falls back to the next backup in order. FE only owns the
// "which one is primary" toggle; backend owns the failover behavior.

export const PAYOUT_METHODS = ["gcash", "maya", "gotyme"] as const;
export type PayoutMethod = (typeof PAYOUT_METHODS)[number];

export const PAYOUT_METHOD_LABEL: Record<PayoutMethod, string> = {
  gcash: "GCash",
  maya: "Maya",
  gotyme: "GoTyme",
};

/** Brand color tokens used to differentiate payout cards visually. */
export const PAYOUT_METHOD_HEX: Record<PayoutMethod, string> = {
  gcash: "#0078FF",
  maya: "#00DC78",
  gotyme: "#F5BB1D",
};

export interface PayoutQr {
  dataUrl: string;
  uploadedAt: string;
}

export interface PayoutAccount {
  id: string;
  method: PayoutMethod;
  /** GCash + Maya: 11-digit phone (e.g. 09175550101). GoTyme: bank acct number. Stored digits-only. */
  accountNumber: string;
  accountName: string;
  qr: PayoutQr | null;
  isPrimary: boolean;
}

export interface PhotographerSettings {
  cover: CoverMedia | null;
  brandName: string;
  brandColor: BrandColor;
  bio: string;
  watermark: WatermarkMedia | null;
  handle: string;
  verificationStatus: VerificationStatus;
  /** ISO timestamp when admin suspended the account, or null. Orthogonal
   *  to verificationStatus — suspended takes priority for gating + banners. */
  suspendedAt: string | null;
  suspensionReason: string | null;
  region: PhotographerRegion | null;
  socials: SocialLink[];
  payouts: PayoutAccount[];
  coupon: PhotographerCoupon | null;
}

interface PhotographerSettingsState extends PhotographerSettings {
  setCover: (cover: CoverMedia | null) => void;
  setBrandName: (name: string) => void;
  setBrandColor: (color: BrandColor) => void;
  setBio: (bio: string) => void;
  setWatermark: (watermark: WatermarkMedia | null) => void;
  setHandle: (handle: string) => void;
  setVerificationStatus: (status: VerificationStatus) => void;
  setSuspension: (suspendedAt: string | null, reason: string | null) => void;
  setRegion: (region: PhotographerRegion | null) => void;
  addSocial: (platform: SocialPlatform, url: string) => string;
  updateSocial: (id: string, url: string) => void;
  removeSocial: (id: string) => void;
  addPayout: (input: {
    method: PayoutMethod;
    accountNumber: string;
    accountName: string;
    qr: PayoutQr | null;
  }) => string;
  updatePayout: (
    id: string,
    patch: Partial<Omit<PayoutAccount, "id" | "isPrimary">>,
  ) => void;
  setPrimaryPayout: (id: string) => void;
  removePayout: (id: string) => void;
  setCoupon: (coupon: PhotographerCoupon | null) => void;
  isComplete: () => boolean;
  reset: () => void;
}

const SEED: PhotographerSettings = {
  cover: null,
  brandName: "",
  brandColor: "none",
  bio: "",
  watermark: null,
  handle: "",
  verificationStatus: "incomplete",
  suspendedAt: null,
  suspensionReason: null,
  region: null,
  socials: [],
  payouts: [],
  coupon: null,
};

function newId(): string {
  if (
    typeof globalThis !== "undefined" &&
    typeof globalThis.crypto?.randomUUID === "function"
  ) {
    return globalThis.crypto.randomUUID();
  }
  // Fallback for non-browser SSR pass — collisions extremely unlikely at this scale.
  return `id-${Math.random().toString(36).slice(2)}-${Date.now().toString(36)}`;
}

export const usePhotographerSettingsStore = create<PhotographerSettingsState>()(
  persist(
    (set, get) => ({
      ...SEED,
      setCover: (cover) => set({ cover }),
      setBrandName: (name) => set({ brandName: name }),
      setBrandColor: (color) => set({ brandColor: color }),
      setBio: (bio) => set({ bio }),
      setWatermark: (watermark) => set({ watermark }),
      setHandle: (handle) => set({ handle: handle.trim().toLowerCase() }),
      setVerificationStatus: (verificationStatus) =>
        set({ verificationStatus }),
      setSuspension: (suspendedAt, suspensionReason) =>
        set({ suspendedAt, suspensionReason }),
      setRegion: (region) => set({ region }),
      setCoupon: (coupon) => set({ coupon }),
      addSocial: (platform, url) => {
        const id = newId();
        set((s) => ({
          socials: [...s.socials, { id, platform, url: url.trim() }],
        }));
        return id;
      },
      updateSocial: (id, url) =>
        set((s) => ({
          socials: s.socials.map((sl) =>
            sl.id === id ? { ...sl, url: url.trim() } : sl,
          ),
        })),
      removeSocial: (id) =>
        set((s) => ({ socials: s.socials.filter((sl) => sl.id !== id) })),
      addPayout: ({ method, accountNumber, accountName, qr }) => {
        const id = newId();
        set((s) => {
          const isFirst = s.payouts.length === 0;
          return {
            payouts: [
              ...s.payouts,
              {
                id,
                method,
                accountNumber: accountNumber.replace(/\D/g, ""),
                accountName: accountName.trim(),
                qr,
                isPrimary: isFirst,
              },
            ],
          };
        });
        return id;
      },
      updatePayout: (id, patch) =>
        set((s) => ({
          payouts: s.payouts.map((p) =>
            p.id === id
              ? {
                  ...p,
                  ...patch,
                  accountNumber:
                    patch.accountNumber !== undefined
                      ? patch.accountNumber.replace(/\D/g, "")
                      : p.accountNumber,
                  accountName:
                    patch.accountName !== undefined
                      ? patch.accountName.trim()
                      : p.accountName,
                }
              : p,
          ),
        })),
      setPrimaryPayout: (id) =>
        set((s) => ({
          payouts: s.payouts.map((p) => ({ ...p, isPrimary: p.id === id })),
        })),
      removePayout: (id) =>
        set((s) => {
          const filtered = s.payouts.filter((p) => p.id !== id);
          // If we just removed the primary, promote the first remaining
          // account so the "exactly one primary" invariant holds when the
          // list is non-empty.
          const stillHasPrimary = filtered.some((p) => p.isPrimary);
          if (!stillHasPrimary && filtered.length > 0) {
            filtered[0] = { ...filtered[0], isPrimary: true };
          }
          return { payouts: filtered };
        }),
      isComplete: () => {
        const s = get();
        return (
          s.cover !== null &&
          s.brandName.trim().length > 0 &&
          s.watermark !== null &&
          s.handle.trim().length > 0 &&
          s.region !== null &&
          s.socials.some((sl) => sl.url.trim().length > 0) &&
          s.payouts.some((p) => p.isPrimary)
        );
      },
      reset: () => set(SEED),
    }),
    { name: "quickpitik-photographer-settings" },
  ),
);
