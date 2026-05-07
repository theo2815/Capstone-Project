// Mock-only seed of photographer settings keyed by userId. Lets the admin
// detail page + verification drawer render the FULL settings (cover, brand,
// watermark, public URL, region, socials, payouts) for any photographer in
// ADMIN_USER_SEED — not just the currently-logged-in one. Without this
// seed, useAdminPhotographerView only had liveSettings (session-scoped) so
// non-self photographers like trailroots displayed an empty Payouts slab
// and zero socials/region/watermark.
//
// TODO(backend): Phase F replaces this with `GET /admin/photographers/{id}/settings`.
// Same shape; backend serves signed S3 URLs for cover/watermark instead of
// the kind/label preview shape used here.
//
// Watermark shape note: real photographer settings store an uploaded PNG as
// a data URL. We don't ship binary previews in the seed — instead each
// seed entry carries a `WatermarkPreview` with kind="label" so the slab
// renders the photographer's mono brand chip as a placeholder. When the
// session photographer is the one being viewed, `useAdminPhotographerView`
// surfaces the real WatermarkMedia from the live store and the slab
// prefers that. Same idea for cover.

import type {
  BrandColor,
  PayoutAccount,
  PhotographerRegion,
  SocialLink,
} from "@/store/photographer-settings-store";
import type { CoverSource } from "@/lib/photographer-registry";

// Watermark in the live store is a PNG data URL. Seeded photographers don't
// carry binary previews — they expose a "label" variant the watermark slab
// renders as a mono brand chip placeholder. The live photographer always
// surfaces "image" via useAdminPhotographerView's effectiveSettings.
export type WatermarkPreview =
  | { kind: "label"; label: string }
  | { kind: "image"; dataUrl: string };

export interface AdminPhotographerSettingsSeed {
  brandName: string;
  brandColor: BrandColor;
  bio: string;
  handle: string;
  cover: CoverSource | null;
  watermark: WatermarkPreview | null;
  region: PhotographerRegion | null;
  socials: SocialLink[];
  payouts: PayoutAccount[];
}

// All photographers currently in ADMIN_USER_SEED. Approved + pending +
// incomplete entries get full or partial settings depending on their
// review state — this matches the snapshot shape (hasCover/hasBrand/etc).
export const ADMIN_PHOTOGRAPHER_SETTINGS_SEED: Record<
  string,
  AdminPhotographerSettingsSeed
> = {
  // ─── Approved ─────────────────────────────────────────────────────────────

  "photog-paksit": {
    brandName: "Paksit Photos",
    brandColor: "amber",
    bio: "Cebu-based race photographer. Bridge climbs, finish-line tape, golden-hour bib portraits. Shooting since 2019.",
    handle: "paksitphotos",
    cover: { kind: "gradient", from: "#D97706", to: "#7C2D12" },
    watermark: { kind: "label", label: "PAKSIT" },
    region: { regionCode: "region-7", provinceCode: "cebu" },
    socials: [
      { id: "soc-paksit-1", platform: "facebook", url: "https://facebook.com/paksitphotos" },
      { id: "soc-paksit-2", platform: "instagram", url: "https://instagram.com/paksitphotos" },
    ],
    payouts: [
      {
        id: "po-paksit-1",
        method: "gcash",
        accountNumber: "09175550101",
        accountName: "Paksit Studio",
        qr: null,
        isPrimary: true,
      },
    ],
  },

  "photog-cebustride": {
    brandName: "Cebu Stride",
    brandColor: "indigo",
    bio: "Two-photographer team covering Cebu road races since 2022. Action-first, candid, no posed shots.",
    handle: "cebustride",
    cover: { kind: "gradient", from: "#4F46E5", to: "#1E1B4B" },
    watermark: { kind: "label", label: "CSC" },
    region: { regionCode: "region-7", provinceCode: "cebu" },
    socials: [
      { id: "soc-cebustride-1", platform: "facebook", url: "https://facebook.com/cebustride" },
      { id: "soc-cebustride-2", platform: "instagram", url: "https://instagram.com/cebustride" },
    ],
    payouts: [
      {
        id: "po-cebustride-1",
        method: "maya",
        accountNumber: "09175550202",
        accountName: "Cebu Stride Co.",
        qr: null,
        isPrimary: true,
      },
    ],
  },

  // ─── Pending — full settings ready for verification review ────────────────

  "photog-mango": {
    brandName: "Mango Frames",
    brandColor: "amber",
    bio: "Solo photographer covering Cebu trail and sunrise races. Specializing in coastal and bridge shots.",
    handle: "mangoframes",
    cover: { kind: "gradient", from: "#F59E0B", to: "#92400E" },
    watermark: { kind: "label", label: "MANGO" },
    region: { regionCode: "region-7", provinceCode: "cebu" },
    socials: [
      { id: "soc-mango-1", platform: "instagram", url: "https://instagram.com/mangoframes.ph" },
    ],
    payouts: [
      {
        id: "po-mango-1",
        method: "gcash",
        accountNumber: "09175550303",
        accountName: "Maria Antonio",
        qr: null,
        isPrimary: true,
      },
    ],
  },

  "photog-southshot": {
    brandName: "South Shot",
    brandColor: "rose",
    bio: "Talisay-based race photographer. Sprint finishes, expression-first portraits, never staged.",
    handle: "southshot",
    cover: { kind: "gradient", from: "#E11D48", to: "#581C87" },
    watermark: { kind: "label", label: "SOUTH" },
    region: { regionCode: "region-7", provinceCode: "cebu" },
    socials: [
      { id: "soc-south-1", platform: "facebook", url: "https://facebook.com/southshotph" },
      { id: "soc-south-2", platform: "instagram", url: "https://instagram.com/southshot.ph" },
    ],
    payouts: [
      {
        id: "po-south-1",
        method: "gcash",
        accountNumber: "09175550404",
        accountName: "Carlos Reyes",
        qr: null,
        isPrimary: true,
      },
      {
        id: "po-south-2",
        method: "gotyme",
        accountNumber: "017000345678",
        accountName: "Carlos Reyes",
        qr: null,
        isPrimary: false,
      },
    ],
  },

  "photog-trailroots": {
    brandName: "Trail Roots",
    brandColor: "fresh",
    bio: "Trail and ultra coverage across Cebu. Forest singletrack, dawn starts, and the long quiet middle of the race.",
    handle: "trailroots",
    cover: { kind: "gradient", from: "#2D9E5E", to: "#14532D" },
    watermark: { kind: "label", label: "TR" },
    region: { regionCode: "region-7", provinceCode: "cebu" },
    socials: [
      { id: "soc-trail-1", platform: "instagram", url: "https://instagram.com/trailroots" },
    ],
    payouts: [
      {
        id: "po-trail-1",
        method: "maya",
        accountNumber: "09175550505",
        accountName: "Ana Villanueva",
        qr: null,
        isPrimary: true,
      },
    ],
  },

  // ─── Incomplete — partial settings, missing fields the photographer
  // hasn't filled. The snapshot drives the UI but admin can still see what
  // IS filled in. ─────────────────────────────────────────────────────────

  "photog-newgen": {
    brandName: "New Gen Photos",
    brandColor: "none",
    bio: "",
    handle: "newgenphotos",
    cover: null,
    watermark: null,
    region: null,
    socials: [],
    payouts: [],
  },

  "photog-gridrun": {
    brandName: "Grid Run",
    brandColor: "ink",
    bio: "Mandaue-based road race coverage.",
    handle: "gridrun",
    cover: { kind: "gradient", from: "#111111", to: "#3F3F46" },
    watermark: null,
    region: null,
    socials: [
      { id: "soc-grid-1", platform: "facebook", url: "https://facebook.com/gridrun" },
    ],
    payouts: [],
  },
};

export function getPhotographerSettingsSeed(
  userId: string,
): AdminPhotographerSettingsSeed | null {
  return ADMIN_PHOTOGRAPHER_SETTINGS_SEED[userId] ?? null;
}
