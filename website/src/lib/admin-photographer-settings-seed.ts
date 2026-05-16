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
> = {};

export function getPhotographerSettingsSeed(
  userId: string,
): AdminPhotographerSettingsSeed | null {
  return ADMIN_PHOTOGRAPHER_SETTINGS_SEED[userId] ?? null;
}
