"use client";

import { useMemo } from "react";
import {
  ADMIN_USER_SEED,
  type AdminUserRow,
} from "@/lib/admin-user-registry";
import {
  PHOTOGRAPHER_REGISTRY,
  getPhotographerByHandle,
  type CoverSource,
  type PhotographerProfile,
} from "@/lib/photographer-registry";
import { useAdminUserStore, type DecisionLogEntry } from "@/store/admin-user-store";
import { useAuthStore } from "@/store/auth-store";
import {
  usePhotographerSettingsStore,
  type BrandColor,
  type PayoutAccount,
  type PhotographerRegion,
  type PhotographerSettings,
  type SocialLink,
} from "@/store/photographer-settings-store";
import {
  getPhotographerSettingsSeed,
  type AdminPhotographerSettingsSeed,
  type WatermarkPreview,
} from "@/lib/admin-photographer-settings-seed";

// Merged read of everything needed to render `/admin/photographers/[handle]`.
// Reconciles four sources:
//   1. AdminUserRow — directory shape (status, suspension, snapshot).
//   2. PhotographerProfile — rich public data (only 2 entries today).
//   3. PhotographerSettings — live state (only when handle matches session).
//   4. AdminPhotographerSettingsSeed — full settings for non-session
//      photographers so admin can review cover/watermark/socials/payouts
//      without needing to log in as that user.
// Plus the filtered admin decision log for the activity slab.

// Effective settings — display-only shape used by the detail page + the
// verification drawer. live > seed > null. The shape is intentionally a
// subset of PhotographerSettings (cover/watermark are display previews,
// not the raw store types) so it normalizes both branches.
export interface EffectivePhotographerSettings {
  brandName: string;
  brandColor: BrandColor;
  bio: string;
  handle: string;
  cover: CoverSource | null;
  watermark: WatermarkPreview | null;
  region: PhotographerRegion | null;
  socials: SocialLink[];
  payouts: PayoutAccount[];
  /** Provenance — useful for surfaces that want to show "live" badges. */
  source: "live" | "seed";
}

export interface AdminPhotographerView {
  row: AdminUserRow;
  profile: PhotographerProfile | null;
  liveSettings: PhotographerSettings | null;
  effectiveSettings: EffectivePhotographerSettings | null;
  decisions: DecisionLogEntry[];
}

function findRowByHandle(
  handle: string,
  effective: ReadonlyArray<AdminUserRow>,
): AdminUserRow | null {
  const normalized = handle.trim().toLowerCase();
  return effective.find((r) => r.handle === normalized) ?? null;
}

function liveToEffective(
  live: PhotographerSettings,
): EffectivePhotographerSettings {
  return {
    brandName: live.brandName,
    brandColor: live.brandColor,
    bio: live.bio,
    handle: live.handle,
    cover: live.cover ? { kind: "image", url: live.cover.dataUrl } : null,
    watermark: live.watermark
      ? { kind: "image", dataUrl: live.watermark.dataUrl }
      : null,
    region: live.region,
    socials: live.socials,
    payouts: live.payouts,
    source: "live",
  };
}

function seedToEffective(
  seed: AdminPhotographerSettingsSeed,
): EffectivePhotographerSettings {
  return {
    brandName: seed.brandName,
    brandColor: seed.brandColor,
    bio: seed.bio,
    handle: seed.handle,
    cover: seed.cover,
    watermark: seed.watermark,
    region: seed.region,
    socials: seed.socials,
    payouts: seed.payouts,
    source: "seed",
  };
}

// Non-reactive: useful for one-shot reads (server components, useEffect, etc).
// React components should prefer the hook below so override mutations re-render.
export function getAdminPhotographerView(
  handle: string,
): AdminPhotographerView | null {
  const overrides = useAdminUserStore.getState().overrides;
  const log = useAdminUserStore.getState().log;
  const effective = ADMIN_USER_SEED.map((row) => {
    const patch = overrides[row.userId];
    return patch ? { ...row, ...patch } : row;
  });
  const row = findRowByHandle(handle, effective);
  if (!row) return null;
  const profile = getPhotographerByHandle(handle);
  const sessionUser = useAuthStore.getState().user;
  const isSelf = !!sessionUser && sessionUser.id === row.userId;
  const liveSettings = isSelf ? usePhotographerSettingsStore.getState() : null;
  const seed = getPhotographerSettingsSeed(row.userId);
  const effectiveSettings: EffectivePhotographerSettings | null = liveSettings
    ? liveToEffective(liveSettings)
    : seed
      ? seedToEffective(seed)
      : null;
  const decisions = log.filter((e) => e.userId === row.userId);
  return { row, profile, liveSettings, effectiveSettings, decisions };
}

// Reactive subscriber. Subscribes to the stable underlying state (never
// `getXxx()` selectors) and derives the merged view via useMemo so React 19's
// useSyncExternalStore Object.is comparison stays stable.
export function useAdminPhotographerView(
  handle: string,
): AdminPhotographerView | null {
  const overrides = useAdminUserStore((s) => s.overrides);
  const log = useAdminUserStore((s) => s.log);
  const sessionUser = useAuthStore((s) => s.user);
  // Subscribe to the WHOLE settings slice — when own status flips via admin
  // approve/reject, this hook re-renders to reflect the new completeness.
  const liveSettings = usePhotographerSettingsStore();

  return useMemo<AdminPhotographerView | null>(() => {
    const effective = ADMIN_USER_SEED.map((row) => {
      const patch = overrides[row.userId];
      return patch ? { ...row, ...patch } : row;
    });
    const row = findRowByHandle(handle, effective);
    if (!row) return null;
    const profile = getPhotographerByHandle(handle);
    const isSelf = !!sessionUser && sessionUser.id === row.userId;
    const decisions = log.filter((e) => e.userId === row.userId);
    const seed = getPhotographerSettingsSeed(row.userId);
    const effectiveSettings: EffectivePhotographerSettings | null = isSelf
      ? liveToEffective(liveSettings)
      : seed
        ? seedToEffective(seed)
        : null;
    return {
      row,
      profile,
      liveSettings: isSelf ? liveSettings : null,
      effectiveSettings,
      decisions,
    };
  }, [handle, overrides, log, sessionUser, liveSettings]);
}

// Reactive lookup of effective settings for a single userId. Used by the
// verifications drawer (which knows the row's userId, not its handle) so it
// can render the same cover/watermark/public-URL/socials/payouts sections
// as /admin/photographers/[handle].
export function useEffectivePhotographerSettings(
  userId: string,
): EffectivePhotographerSettings | null {
  const sessionUser = useAuthStore((s) => s.user);
  const liveSettings = usePhotographerSettingsStore();
  return useMemo<EffectivePhotographerSettings | null>(() => {
    const isSelf = !!sessionUser && sessionUser.id === userId;
    if (isSelf) return liveToEffective(liveSettings);
    const seed = getPhotographerSettingsSeed(userId);
    return seed ? seedToEffective(seed) : null;
  }, [userId, sessionUser, liveSettings]);
}

// Synthetic gradient for photographers who don't have a PhotographerProfile
// (i.e. no entry in PHOTOGRAPHER_REGISTRY). Hash-derived from userId so each
// photographer gets a stable, unique-ish cover without us hand-crafting one.
export function syntheticCoverGradient(userId: string): { from: string; to: string } {
  let h = 2166136261;
  for (let i = 0; i < userId.length; i++) {
    h ^= userId.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  const hue = (h >>> 0) % 360;
  const from = `hsl(${hue}, 38%, 38%)`;
  const to = `hsl(${(hue + 30) % 360}, 32%, 14%)`;
  return { from, to };
}

// Re-export for convenience so detail page imports stay tight.
export { PHOTOGRAPHER_REGISTRY };
