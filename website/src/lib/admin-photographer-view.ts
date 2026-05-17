"use client";

import { useMemo } from "react";
import { type AdminUserRow } from "@/lib/admin-user-registry";
import {
  PHOTOGRAPHER_REGISTRY,
  getPhotographerByHandle,
  type CoverSource,
  type PhotographerProfile,
} from "@/lib/photographer-registry";
import {
  useAdminUserStore,
  type DecisionLogEntry,
} from "@/store/admin-user-store";
import {
  getAdminUsersDataSnapshot,
  useAdminUsersData,
} from "@/lib/admin-users-data";
import { useAdminPhotographerSettings } from "@/lib/admin-photographer-settings-data";
import { type AdminPhotographerSettingsResponse } from "@/lib/api-admin";
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
  /** Presigned avatar URL from the BE, or the live user-media-store data URL
   *  when the admin happens to be the same session as the photographer. */
  avatarUrl: string | null;
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

// Live branch only fires for the session photographer's own row — pull
// the in-flight avatar URL from auth (Q-007 BE-canonical) or fall back
// to the local user-media-store data URL so the admin sees the same
// preview the photographer just picked.
function liveToEffective(
  live: PhotographerSettings,
  avatarUrl: string | null,
): EffectivePhotographerSettings {
  return {
    brandName: live.brandName,
    brandColor: live.brandColor,
    bio: live.bio,
    handle: live.handle,
    avatarUrl,
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
    avatarUrl: null,
    cover: seed.cover,
    watermark: seed.watermark,
    region: seed.region,
    socials: seed.socials,
    payouts: seed.payouts,
    source: "seed",
  };
}

// F-NEW-1 — convert the BE admin-photographer-settings shape to the
// FE-internal EffectivePhotographerSettings used by the drawer + detail
// page. `source: "live"` semantically means "authoritative from the
// backend" (whether the BE row or the photographer's in-progress edits) —
// distinct from `"seed"` which is the mock fallback.
function beToEffective(
  be: AdminPhotographerSettingsResponse,
): EffectivePhotographerSettings {
  const cover: EffectivePhotographerSettings["cover"] =
    be.cover?.url
      ? { kind: "image", url: be.cover.url }
      : be.cover?.gradientFrom && be.cover?.gradientTo
        ? {
            kind: "gradient",
            from: be.cover.gradientFrom,
            to: be.cover.gradientTo,
          }
        : null;
  const watermark: EffectivePhotographerSettings["watermark"] =
    be.watermark?.dataUrl
      ? { kind: "image", dataUrl: be.watermark.dataUrl }
      : be.watermark?.label
        ? { kind: "label", label: be.watermark.label }
        : null;
  return {
    brandName: be.brandName ?? "",
    brandColor: (be.brandColor as BrandColor) ?? "none",
    bio: be.bio,
    handle: be.handle ?? "",
    avatarUrl: be.avatarUrl ?? null,
    cover,
    watermark,
    region: be.region
      ? {
          regionCode: be.region.regionCode,
          provinceCode: be.region.provinceCode,
        }
      : null,
    socials: be.socials,
    payouts: be.payouts,
    source: "live",
  };
}

// Non-reactive: useful for one-shot reads (server components, useEffect, etc).
// React components should prefer the hook below so override mutations re-render.
export function getAdminPhotographerView(
  handle: string,
): AdminPhotographerView | null {
  const submissions = useAdminUserStore.getState().submissions;
  const log = useAdminUserStore.getState().log;
  const effective = getAdminUsersDataSnapshot();
  const row = findRowByHandle(handle, effective);
  if (!row) return null;
  const profile = getPhotographerByHandle(handle);
  const sessionUser = useAuthStore.getState().user;
  const isSelf = !!sessionUser && sessionUser.id === row.userId;
  const isSubmission = !!submissions[row.userId];
  const liveSettings =
    isSelf || isSubmission ? usePhotographerSettingsStore.getState() : null;
  const seed = getPhotographerSettingsSeed(row.userId);
  const effectiveSettings: EffectivePhotographerSettings | null = liveSettings
    ? liveToEffective(liveSettings, sessionUser?.avatarUrl ?? null)
    : seed
      ? seedToEffective(seed)
      : null;
  const decisions = log.filter((e) => e.userId === row.userId);
  return {
    row,
    profile,
    liveSettings: isSelf ? liveSettings : null,
    effectiveSettings,
    decisions,
  };
}

// Reactive subscriber. Subscribes to the merged admin user data + decision
// log; derives the merged view via useMemo so React 19's
// useSyncExternalStore Object.is comparison stays stable.
//
// Resolution chain for effectiveSettings mirrors useEffectivePhotographerSettings
// (used by the inbox drawer) so /admin/photographers/[handle] hydrates the
// same cover / watermark / socials / payouts from GET /admin/users/{id}/settings
// instead of falling through to the empty seed and rendering empty slabs.
export function useAdminPhotographerView(
  handle: string,
): AdminPhotographerView | null {
  const { rows: effective } = useAdminUsersData();
  const submissions = useAdminUserStore((s) => s.submissions);
  const log = useAdminUserStore((s) => s.log);
  const sessionUser = useAuthStore((s) => s.user);
  // Subscribe to the WHOLE settings slice — when own status flips via admin
  // approve/reject, this hook re-renders to reflect the new completeness.
  const liveSettings = usePhotographerSettingsStore();

  const row = useMemo(
    () => findRowByHandle(handle, effective),
    [handle, effective],
  );
  const isSelf = !!sessionUser && !!row && sessionUser.id === row.userId;
  const isSubmission = !!row && !!submissions[row.userId];
  const useLocal = isSelf || isSubmission;

  // Fetch BE settings for any non-local photographer; pass null otherwise
  // to skip the hit (session photographer's own row or a fresh submission
  // mirrored from /dashboard/settings).
  const { data: adminBeSettings } = useAdminPhotographerSettings(
    row && !useLocal ? row.userId : null,
  );

  return useMemo<AdminPhotographerView | null>(() => {
    if (!row) return null;
    const profile = getPhotographerByHandle(handle);
    const decisions = log.filter((e) => e.userId === row.userId);
    const effectiveSettings: EffectivePhotographerSettings | null = useLocal
      ? liveToEffective(liveSettings, sessionUser?.avatarUrl ?? null)
      : adminBeSettings
        ? beToEffective(adminBeSettings)
        : (() => {
            const seed = getPhotographerSettingsSeed(row.userId);
            return seed ? seedToEffective(seed) : null;
          })();
    return {
      row,
      profile,
      liveSettings: isSelf ? liveSettings : null,
      effectiveSettings,
      decisions,
    };
  }, [
    handle,
    row,
    log,
    useLocal,
    isSelf,
    liveSettings,
    adminBeSettings,
    sessionUser,
  ]);
}

// Reactive lookup of effective settings for a single userId. Used by the
// verifications drawer (which knows the row's userId, not its handle) so it
// can render the same cover/watermark/public-URL/socials/payouts sections
// as /admin/photographers/[handle]. Resolution order:
//   1. live (when isSelf, OR when the row is a self-submission) — always
//      reflects in-flight edits, which matches the design intent: edits
//      stay live until admin approves/rejects. Demo assumption: one
//      photographer per browser session.
//   2. BE-fetched admin settings (F-NEW-1) — authoritative for every
//      other photographer the admin reviews.
//   3. mock seed — last-ditch fallback while the BE fetch is in flight or
//      if it errored; empty post-Phase-5 cleanup so usually returns null.
//   4. null
export function useEffectivePhotographerSettings(
  userId: string,
): EffectivePhotographerSettings | null {
  const sessionUser = useAuthStore((s) => s.user);
  const liveSettings = usePhotographerSettingsStore();
  const submissions = useAdminUserStore((s) => s.submissions);

  const isSelf = !!sessionUser && sessionUser.id === userId;
  const isSubmission = !!submissions[userId];
  const useLocal = isSelf || isSubmission;

  // Always call the hook — pass null to skip the fetch when we already have
  // the data locally (session photographer's own row or a fresh submission
  // mirrored from /dashboard/settings).
  const { data: adminBeSettings } = useAdminPhotographerSettings(
    useLocal ? null : userId,
  );

  return useMemo<EffectivePhotographerSettings | null>(() => {
    if (useLocal)
      return liveToEffective(liveSettings, sessionUser?.avatarUrl ?? null);
    if (adminBeSettings) return beToEffective(adminBeSettings);
    const seed = getPhotographerSettingsSeed(userId);
    return seed ? seedToEffective(seed) : null;
  }, [userId, useLocal, liveSettings, adminBeSettings, sessionUser]);
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
