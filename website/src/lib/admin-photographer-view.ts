"use client";

import { useMemo } from "react";
import { useAdminUsers, EMPTY_ADMIN_USERS, useAdminUserDetail, useAdminPhotographerSettings } from "@/hooks/use-admin-users";
import type { AdminPhotographerSettingsResponse } from "@/lib/api-admin";
import { getPhotographerByHandle, type CoverSource } from "@/lib/photographer-registry";
import type { BrandColor, PayoutAccount, PhotographerRegion, SocialLink } from "@/store/photographer-settings-store";
import type { WatermarkPreview } from "@/lib/admin-photographer-settings-seed";

export interface EffectivePhotographerSettings {
  brandName: string;
  brandColor: BrandColor;
  bio: string;
  handle: string;
  avatarUrl: string | null;
  cover: CoverSource | null;
  watermark: WatermarkPreview | null;
  region: PhotographerRegion | null;
  socials: SocialLink[];
  payouts: PayoutAccount[];
  source: "live";
}

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

// All private review data belongs to the authenticated admin's Query cache.
export function useAdminPhotographerView(handle: string) {
  const users = useAdminUsers();
  const row = (users.data ?? EMPTY_ADMIN_USERS).find((row) => row.handle?.toLowerCase() === handle.trim().toLowerCase());
  const detail = useAdminUserDetail(row?.userId ?? null);
  const settings = useAdminPhotographerSettings(row?.userId ?? null);
  const effectiveSettings = useMemo(() => settings.data ? beToEffective(settings.data) : null, [settings.data]);
  return {
    loading: users.isPending || (!row && users.isFetching) || (!!row && (detail.isPending || settings.isPending)),
    error: users.error ?? detail.error ?? settings.error,
    refetch: () => Promise.all([users.refetch(), ...(row ? [detail.refetch(), settings.refetch()] : [])]),
    view: row && detail.data ? {
      row: detail.data,
      profile: getPhotographerByHandle(handle),
      effectiveSettings,
      decisions: detail.data.decisionLog,
    } : null,
  };
}

export function useEffectivePhotographerSettings(userId: string) {
  const query = useAdminPhotographerSettings(userId);
  const data = useMemo(() => query.data ? beToEffective(query.data) : null, [query.data]);
  return { ...query, data };
}

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
