"use client";

import { useMemo } from "react";
import { useAdminUsers, EMPTY_ADMIN_USERS, useAdminUserDetail, useAdminPhotographerSettings } from "@/hooks/use-admin-users";
import type { AdminPhotographerSettingsResponse } from "@/lib/api-admin";
import type { CoverSource } from "@/lib/photographer-registry";
import type { BrandColor, PayoutAccount, PhotographerRegion, SocialLink } from "@/store/photographer-settings-store";
import type { WatermarkPreview } from "@/lib/admin-photographer-settings-seed";
import type { AdminUserRow, PhotographerSettingsSnapshot } from "@/lib/admin-user-registry";
import type { AdminStatusPillTone } from "@/components/admin/admin-status-pill";
import { ROUTES } from "@/lib/constants";

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
// Handles are 3-30 characters, so a 36-character user id never collides with one.
const USER_ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

/** The detail route takes a handle, or the user id of a photographer who has none yet. */
export function useAdminPhotographerView(handle: string) {
  const key = handle.trim().toLowerCase();
  const byId = USER_ID.test(key);
  const users = useAdminUsers();
  const row = byId ? undefined : (users.data ?? EMPTY_ADMIN_USERS).find((row) => row.handle?.toLowerCase() === key);
  const userId = byId ? key : row?.userId ?? null;
  const detail = useAdminUserDetail(userId);
  const settings = useAdminPhotographerSettings(userId);
  const effectiveSettings = useMemo(() => settings.data ? beToEffective(settings.data) : null, [settings.data]);
  return {
    loading: byId ? detail.isPending || settings.isPending
      : users.isPending || (!row && users.isFetching) || (!!row && (detail.isPending || settings.isPending)),
    error: (byId ? null : users.error) ?? detail.error ?? settings.error,
    refetch: () => Promise.all([...(byId ? [] : [users.refetch()]), ...(userId ? [detail.refetch(), settings.refetch()] : [])]),
    view: userId && detail.data ? {
      row: detail.data,
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

const COMPLETENESS_FIELDS: ReadonlyArray<{ key: keyof PhotographerSettingsSnapshot; label: string }> = [
  { key: "hasCover", label: "Cover banner" },
  { key: "hasBrandName", label: "Brand name" },
  { key: "hasWatermark", label: "Watermark" },
  { key: "hasHandle", label: "Public handle" },
  { key: "hasRegion", label: "Region" },
];

export function snapshotFromSettings(s: EffectivePhotographerSettings): PhotographerSettingsSnapshot {
  return {
    hasCover: !!s.cover,
    hasBrandName: s.brandName.trim().length > 0,
    hasWatermark: !!s.watermark,
    hasHandle: s.handle.trim().length >= 3,
    hasRegion: !!s.region,
    socialCount: s.socials.length,
    payoutCount: s.payouts.length,
  };
}

// The one completeness checklist: the queue, the directory card and the detail page all count these seven.
export function completenessItems(snapshot: PhotographerSettingsSnapshot): Array<{ label: string; filled: boolean; detail?: string }> {
  return [
    ...COMPLETENESS_FIELDS.map((f) => ({ label: f.label, filled: !!snapshot[f.key] })),
    { label: "Social profiles", filled: snapshot.socialCount > 0, detail: `${snapshot.socialCount} on file` },
    { label: "Payout accounts", filled: snapshot.payoutCount > 0, detail: `${snapshot.payoutCount} on file` },
  ];
}

export function computeCompleteness(snapshot: PhotographerSettingsSnapshot | null): { filled: number; total: number } {
  const total = COMPLETENESS_FIELDS.length + 2;
  return { filled: snapshot ? completenessItems(snapshot).filter((i) => i.filled).length : 0, total };
}

const STATUS_LABEL = { approved: "Approved", pending: "Pending", incomplete: "Incomplete" } as const;

// PH3: the one status look for the directory card and the detail page. Suspended wins; amber = waiting on an admin.
export function photographerStatus(row: AdminUserRow): { label: string; tone: AdminStatusPillTone } {
  if (row.suspendedAt !== null) return { label: "Suspended", tone: "ink" };
  const tone = { approved: "fresh", pending: "amber", incomplete: "slate" } as const;
  return { label: STATUS_LABEL[row.verificationStatus], tone: tone[row.verificationStatus] };
}

// PH1: a photographer with no handle yet is opened by user id.
export function photographerHref(row: Pick<AdminUserRow, "handle" | "userId">): string {
  return `${ROUTES.ADMIN_PHOTOGRAPHERS}/${row.handle ?? row.userId}`;
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
