"use client";

import { useEffect, useRef } from "react";
import {
  fetchBrand,
  fetchPayoutAccounts,
  fetchSocials,
} from "@/lib/api-photographer-settings";
import { useAuthStore } from "@/store/auth-store";
import {
  BRAND_COLOR_HEX,
  usePhotographerSettingsStore,
  type BrandColor,
} from "@/store/photographer-settings-store";
import { useUserMediaStore } from "@/store/user-media-store";

// Photographer-settings hydration, mounted in DashboardShell so every
// /dashboard/* page sees populated state on mount — not just /dashboard/settings.
//
// Without this hook, `resetUserScopedStores()` (lib/auth-reset.ts) leaves the
// photographer-settings + user-media stores empty on every fresh login. Symptoms:
// /dashboard/billing disabled the Request payout button (no primary payout in
// store), the focused share page copied a `quickpitik.com/your-handle/…`
// placeholder URL to clipboard, and the verification banner flashed "Add your
// profile picture" before usePhotographerVerificationSync resolved to "approved".
// See vault audit `tasks#-photographer-flow-audit--2026-05-27` P0-1.
//
// Three blocks hydrate independently (no shared Promise.all) so a null on one
// block doesn't abort the others. The settings page's Edit→Save state machine
// still owns writes; this hook is read-only into the store.
export function usePhotographerSettingsHydration(): void {
  const user = useAuthStore((s) => s.user);
  const isPhotographer = user?.role === "PHOTOGRAPHER";
  // Per-user ref so a logout → login within the same shell instance re-fires
  // hydration for the new user.
  const hydratedForRef = useRef<string | null>(null);

  useEffect(() => {
    if (!isPhotographer || !user) return;
    if (hydratedForRef.current === user.id) return;
    hydratedForRef.current = user.id;
    void hydrateAll();
  }, [isPhotographer, user]);
}

async function hydrateAll(): Promise<void> {
  const now = new Date().toISOString();

  // Brand block: brand name/color/bio + handle + region + presigned
  // cover/watermark/avatar URLs.
  void fetchBrand()
    .then((brand) => {
      if (!brand) return;
      const isKnownBrandColor = (v: string): v is BrandColor =>
        v in BRAND_COLOR_HEX;
      const validBrandColor: BrandColor = isKnownBrandColor(brand.brandColor)
        ? brand.brandColor
        : "none";
      const hasRegion =
        brand.regionCode.length > 0 && brand.provinceCode.length > 0;

      usePhotographerSettingsStore.setState({
        brandName: brand.brandName,
        brandColor: validBrandColor,
        bio: brand.bio,
        handle: brand.handle,
        region: hasRegion
          ? { regionCode: brand.regionCode, provinceCode: brand.provinceCode }
          : null,
        cover: brand.coverUrl
          ? { dataUrl: brand.coverUrl, uploadedAt: now }
          : null,
        watermark: brand.watermarkUrl
          ? { dataUrl: brand.watermarkUrl, uploadedAt: now }
          : null,
      });

      // Avatar lives in useUserMediaStore (shared with the runner account
      // flow). Same auth-transition wipe applies, so re-hydrate from the
      // same payload.
      if (brand.avatarUrl) {
        useUserMediaStore.getState().setAvatar({
          dataUrl: brand.avatarUrl,
          uploadedAt: now,
        });
      }
    })
    .catch((err) => {
      console.warn("[photographer/hydration] brand fetch failed", err);
    });

  void fetchSocials()
    .then((socials) => {
      if (!socials) return;
      usePhotographerSettingsStore.setState({ socials });
    })
    .catch((err) => {
      console.warn("[photographer/hydration] socials fetch failed", err);
    });

  void fetchPayoutAccounts()
    .then((payouts) => {
      if (!payouts) return;
      usePhotographerSettingsStore.setState({ payouts });
    })
    .catch((err) => {
      console.warn("[photographer/hydration] payouts fetch failed", err);
    });
}
