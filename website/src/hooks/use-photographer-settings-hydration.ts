"use client";

import { useEffect, useRef, useState } from "react";
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
  // Retry trigger. DashboardShell is mounted from app/dashboard/layout.tsx,
  // an App Router layout — it persists across /dashboard/* navigation, so a
  // remount-based retry would never fire. Bumping state is the only way to
  // re-run the effect, and without it one bad network moment leaves every
  // dashboard page reading an empty store for the rest of the session.
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    if (!isPhotographer || !user) return;
    if (hydratedForRef.current === user.id) return;
    hydratedForRef.current = user.id;
    void hydrateAll().then((ok) => {
      if (ok || attempt >= MAX_HYDRATION_RETRIES) return;
      hydratedForRef.current = null;
      setTimeout(() => setAttempt((a) => a + 1), HYDRATION_RETRY_MS);
    });
  }, [isPhotographer, user, attempt]);
}

const MAX_HYDRATION_RETRIES = 2;
const HYDRATION_RETRY_MS = 2000;

// Resolves true when at least one block populated the store. All three
// failing means the photographer is looking at empty slabs, which is what
// the caller retries on.
async function hydrateAll(): Promise<boolean> {
  const now = new Date().toISOString();

  // Brand block: brand name/color/bio + handle + region + presigned
  // cover/watermark/avatar URLs.
  const brandBlock = fetchBrand()
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
    });

  const socialsBlock = fetchSocials().then((socials) => {
    if (!socials) return;
    usePhotographerSettingsStore.setState({ socials });
  });

  const payoutsBlock = fetchPayoutAccounts().then((payouts) => {
    if (!payouts) return;
    usePhotographerSettingsStore.setState({ payouts });
  });

  // allSettled keeps the blocks independent — one rejection can't abort the
  // others, which is the property the 2026-05-27 P0-1 hoist introduced.
  const results = await Promise.allSettled([
    brandBlock,
    socialsBlock,
    payoutsBlock,
  ]);
  const labels = ["brand", "socials", "payouts"];
  results.forEach((r, i) => {
    if (r.status === "rejected") {
      console.warn(
        `[photographer/hydration] ${labels[i]} fetch failed`,
        r.reason,
      );
    }
  });

  // A block that resolves null counts as success — the backend answered and
  // the account genuinely has nothing there yet. Only a rejection is worth
  // retrying.
  return results.some((r) => r.status === "fulfilled");
}
