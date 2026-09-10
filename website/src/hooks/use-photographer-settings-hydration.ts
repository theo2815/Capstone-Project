"use client";

import { captureAuthSession } from "@/lib/auth-session";
import { useEffect, useState } from "react";
import {
  fetchBrand,
  fetchPayoutAccounts,
  fetchSocials,
} from "@/lib/api-photographer-settings";
import { useAuthStore } from "@/store/auth-store";
import {
  BRAND_COLOR_HEX,
  DEFAULT_WATERMARK_PLACEMENT,
  usePhotographerSettingsStore,
  type BrandColor,
} from "@/store/photographer-settings-store";
import { useUserMediaStore } from "@/store/user-media-store";

// Transitional Settings hydration shared across dashboard remounts. Track each
// block independently so a failed payout read can retry without overwriting
// edits to an already-hydrated brand. Auth reset retires the entire run.
type Block = "brand" | "socials" | "payouts";
interface HydrationRun {
  isCurrent: () => boolean;
  completed: Set<Block>;
  request: Promise<void> | null;
  attempts: number;
}
let hydration: HydrationRun | null = null;
const MAX_ATTEMPTS = 3;
const RETRY_DELAY_MS = 2000;

export function resetPhotographerHydrationGuard(): void {
  hydration = null;
}

export function usePhotographerSettingsHydration(): void {
  const user = useAuthStore((s) => s.user);
  const sessionVersion = useAuthStore((s) => s.sessionVersion);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    if (user?.role !== "PHOTOGRAPHER") return;
    if (!hydration?.isCurrent()) {
      hydration = {
        isCurrent: captureAuthSession(),
        completed: new Set(),
        request: null,
        attempts: 0,
      };
    }
    const run = hydration;
    if (run.completed.size === 3 || run.attempts >= MAX_ATTEMPTS) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    if (!run.request) {
      run.attempts++;
      run.request = hydrateAll(run).finally(() => {
        run.request = null;
      });
    }
    void run.request.then(() => {
      if (
        cancelled || hydration !== run || !run.isCurrent() ||
        run.completed.size === 3 || run.attempts >= MAX_ATTEMPTS
      ) return;
      timer = setTimeout(() => {
        if (hydration === run && run.isCurrent()) setAttempt((a) => a + 1);
      }, RETRY_DELAY_MS);
    });
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [user?.id, user?.role, sessionVersion, attempt]);
}

async function hydrateAll(run: HydrationRun): Promise<void> {
  const isCurrent = () => hydration === run && run.isCurrent();
  async function block(name: Block, read: () => Promise<void>) {
    if (run.completed.has(name)) return;
    try {
      await read();
      if (isCurrent()) run.completed.add(name);
    } catch (error) {
      if (isCurrent()) console.warn(`[photographer/hydration] ${name} fetch failed`, error);
    }
  }
  const now = new Date().toISOString();

  // Brand block: brand name/color/bio + handle + region + presigned
  // cover/watermark/avatar URLs.
  const brandBlock = block("brand", async () => {
    const brand = await fetchBrand();
    if (!isCurrent() || !brand) return;
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
      watermarkPlacement: brand.watermarkSettings
        ? {
            opacity: brand.watermarkSettings.opacity,
            scale: brand.watermarkSettings.scale,
            landscape: {
              x: brand.watermarkSettings.landscapeX,
              y: brand.watermarkSettings.landscapeY,
            },
            portrait: {
              x: brand.watermarkSettings.portraitX,
              y: brand.watermarkSettings.portraitY,
            },
          }
        : DEFAULT_WATERMARK_PLACEMENT,
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

  const socialsBlock = block("socials", async () => {
    const socials = await fetchSocials();
    if (!isCurrent() || !socials) return;
    usePhotographerSettingsStore.setState({ socials });
  });

  const payoutsBlock = block("payouts", async () => {
    const payouts = await fetchPayoutAccounts();
    if (!isCurrent() || !payouts) return;
    usePhotographerSettingsStore.setState({ payouts });
  });

  await Promise.all([brandBlock, socialsBlock, payoutsBlock]);
}
