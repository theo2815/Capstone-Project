"use client";

import { useEffect, useRef } from "react";
import {
  usePhotographerSettingsQueries,
} from "@/hooks/use-photographer-settings";
import {
  BRAND_COLOR_HEX,
  DEFAULT_WATERMARK_PLACEMENT,
  usePhotographerSettingsStore,
  type BrandColor,
} from "@/store/photographer-settings-store";
import { useUserMediaStore } from "@/store/user-media-store";

/**
 * Settings server data is cached by Query; Zustand remains the editable draft
 * bridge used by the existing Settings slabs and dashboard readiness consumers.
 * The bridge is populated only from successful query results and is reset on
 * auth transitions.
 */
export function resetPhotographerHydrationGuard(): void {
  // Query keys include the auth session version, so a logout/login cannot
  // reuse old results. Keep this compatibility export for auth-reset callers.
}

export function usePhotographerSettingsHydration(): void {
  const { session, brand, socials, payouts } = usePhotographerSettingsQueries();
  const sessionVersion = session.key[2];
  const draftEditing = usePhotographerSettingsStore((s) => s.draftEditing);
  const applied = useRef({
    version: null as number | null,
    brand: undefined as typeof brand.data,
    socials: undefined as typeof socials.data,
    payouts: undefined as typeof payouts.data,
  });

  useEffect(() => {
    if (
      !session.enabled || draftEditing ||
      !brand.isSuccess || !socials.isSuccess || !payouts.isSuccess ||
      brand.data == null || socials.data === undefined || payouts.data === undefined
    ) return;
    if (
      applied.current.version === sessionVersion &&
      applied.current.brand === brand.data &&
      applied.current.socials === socials.data &&
      applied.current.payouts === payouts.data
    ) return;
    applied.current = {
      version: sessionVersion,
      brand: brand.data,
      socials: socials.data,
      payouts: payouts.data,
    };
    const value = brand.data;
    const isKnownBrandColor = (v: string): v is BrandColor =>
      v in BRAND_COLOR_HEX;
    const validBrandColor: BrandColor = isKnownBrandColor(value.brandColor)
      ? value.brandColor
      : "none";
    const hasRegion =
      value.regionCode.length > 0 && value.provinceCode.length > 0;
    const now = new Date().toISOString();

    usePhotographerSettingsStore.setState({
      brandName: value.brandName,
      brandColor: validBrandColor,
      bio: value.bio,
      handle: value.handle,
      region: hasRegion
        ? { regionCode: value.regionCode, provinceCode: value.provinceCode }
        : null,
      cover: value.coverUrl ? { dataUrl: value.coverUrl, uploadedAt: now } : null,
      watermark: value.watermarkUrl
        ? { dataUrl: value.watermarkUrl, uploadedAt: now }
        : null,
      watermarkPlacement: value.watermarkSettings
        ? {
            opacity: value.watermarkSettings.opacity,
            scale: value.watermarkSettings.scale,
            landscape: {
              x: value.watermarkSettings.landscapeX,
              y: value.watermarkSettings.landscapeY,
            },
            portrait: {
              x: value.watermarkSettings.portraitX,
              y: value.watermarkSettings.portraitY,
            },
          }
        : DEFAULT_WATERMARK_PLACEMENT,
      socials: socials.data,
      payouts: payouts.data,
      hydratedSessionVersion: sessionVersion,
    });

    if (value.avatarUrl) {
      useUserMediaStore.getState().setAvatar({
        dataUrl: value.avatarUrl,
        uploadedAt: now,
      });
    }
  }, [
    brand.data,
    brand.isSuccess,
    draftEditing,
    payouts.data,
    payouts.isSuccess,
    session.enabled,
    sessionVersion,
    socials.data,
    socials.isSuccess,
  ]);
}
