"use client";

import { useRegions } from "@/hooks/use-regions";
import { getProvince } from "@/lib/ph-regions";
import { usePhotographerSettingsStore } from "@/store/photographer-settings-store";

// The photographer's saved province name ("Cebu"), or null while the region
// list loads or when no region is set. Callers hide the label on null.
export function usePhotographerProvinceName(): string | null {
  const region = usePhotographerSettingsStore((s) => s.region);
  const { regions } = useRegions();
  if (!region) return null;
  return getProvince(regions, region.regionCode, region.provinceCode)?.name ?? null;
}
