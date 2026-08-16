// Philippine regions + provinces — types and lookups.
//
// The DATA lives in the backend (`GET /api/v1/regions`, `RegionsService`), not
// here. This module used to carry a hardcoded ~220-line copy marked
// `TODO(backend)`; the mobile app carried a third copy of its own, so the same
// list existed in three places and a backend change reached neither client.
// Resolved 2026-08-16 — read the list with `useRegions()` and pass it to the
// lookups below.
//
// The types stay here because they describe the wire shape (identical to
// backend `RegionDto`/`ProvinceDto` and to the `ApiRegion` alias in
// `api-photographer-settings.ts`). `REGION_GROUP_LABEL` stays because it is UI
// copy, not data — the backend sends `group` as a bare key.

export type RegionGroup = "luzon" | "visayas" | "mindanao";

export interface PHProvince {
  /** kebab-case identifier, stable across renames */
  code: string;
  /** display name */
  name: string;
}

export interface PHRegion {
  /** kebab-case identifier */
  code: string;
  /** full display name, e.g. "Region IV-A (CALABARZON)" */
  name: string;
  /** short label, e.g. "CALABARZON" — used in compact contexts */
  shortName: string;
  group: RegionGroup;
  provinces: PHProvince[];
}

export const REGION_GROUP_LABEL: Record<RegionGroup, string> = {
  luzon: "Luzon",
  visayas: "Visayas",
  mindanao: "Mindanao",
};

// All three lookups take the region list explicitly rather than closing over a
// module-level array, so the caller decides where the data came from. Callers
// pass `useRegions().regions` — the backend list. Returning undefined/null
// while that list is still loading is the same "unknown code" path these
// helpers already had, so callers need no new branch.

export function getRegion(
  regions: readonly PHRegion[],
  code: string,
): PHRegion | undefined {
  return regions.find((r) => r.code === code);
}

export function getProvince(
  regions: readonly PHRegion[],
  regionCode: string,
  provinceCode: string,
): PHProvince | undefined {
  return getRegion(regions, regionCode)?.provinces.find(
    (p) => p.code === provinceCode,
  );
}

/** "Cebu · Region VII" — short two-tier label for compact display. */
export function formatRegionLabel(
  regions: readonly PHRegion[],
  regionCode: string,
  provinceCode: string,
): string | null {
  const region = getRegion(regions, regionCode);
  const province = region?.provinces.find((p) => p.code === provinceCode);
  if (!region || !province) return null;
  return `${province.name} · ${region.shortName}`;
}
