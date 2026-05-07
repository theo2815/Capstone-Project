// Platform economics — single source of truth for the placeholder
// per-photo price and platform cut. Until backend Phase F lands a real
// /api/platform/fees endpoint, every surface that needs to reason about
// money (the photographer-facing PlatformCutModal, the admin sales page)
// imports from here so the same numbers move together.

export const PHOTO_PRICE_PHP = 125;
export const PLATFORM_CUT_RATE = 0.25;
export const PHOTOGRAPHER_KEEP_RATE = 1 - PLATFORM_CUT_RATE;

export const PER_PHOTO_PLATFORM_CUT = PHOTO_PRICE_PHP * PLATFORM_CUT_RATE;
export const PER_PHOTO_PHOTOGRAPHER_KEEP =
  PHOTO_PRICE_PHP * PHOTOGRAPHER_KEEP_RATE;

export interface GmvSplit {
  readonly gross: number;
  readonly cut: number;
  readonly keep: number;
}

export function splitGmv(gross: number): GmvSplit {
  return {
    gross,
    cut: gross * PLATFORM_CUT_RATE,
    keep: gross * PHOTOGRAPHER_KEEP_RATE,
  };
}
