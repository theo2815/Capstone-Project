import { api } from "@/lib/api";
import { BACKEND_LIVE } from "@/lib/backend-flag";
import {
  PHOTO_PRICE_PHP,
  PHOTOGRAPHER_KEEP_RATE,
  PLATFORM_CUT_RATE,
} from "@/lib/platform-economics";

// Phase F.2 platform-fees backend contract — see photographer-earnings.md
//
//   GET /api/v1/platform/fees → { photoPricePhp, platformCutRate, photographerKeepRate }
//
// On `BACKEND_LIVE=false` the lib/platform-economics.ts constants stand in.

export interface PlatformFees {
  photoPricePhp: number;
  platformCutRate: number;
  photographerKeepRate: number;
}

export const PLATFORM_FEES_FALLBACK: PlatformFees = {
  photoPricePhp: PHOTO_PRICE_PHP,
  platformCutRate: PLATFORM_CUT_RATE,
  photographerKeepRate: PHOTOGRAPHER_KEEP_RATE,
};

export async function fetchPlatformFees(): Promise<PlatformFees | null> {
  if (!BACKEND_LIVE) return null;
  return api.get<PlatformFees>("/platform/fees");
}
