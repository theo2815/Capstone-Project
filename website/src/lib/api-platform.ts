import { api } from "@/lib/api";
import {
  COUPON_MAX_PERCENT,
  PHOTO_PRICE_PHP,
  PHOTOGRAPHER_KEEP_RATE,
  PLATFORM_CUT_RATE,
} from "@/lib/platform-economics";

// Phase F.2 platform-fees backend contract
//   GET /api/v1/platform/fees → { photoPricePhp, platformCutRate, photographerKeepRate, couponMaxPercent }
//
// PLATFORM_FEES_FALLBACK is the resilience default used when the backend
// call fails. The constants in platform-economics.ts match the backend's
// PlatformProperties defaults exactly.

export interface PlatformFees {
  photoPricePhp: number;
  platformCutRate: number;
  photographerKeepRate: number;
  couponMaxPercent: number;
}

export const PLATFORM_FEES_FALLBACK: PlatformFees = {
  photoPricePhp: PHOTO_PRICE_PHP,
  platformCutRate: PLATFORM_CUT_RATE,
  photographerKeepRate: PHOTOGRAPHER_KEEP_RATE,
  couponMaxPercent: COUPON_MAX_PERCENT,
};

export async function fetchPlatformFees(): Promise<PlatformFees | null> {
  try {
    return await api.get<PlatformFees>("/platform/fees");
  } catch {
    return null;
  }
}
