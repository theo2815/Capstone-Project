import { api } from "@/lib/api";

// Photographer coupons (V45) backend contract
//   POST /api/v1/coupons/preview   { code, photoIds[] } → CouponPreview   (public)
//
// The preview is priced by the same backend rule checkout charges — the
// discount is a percentage of the photographer's share, never of the list
// price — so this is display-only: the client never computes a discount.
// Errors arrive as ApiError with code COUPON_INVALID / COUPON_EXPIRED /
// COUPON_NOT_APPLICABLE and `field: "couponCode"`.

export interface CouponPreviewItem {
  photoId: string;
  price: number;
  discount: number;
}

// `items` holds eligible photos only; anything absent is not covered.
export interface CouponPreview {
  code: string;
  percentOff: number;
  photographerName: string | null;
  photographerHandle: string | null;
  items: CouponPreviewItem[];
  eligibleCount: number;
  discountTotal: number;
}

export async function postCouponPreview(args: {
  code: string;
  photoIds: string[];
}): Promise<CouponPreview> {
  return api.post<CouponPreview>("/coupons/preview", args);
}
