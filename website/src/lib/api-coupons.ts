import { api } from "@/lib/api";

export interface EventCoupon {
  eventId: string;
  code: string;
  percentOff: number;
  active: boolean;
  expiresAt: string | null;
  usageLimit: number | null;
  usageCount: number;
  updatedAt: string;
}

export type EventCouponInput = Omit<
  EventCoupon,
  "eventId" | "usageCount" | "updatedAt"
>;

const eventCouponPath = (eventId: string) =>
  `/me/photographer/events/${encodeURIComponent(eventId)}/coupon`;

export async function fetchEventCoupon(
  eventId: string,
): Promise<EventCoupon | null> {
  return api.get<EventCoupon | null>(eventCouponPath(eventId));
}

export async function putEventCoupon(
  eventId: string,
  coupon: EventCouponInput,
): Promise<EventCoupon> {
  return api.put<EventCoupon>(eventCouponPath(eventId), coupon);
}

export async function deleteEventCoupon(eventId: string): Promise<void> {
  await api.delete<unknown>(eventCouponPath(eventId));
}

// Event coupon preview contract
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
