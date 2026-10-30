import type { DisputeReason, DisputeResolution } from "@/lib/admin-disputes";

export type OrderStatus =
  | "PENDING"
  | "PAID"
  | "FULFILLED"
  | "REFUNDED"
  | "EXPIRED";

export interface CartItem {
  photoId: string;
  eventId: string;
  thumbnailUrl: string;
  price: number;
  bib?: string | null;
  eventName?: string;
  eventSlug?: string;
  tone?: number;
  time?: string;
}

export interface Order {
  id: string;
  status: OrderStatus;
  items: OrderItem[];
  totalAmount: number;
  checkoutTotalAmount: number;
  paymentMethod: string;
  createdAt: string;
  // PayMongo hosted-checkout URL. Present for newly-created PENDING orders
  // and for idempotent replays of PENDING orders. Null when the order is
  // already PAID (success path on replay) or when the legacy stub flow ran.
  redirectUrl?: string | null;
  // Photographer coupon entered at checkout (V45), if any.
  couponCode?: string | null;
  qrPh?: {
    imageUrl: string;
    expiresAt: string;
    returnToken?: string | null;
    // PayMongo's test-mode simulator for this QR. Null on live keys — the
    // backend never emits it in production.
    testUrl?: string | null;
  } | null;
  // Guest return capability for an order that is already settled when
  // POST /orders answers (₱0 checkout, replay of a paid key). Null for
  // signed-in runners and for pending orders (theirs rides in qrPh).
  returnToken?: string | null;
}

// `price` is the list price; `discount` the coupon's share of it (0 or absent
// when none). `totalAmount` on the order is what was charged.
export interface OrderItem {
  photoId: string;
  downloadUrl?: string;
  price: number;
  discount?: number;
  availability?: OrderPhotoAvailability;
  unavailableReason?: string | null;
}

// REMOVAL_REVIEW keeps a paid buyer's access; REMOVED ("PHOTO_REMOVED") drops
// the image and download but never the receipt, amounts or refund access.
export type OrderPhotoAvailability = "AVAILABLE" | "REMOVAL_REVIEW" | "REMOVED";

// Phase E hydration shape for /me/orders/{id}.photos[]. Server bakes the
// presigned URLs (Q-005); mock fallback leaves URL fields undefined so
// PhotoStrip + lightbox fall back to ID-chip rendering.
export interface OrderPhotoDetail {
  id: string;
  bib: string | null;
  time: string;
  tone: number;
  thumbnailUrl?: string;
  previewUrl?: string;
  downloadUrl?: string;
  // api-orders defaults these for payloads from before the field existed.
  availability: OrderPhotoAvailability;
  unavailableReason: string | null;
}

// Backend order-history row, distinct from the checkout creation response.
export interface OrderSummary {
  id: string;
  eventId: string;
  photoIds: string[];
  total: number;
  paymentMethod: string;
  paidAt: string;
  eventName?: string;
  eventSlug?: string;
  eventDate?: string;
  eventState?: "upcoming" | "live" | "open" | "past";
  status?: OrderStatus;
  couponCode?: string | null;
  discountTotal?: number;
  disputes?: RunnerDispute[];
}

// Embedded on every OrderSummary / OrderDetail payload from /me/orders. Carries
// what the /orders surface needs to render the status chip, timeline, cancel
// button, and admin's resolution note — without a separate per-order fetch.
// Mirrors the BE RunnerDisputeDto.
export type RunnerDisputeStatus =
  | "open"
  | "resolved"
  | "denied"
  | "escalated"
  | "withdrawn";

export interface RunnerDispute {
  id: string;
  photoId: string;
  reason: DisputeReason;
  note: string;
  status: RunnerDisputeStatus;
  resolution: DisputeResolution | null;
  refundAmount: number | null;
  // Most-recent admin_decision_log.reason for this dispute (resolve / deny /
  // escalate). Null when admin hasn't engaged yet.
  resolutionNote: string | null;
  openedAt: string;
  resolvedAt: string | null;
  withdrawnAt: string | null;
  // Manual QR Ph refund: "manual_review" until an admin marks it refunded; the link
  // is set once the admin sends it. Older responses omit these.
  refundStatus?: string | null;
  refundLinkUrl?: string | null;
  refundLinkSentAt?: string | null;
  refundLinkExpiresAt?: string | null;
}
