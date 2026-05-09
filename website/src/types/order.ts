export type OrderStatus = "PENDING" | "PAID" | "FULFILLED" | "REFUNDED";

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
  paymentMethod: string;
  createdAt: string;
}

export interface OrderItem {
  photoId: string;
  downloadUrl?: string;
  price: number;
}

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
}
