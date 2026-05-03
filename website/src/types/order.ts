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
