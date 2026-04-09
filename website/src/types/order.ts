export type OrderStatus = "PENDING" | "PAID" | "FULFILLED" | "REFUNDED";

export interface CartItem {
  photoId: string;
  eventId: string;
  thumbnailUrl: string;
  price: number;
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
