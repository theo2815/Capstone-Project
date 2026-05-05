export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8080/api/v1";

export const ROUTES = {
  HOME: "/",
  RUNNERS: "/runners",
  PHOTOGRAPHERS: "/photographers",
  LOGIN: "/login",
  REGISTER: "/register",
  FORGOT_PASSWORD: "/forgot-password",
  EVENTS: "/events",
  CART: "/cart",
  CHECKOUT: "/checkout",
  ORDERS: "/orders",
  PROFILE: "/profile",
  ACCOUNT: "/account",
  DASHBOARD: "/dashboard",
  DASHBOARD_EVENTS: "/dashboard/events",
  DASHBOARD_EARNINGS: "/dashboard/earnings",
  ADMIN: "/admin",
  ADMIN_EVENTS: "/admin/events",
  ADMIN_USERS: "/admin/users",
} as const;

export const ROLES = {
  ADMIN: "ADMIN",
  PHOTOGRAPHER: "PHOTOGRAPHER",
  RUNNER: "RUNNER",
} as const;

export const MAX_UPLOAD_SIZE = 10 * 1024 * 1024; // 10 MB
export const MAX_BATCH_UPLOAD = 50;
export const ACCEPTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"];
