export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8080/api/v1";

export const ROUTES = {
  HOME: "/",
  RUNNERS: "/runners",
  PHOTOGRAPHERS: "/photographers",
  LOGIN: "/login",
  REGISTER: "/register",
  FORGOT_PASSWORD: "/forgot-password",
  ONBOARDING: "/onboarding",
  EVENTS: "/events",
  CART: "/cart",
  CHECKOUT: "/checkout",
  ORDERS: "/orders",
  PROFILE: "/profile",
  ACCOUNT: "/account",
  DASHBOARD: "/dashboard",
  DASHBOARD_UPLOAD: "/dashboard/upload",
  DASHBOARD_EVENTS: "/dashboard/events",
  UPLOAD: "/upload",
  DASHBOARD_EARNINGS: "/dashboard/earnings",
  DASHBOARD_BILLING: "/dashboard/billing",
  DASHBOARD_SETTINGS: "/dashboard/settings",
  ADMIN: "/admin",
  ADMIN_INBOX: "/admin/inbox",
  ADMIN_OVERVIEW: "/admin/overview",
  ADMIN_EVENTS: "/admin/events",
  ADMIN_VERIFICATIONS: "/admin/verifications",
  ADMIN_DISPUTES: "/admin/disputes",
  ADMIN_PAYOUTS: "/admin/payouts",
  ADMIN_PHOTOGRAPHERS: "/admin/photographers",
  ADMIN_FLAGS: "/admin/flags",
  ADMIN_SALES: "/admin/sales",
} as const;

export const ROLES = {
  ADMIN: "ADMIN",
  PHOTOGRAPHER: "PHOTOGRAPHER",
  RUNNER: "RUNNER",
} as const;

export const MAX_UPLOAD_SIZE = 10 * 1024 * 1024; // 10 MB
export const MAX_BATCH_UPLOAD = 50;
export const ACCEPTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"];

// Admin Flags queue is hidden for v1 — out of scope per 2026-05-08 product
// decision (no content moderation in scope until post-launch). Flip to true
// to revive the rail row, KPI tile, inbox chip, palette entries, and the
// /admin/flags focus-mode route. Underlying store + queue body are kept
// intact so reviving needs no code change beyond this constant.
export const ADMIN_FLAGS_ENABLED: boolean = false;
