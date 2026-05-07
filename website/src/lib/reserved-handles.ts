// Reserved handles — photographer public URLs are flat-namespace
// (`quickpitik.com/{handle}`), so the handle MUST NOT collide with any app
// route. This list covers existing routes plus likely future ones (settings,
// admin tooling, marketing pages).
//
// When adding a new top-level route, append its slug here so existing
// photographers don't get squatted on by the new route at runtime.

export const RESERVED_HANDLES: ReadonlySet<string> = new Set([
  // Existing app routes
  "home",
  "profile",
  "account",
  "dashboard",
  "events",
  "login",
  "register",
  "forgot-password",
  "onboarding",
  "cart",
  "checkout",
  "orders",
  "runners",
  "photographers",
  "admin",
  "wireframes",
  // Dashboard sub-routes (also reserved at root since handle is flat)
  "settings",
  "billing",
  "earnings",
  "upload",
  "uploads",
  // Future routes / common conventions
  "search",
  "find",
  "support",
  "help",
  "about",
  "contact",
  "terms",
  "privacy",
  "blog",
  "api",
  "_next",
  "static",
  "public",
  "by",
  "p",
  "u",
  "user",
  "users",
  "photographer",
  "auth",
  "oauth",
  "callback",
  // QuickPitik brand reservations
  "quickpitik",
  "qp",
  "official",
  "team",
  "staff",
  "moderator",
  "mod",
  "verified",
  // Footguns
  "null",
  "undefined",
  "true",
  "false",
  "new",
  "edit",
  "delete",
  "create",
]);

export function isReservedHandle(handle: string): boolean {
  return RESERVED_HANDLES.has(handle.trim().toLowerCase());
}

// Photographer handles: lowercase letters, digits, single dashes between
// alphanumerics. 3–32 characters. No leading/trailing dash, no consecutive
// dashes. Mirrors GitHub-style validation.
const HANDLE_PATTERN = /^[a-z0-9](?:[a-z0-9]|-(?=[a-z0-9])){2,31}$/;

export function isValidHandleFormat(handle: string): boolean {
  return HANDLE_PATTERN.test(handle.trim().toLowerCase());
}

// Single check: format-valid AND not reserved. Returns null if OK, or a
// human-readable error string explaining why the handle was rejected.
export function validateHandle(handle: string): string | null {
  const cleaned = handle.trim().toLowerCase();
  if (cleaned.length === 0) return "Handle is required.";
  if (cleaned.length < 3) return "Handle must be at least 3 characters.";
  if (cleaned.length > 32) return "Handle is limited to 32 characters.";
  if (!isValidHandleFormat(cleaned)) {
    return "Use lowercase letters, numbers, and dashes. No spaces or symbols.";
  }
  if (isReservedHandle(cleaned)) {
    return "That handle is reserved. Try another.";
  }
  return null;
}
