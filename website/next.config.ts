import type { NextConfig } from "next";

// Security headers (2026-08-28 audit). The backend API sets its own via
// Spring Security; these cover the Next-served pages, which previously
// shipped none — the guest share-token flow was leaning on the browser's
// *default* referrer policy to keep ?token= out of cross-origin Referers.

// connect-src must name the API + WS origins the app actually calls.
// Derived from the same env vars the client code reads (lib/constants.ts,
// lib/ws-url.ts), with their localhost fallbacks.
const API_ORIGIN = new URL(
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8080/api/v1",
).origin;
const WS_ORIGIN = (
  process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8080/ws"
).replace(/^(wss?:\/\/[^/]+).*$/, "$1");

// Production-only: dev needs HMR's eval + the dev overlay, and a CSP that
// only exists in prod is one that can't drift out of sync with dev tooling.
// script/style allow 'unsafe-inline' — Next's bootstrap and React's style={}
// attributes require it without nonce middleware (filed as a follow-up).
// img-src allows any https host: photo thumbnails are presigned R2 URLs on
// an account-specific domain the FE can't know at build time; images can't
// execute script, so the wildcard is the pragmatic trade.
const CSP = [
  "default-src 'self'",
  "script-src 'self' 'unsafe-inline'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob: https: http://localhost:8080",
  "font-src 'self' data:",
  `connect-src 'self' ${API_ORIGIN} ${WS_ORIGIN}`,
  "frame-ancestors 'none'",
  "object-src 'none'",
  "base-uri 'self'",
  "form-action 'self'",
].join("; ");

const securityHeaders = [
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "X-Frame-Options", value: "DENY" },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  {
    key: "Permissions-Policy",
    value: "camera=(), microphone=(), geolocation=()",
  },
  ...(process.env.NODE_ENV === "production"
    ? [{ key: "Content-Security-Policy", value: CSP }]
    : []),
];

const nextConfig: NextConfig = {
  async headers() {
    return [{ source: "/:path*", headers: securityHeaders }];
  },
};

export default nextConfig;
