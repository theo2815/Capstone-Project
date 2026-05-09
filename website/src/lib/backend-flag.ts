// Runtime gate for backend integration. The website FE is wired to call real
// Spring Boot endpoints; until the relevant phase ships, hooks fall back to
// the in-memory mock catalog / generators so dev keeps working.
//
// Phase A (auth) is always live — `<AuthHydrator>` and `useAuth` call
// `/auth/*` regardless of this flag. The flag gates Phase B/C/E/F/G domains.
//
// Set `NEXT_PUBLIC_BACKEND_LIVE=true` in `.env.local` to flip to live mode.
// Production deploys flip when the matching backend phase ships.
export const BACKEND_LIVE: boolean =
  process.env.NEXT_PUBLIC_BACKEND_LIVE === "true";
