// Runtime gate for backend integration. The website FE is wired to call real
// Spring Boot endpoints by default; the in-memory mock catalog / generators
// remain in place as an explicit opt-out fallback for working without a
// running backend.
//
// Phase A (auth) is always live — `<AuthHydrator>` and `useAuth` call
// `/auth/*` regardless of this flag. The flag gates Phase B/C/E/F/G domains.
//
// Default (unset or any non-"false" value): live mode — calls real backend.
// Set `NEXT_PUBLIC_BACKEND_LIVE=false` in `.env.local` to fall back to mocks.
export const BACKEND_LIVE: boolean =
  process.env.NEXT_PUBLIC_BACKEND_LIVE !== "false";
