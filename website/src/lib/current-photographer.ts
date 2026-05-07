import type { User } from "@/types/user";

// Mock-only mapping from auth user → photographer mock-id used by:
//   • photographer-message-store filters (whose inbox is this?)
//   • admin-payout-report-store.submitReport (who's filing?)
//
// In the mock seed data (admin-payouts.ts, admin-payout-reports.ts), the
// held cycle + the seed report both belong to "photog-cebustride", so we
// resolve any logged-in photographer to that id — the demo flow then has
// continuous content (seed message visible, seed report visible, plus any
// runtime admin action on cebustride cycles lands in the same inbox).
//
// Phase F replaces this with `user.photographerId` once the backend wires
// the User type to the photographers table.

export const DEMO_PHOTOGRAPHER_ID = "photog-cebustride";
export const DEMO_PHOTOGRAPHER_NAME = "Cebu Stride";
export const DEMO_PHOTOGRAPHER_HANDLE = "cebustride";

export interface CurrentPhotographer {
  id: string;
  name: string;
  handle: string;
}

export function resolveCurrentPhotographer(
  user: User | null,
): CurrentPhotographer | null {
  if (!user || user.role !== "PHOTOGRAPHER") return null;
  return {
    id: DEMO_PHOTOGRAPHER_ID,
    name: DEMO_PHOTOGRAPHER_NAME,
    handle: DEMO_PHOTOGRAPHER_HANDLE,
  };
}
