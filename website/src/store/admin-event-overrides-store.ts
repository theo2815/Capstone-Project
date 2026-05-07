import { create } from "zustand";
import type { ListEvent } from "@/app/events/events-browser";

// Mock-only Zustand store holding admin edits to the EVENT_CATALOG. NOT
// persisted — overrides reset on page refresh. The real backend is the
// source of truth once Phase F lands; this store exists only so the
// admin demo loop (`/admin/events` flips state → photographer's
// `/dashboard/upload` reflects it) works end-to-end without backend.
//
// Consumers should read through `getCatalogWithOverrides()` /
// `useEventCatalog()` from `@/lib/event-catalog` — never call
// `useAdminEventOverridesStore()` directly inside non-admin pages.

export type EventOverridePatch = Partial<
  Pick<ListEvent, "state" | "name" | "date" | "location" | "city" | "status">
>;

interface AdminEventOverridesState {
  overrides: Record<string, EventOverridePatch>;
  setEventState: (id: string, state: ListEvent["state"]) => void;
  editEvent: (id: string, patch: EventOverridePatch) => void;
  clear: () => void;
}

export const useAdminEventOverridesStore = create<AdminEventOverridesState>(
  (set) => ({
    overrides: {},
    setEventState: (id, state) =>
      set((s) => ({
        overrides: {
          ...s.overrides,
          [id]: { ...s.overrides[id], state },
        },
      })),
    editEvent: (id, patch) =>
      set((s) => ({
        overrides: {
          ...s.overrides,
          [id]: { ...s.overrides[id], ...patch },
        },
      })),
    clear: () => set({ overrides: {} }),
  }),
);
