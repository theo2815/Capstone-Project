import { create } from "zustand";
import type { ListEvent } from "@/app/events/events-browser";

// Mock-only Zustand store backing admin CRUD on the EVENT_CATALOG. NOT
// persisted — overrides reset on page refresh. The real backend is the
// source of truth once Phase F lands; this store exists only so the
// admin demo loop (`/admin/events` create/edit/delete → photographer's
// `/dashboard/upload` reflects it) works end-to-end without backend.
//
// Three layers:
//   - `submissions`: brand-new events created in the admin UI.
//   - `overrides`:  per-id edit patches over the seed catalog.
//   - `tombstones`: seed IDs that admin deleted.
//
// Consumers should read through `getCatalogWithOverrides()` /
// `useEventCatalog()` from `@/lib/event-catalog` — never call
// `useAdminEventOverridesStore()` directly inside non-admin pages.

export type EventOverridePatch = Partial<
  Pick<ListEvent, "name" | "date" | "location" | "city" | "status" | "state">
> & {
  /** Data URL for the cover banner. Pass `undefined` to clear an existing
   *  cover (the spread-merger reads it as "no banner"). */
  bannerUrl?: string;
};

export interface CreateEventInput {
  name: string;
  date: string;
  location: string;
  bannerUrl?: string;
}

interface AdminEventOverridesState {
  overrides: Record<string, EventOverridePatch>;
  submissions: ReadonlyArray<ListEvent>;
  tombstones: ReadonlyArray<string>;
  /** @deprecated state is derived from date — kept for callers still in flight. */
  setEventState: (id: string, state: ListEvent["state"]) => void;
  editEvent: (id: string, patch: EventOverridePatch) => void;
  createEvent: (input: CreateEventInput) => string;
  deleteEvent: (id: string) => void;
  restoreEvent: (id: string) => void;
  clear: () => void;
}

function slugify(name: string): string {
  return name
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9\s-]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-");
}

// Pull a coarse city from the location field. "IT Park, Cebu City" → "Cebu City".
// Falls back to the full location when there's no comma.
function deriveCity(location: string): string {
  const parts = location.split(",").map((p) => p.trim()).filter(Boolean);
  return parts[parts.length - 1] ?? location;
}

export const useAdminEventOverridesStore = create<AdminEventOverridesState>(
  (set) => ({
    overrides: {},
    submissions: [],
    tombstones: [],
    setEventState: (id, state) =>
      set((s) => ({
        overrides: {
          ...s.overrides,
          [id]: { ...s.overrides[id], state },
        },
      })),
    editEvent: (id, patch) =>
      set((s) => {
        // Patch may target a submission or a seed event. Try submissions first.
        const subIdx = s.submissions.findIndex((e) => e.id === id);
        if (subIdx >= 0) {
          const next = s.submissions.slice();
          const merged = { ...next[subIdx], ...patch };
          if (patch.location && !patch.city) {
            merged.city = deriveCity(patch.location);
          }
          next[subIdx] = merged;
          return { submissions: next };
        }
        return {
          overrides: {
            ...s.overrides,
            [id]: { ...s.overrides[id], ...patch },
          },
        };
      }),
    createEvent: (input) => {
      const id = `evt-${Date.now().toString(36)}`;
      const slugBase = slugify(input.name) || "event";
      const slug = `${slugBase}-${id.slice(-4)}`;
      const event: ListEvent = {
        id,
        slug,
        name: input.name.trim(),
        date: input.date,
        location: input.location.trim(),
        city: deriveCity(input.location.trim()),
        photoCount: 0,
        participantCount: 0,
        status: "ACTIVE",
        // Catalog mergers recompute state from date, but we set a sensible
        // default here so direct readers see the right pill.
        state: "upcoming",
        bannerUrl: input.bannerUrl,
      };
      set((s) => ({ submissions: [event, ...s.submissions] }));
      return id;
    },
    deleteEvent: (id) =>
      set((s) => {
        const subIdx = s.submissions.findIndex((e) => e.id === id);
        if (subIdx >= 0) {
          // Brand-new submission — drop it entirely, no tombstone needed.
          const next = s.submissions.slice();
          next.splice(subIdx, 1);
          return { submissions: next };
        }
        if (s.tombstones.includes(id)) return {};
        return { tombstones: [...s.tombstones, id] };
      }),
    restoreEvent: (id) =>
      set((s) => ({ tombstones: s.tombstones.filter((t) => t !== id) })),
    clear: () => set({ overrides: {}, submissions: [], tombstones: [] }),
  }),
);
