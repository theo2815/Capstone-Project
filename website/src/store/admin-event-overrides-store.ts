import { create } from "zustand";
import type { ListEvent } from "@/app/events/events-browser";
import { deriveEventState } from "@/lib/event-catalog";
import {
  createAdminEvent as apiCreateEvent,
  updateAdminEvent as apiUpdateEvent,
  deleteAdminEvent as apiDeleteEvent,
} from "@/lib/api-admin";

// Admin CRUD layer on top of the catalog. After Phase G, the backend is the
// source of truth for /admin/events listings (server-rendered) — this store
// only carries optimistic edits made in the current session:
//
//   - `submissions`: rows created in the current session, awaiting the next
//     page refetch. Each entry IS the backend response (real id, slug, and
//     presigned bannerUrl), so the View ↗ link doesn't 404.
//   - `overrides`:  per-id edit patches mirrored after a successful PATCH.
//   - `tombstones`: ids the admin deleted; merged out of the seed list.
//
// Consumers should read through `getCatalogWithOverrides()` /
// `useEventCatalog()` from `@/lib/event-catalog` — never call
// `useAdminEventOverridesStore()` directly inside non-admin pages.

function fireBackendEventAction(label: string, p: Promise<unknown>): void {
  void p.catch((err) => {
    console.error(`[admin/events] ${label} backend call failed`, err);
  });
}

export type EventOverridePatch = Partial<
  Pick<ListEvent, "name" | "date" | "location" | "city" | "status" | "state">
> & {
  /** Presigned URL for the cover banner. Pass `undefined` to clear an
   *  existing cover (the spread-merger reads it as "no banner"). */
  bannerUrl?: string;
  /** New admin-set per-photo price in PHP. Optional — only forwarded to the
   *  BE when it actually changed in the modal. */
  pricePerPhoto?: number;
  /** Organizer name / race-day notes for the "About this race" strip. Only
   *  forwarded to the BE when changed in the modal. */
  organizerName?: string;
  description?: string;
};

export interface CreateEventInput {
  name: string;
  date: string;
  location: string;
  /** Per-photo price in PHP. Admin sets it at event creation; the BE seeds
   *  every uploaded photo to this value via PhotoUploadService. */
  pricePerPhoto: number;
  /** Raw cover file picked from disk. Uploaded as multipart; the backend
   *  re-encodes to JPEG and returns the presigned URL on the response. */
  cover?: File | null;
  /** Organizer name / race-day notes for the "About this race" strip. */
  organizerName?: string;
  description?: string;
}

interface AdminEventOverridesState {
  overrides: Record<string, EventOverridePatch>;
  submissions: ReadonlyArray<ListEvent>;
  tombstones: ReadonlyArray<string>;
  /** @deprecated state is derived from date — kept for callers still in flight. */
  setEventState: (id: string, state: ListEvent["state"]) => void;
  /** Returns a promise that resolves on backend ack so the caller can
   *  invalidate the admin-events query and pick up the new presigned
   *  cover URL. Rejects with the backend's ApiError on failure. */
  editEvent: (
    id: string,
    patch: EventOverridePatch & {
      cover?: File | null;
      removeCover?: boolean;
    },
  ) => Promise<void>;
  /** Awaits the backend POST so the id/slug/bannerUrl in the optimistic row
   *  match the persisted record. Rejects on backend failure — callers should
   *  surface a toast and keep the form open. */
  createEvent: (input: CreateEventInput) => Promise<ListEvent>;
  deleteEvent: (id: string) => void;
  restoreEvent: (id: string) => void;
  clear: () => void;
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
    editEvent: (id, patch) => {
      // Strip transient cover/removeCover before merging into the local store
      // — those are upload signals, not row state. The new presigned bannerUrl
      // arrives via the post-PATCH refetch.
      const { cover, removeCover, ...localPatch } = patch;
      set((s) => {
        const subIdx = s.submissions.findIndex((e) => e.id === id);
        if (subIdx >= 0) {
          const next = s.submissions.slice();
          const merged = { ...next[subIdx], ...localPatch };
          if (localPatch.location && !localPatch.city) {
            merged.city = deriveCity(localPatch.location);
          }
          next[subIdx] = merged;
          return { submissions: next };
        }
        return {
          overrides: {
            ...s.overrides,
            [id]: { ...s.overrides[id], ...localPatch },
          },
        };
      });
      const subset = {
        ...(patch.name !== undefined ? { title: patch.name } : {}),
        ...(patch.date !== undefined ? { date: patch.date } : {}),
        ...(patch.location !== undefined ? { location: patch.location } : {}),
        ...(patch.pricePerPhoto !== undefined
          ? { pricePerPhoto: patch.pricePerPhoto }
          : {}),
        ...(patch.organizerName !== undefined
          ? { organizerName: patch.organizerName }
          : {}),
        ...(patch.description !== undefined
          ? { description: patch.description }
          : {}),
        ...(cover ? { cover } : {}),
        ...(removeCover ? { removeCover: true } : {}),
      };
      if (Object.keys(subset).length === 0) return Promise.resolve();
      return apiUpdateEvent(id, subset).then(() => undefined);
    },
    createEvent: async (input) => {
      // Backend is the source of truth for id/slug — local heuristics
      // produced a different slug from the server, so the admin card's
      // View ↗ link 404'd against /events/{backend-slug}. Awaiting the
      // POST means the optimistic row is the real row.
      const created = await apiCreateEvent({
        title: input.name.trim(),
        date: input.date,
        location: input.location.trim(),
        pricePerPhoto: input.pricePerPhoto,
        organizerName: input.organizerName,
        description: input.description,
        cover: input.cover ?? null,
      });
      const event: ListEvent = {
        ...created,
        city: deriveCity(created.location),
        state: deriveEventState(created.date),
      };
      set((s) => ({ submissions: [event, ...s.submissions] }));
      return event;
    },
    deleteEvent: (id) => {
      let needsBackendDelete = false;
      set((s) => {
        const subIdx = s.submissions.findIndex((e) => e.id === id);
        if (subIdx >= 0) {
          // Brand-new submission — drop it entirely, no tombstone needed.
          const next = s.submissions.slice();
          next.splice(subIdx, 1);
          // Submission was never persisted on the backend if create call
          // hadn't completed yet; live-mode safety: still issue DELETE so a
          // backend-side row created in flight gets cleaned up.
          needsBackendDelete = true;
          return { submissions: next };
        }
        if (s.tombstones.includes(id)) return {};
        needsBackendDelete = true;
        return { tombstones: [...s.tombstones, id] };
      });
      if (needsBackendDelete) {
        fireBackendEventAction("deleteEvent", apiDeleteEvent(id));
      }
    },
    restoreEvent: (id) =>
      set((s) => ({ tombstones: s.tombstones.filter((t) => t !== id) })),
    clear: () => set({ overrides: {}, submissions: [], tombstones: [] }),
  }),
);
