import { create } from "zustand";
import { persist } from "zustand/middleware";

// TODO(backend): swap localStorage persist for `/me/saved-events` endpoints
// once Spring Boot Phase B exposes them. Same pattern as user-media-store.

// Demo seed so a fresh user sees Race Log work immediately.
// - "u1" = Cebu Bay Run (upcoming)  → saved-only upcoming row
// - "1"  = Cebu Marathon (live)     → paired with a seeded order = saved + bought
// - "6"  = Cebu Night Run 2025      → saved-only past archive row
const SEED_SAVED_IDS: ReadonlyArray<string> = ["u1", "1", "6"];

interface SavedEventsState {
  ids: string[];
  save: (id: string) => void;
  unsave: (id: string) => void;
  toggle: (id: string) => void;
  clear: () => void;
}

export const useSavedEventsStore = create<SavedEventsState>()(
  persist(
    (set) => ({
      ids: [...SEED_SAVED_IDS],
      save: (id) =>
        set((state) =>
          state.ids.includes(id) ? state : { ids: [...state.ids, id] },
        ),
      unsave: (id) =>
        set((state) => ({ ids: state.ids.filter((x) => x !== id) })),
      toggle: (id) =>
        set((state) =>
          state.ids.includes(id)
            ? { ids: state.ids.filter((x) => x !== id) }
            : { ids: [...state.ids, id] },
        ),
      clear: () => set({ ids: [] }),
    }),
    { name: "quickpitik-saved-events" },
  ),
);
