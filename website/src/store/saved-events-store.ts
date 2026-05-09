import { create } from "zustand";
import { persist } from "zustand/middleware";
import { postSaveEvent, postUnsaveEvent } from "@/lib/api-saved-events";

// `localStorage` persists the offline-buffer mode for guests; on auth, the
// AuthHydrator merge replaces the buffer with the canonical server list and
// flips `syncEnabled` so subsequent toggles mirror to /me/saved-events.
//
// Demo seed so a fresh user sees Race Log work immediately.
// - "u1" = Cebu Bay Run (upcoming)  → saved-only upcoming row
// - "1"  = Cebu Marathon (live)     → paired with a seeded order = saved + bought
// - "6"  = Cebu Night Run 2025      → saved-only past archive row
const SEED_SAVED_IDS: ReadonlyArray<string> = ["u1", "1", "6"];

interface SavedEventsState {
  ids: string[];
  // When true, mutators mirror to /me/saved-events. Set by `<AuthHydrator>`
  // after the post-login merge resolves; cleared on logout.
  syncEnabled: boolean;
  setSyncEnabled: (enabled: boolean) => void;
  setIds: (ids: string[]) => void;
  save: (id: string) => void;
  unsave: (id: string) => void;
  toggle: (id: string) => void;
  clear: () => void;
}

export const useSavedEventsStore = create<SavedEventsState>()(
  persist(
    (set, get) => ({
      ids: [...SEED_SAVED_IDS],
      syncEnabled: false,
      setSyncEnabled: (syncEnabled) => set({ syncEnabled }),
      setIds: (ids) => set({ ids }),
      save: (id) => {
        let added = false;
        set((state) => {
          if (state.ids.includes(id)) return state;
          added = true;
          return { ids: [...state.ids, id] };
        });
        if (added && get().syncEnabled) {
          postSaveEvent(id).catch(() => {
            set((state) => ({ ids: state.ids.filter((x) => x !== id) }));
          });
        }
      },
      unsave: (id) => {
        let removed = false;
        set((state) => {
          if (!state.ids.includes(id)) return state;
          removed = true;
          return { ids: state.ids.filter((x) => x !== id) };
        });
        if (removed && get().syncEnabled) {
          postUnsaveEvent(id).catch(() => {
            set((state) =>
              state.ids.includes(id) ? state : { ids: [...state.ids, id] },
            );
          });
        }
      },
      toggle: (id) => {
        const has = get().ids.includes(id);
        if (has) get().unsave(id);
        else get().save(id);
      },
      clear: () => set({ ids: [] }),
    }),
    {
      name: "quickpitik-saved-events",
      partialize: (state) => ({ ids: state.ids }),
    },
  ),
);
