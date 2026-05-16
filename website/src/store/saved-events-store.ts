import { create } from "zustand";
import { persist } from "zustand/middleware";
import { postSaveEvent, postUnsaveEvent } from "@/lib/api-saved-events";

// `localStorage` persists the offline-buffer mode for guests; on auth, the
// AuthHydrator merge replaces the buffer with the canonical server list and
// flips `syncEnabled` so subsequent toggles mirror to /me/saved-events.

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

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
      ids: [],
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
      // Bump on 2026-05-16 to evict the pre-removal demo seed
      // ("u1", "1", "6") from existing browsers — backend rejects non-UUIDs
      // on /me/saved-events/merge with a 400.
      version: 1,
      migrate: (persisted) => {
        if (!persisted || typeof persisted !== "object") return { ids: [] };
        const ids = (persisted as { ids?: unknown }).ids;
        return {
          ids: Array.isArray(ids)
            ? ids.filter(
                (x): x is string => typeof x === "string" && UUID_RE.test(x),
              )
            : [],
        };
      },
    },
  ),
);
