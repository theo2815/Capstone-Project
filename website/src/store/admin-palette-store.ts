import { create } from "zustand";
import { persist } from "zustand/middleware";

// Singleton open/close + persisted recents for the admin command palette.
// Open state is ephemeral; `recents` (last picked entry IDs, max 5) is
// persisted to localStorage so the palette remembers your common
// destinations across sessions.
//
// Other admin-keyboard hooks read `open` to suppress themselves while the
// palette is on top, so the keystroke that closes the palette doesn't
// leak through to whatever surface is sitting underneath. Same pattern
// as admin-legend-store.

const RECENTS_MAX = 5;

interface AdminPaletteStoreState {
  open: boolean;
  recents: string[];
  setOpen: (open: boolean) => void;
  toggle: () => void;
  pushRecent: (id: string) => void;
  clearRecents: () => void;
}

export const useAdminPaletteStore = create<AdminPaletteStoreState>()(
  persist(
    (set) => ({
      open: false,
      recents: [],
      setOpen: (open) => set({ open }),
      toggle: () => set((s) => ({ open: !s.open })),
      pushRecent: (id) =>
        set((s) => {
          const next = [id, ...s.recents.filter((x) => x !== id)].slice(
            0,
            RECENTS_MAX,
          );
          return { recents: next };
        }),
      clearRecents: () => set({ recents: [] }),
    }),
    {
      name: "quickpitik-admin-palette",
      // Only persist recents; `open` is session state.
      partialize: (s) => ({ recents: s.recents }),
    },
  ),
);
