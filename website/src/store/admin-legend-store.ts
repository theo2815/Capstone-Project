import { create } from "zustand";

// Singleton open/close for the admin keyboard-shortcut legend. Module-level
// so the global `?` shortcut, the bottom-right hint pill, and the drawer/
// queue keyboard hooks all share one source of truth — pressing `?` from
// anywhere on /admin/* toggles the same modal.
//
// Other admin-keyboard hooks (queue nav, drawer verbs, slash-to-search)
// read `open` to suppress themselves while the legend is on top, so the
// keystroke that closes the legend doesn't leak through to whatever surface
// is sitting underneath.

interface AdminLegendStoreState {
  open: boolean;
  setOpen: (open: boolean) => void;
  toggle: () => void;
}

export const useAdminLegendStore = create<AdminLegendStoreState>((set) => ({
  open: false,
  setOpen: (open) => set({ open }),
  toggle: () => set((s) => ({ open: !s.open })),
}));
