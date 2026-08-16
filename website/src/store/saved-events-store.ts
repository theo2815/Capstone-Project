import { create } from "zustand";
import { persist } from "zustand/middleware";
import {
  postSaveEvent,
  postUnsaveEvent,
  type SavedEventSummary,
} from "@/lib/api-saved-events";

// `localStorage` persists IDs only — they survive refresh so the bookmark UI
// stays consistent. Summaries are rehydrated from BE on auth (AuthHydrator's
// merge call). Guest buffer mode persists IDs without summaries; on sign-in
// the merge response provides the full summaries the race log needs.
//
// `summaries` is the source of truth for /profile race log. `ids` is kept in
// lockstep so the SaveButton's isSaved lookup stays O(1) and survives the
// initial render before summaries hydrate.

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

interface SavedEventsState {
  ids: string[];
  summaries: SavedEventSummary[];
  syncEnabled: boolean;
  setSyncEnabled: (enabled: boolean) => void;
  setIds: (ids: string[]) => void;
  setSummaries: (summaries: SavedEventSummary[]) => void;
  save: (eventId: string, optimistic?: SavedEventSummary) => void;
  unsave: (eventId: string) => void;
  toggle: (eventId: string, optimistic?: SavedEventSummary) => void;
  clear: () => void;
}

export const useSavedEventsStore = create<SavedEventsState>()(
  persist(
    (set, get) => ({
      ids: [],
      summaries: [],
      syncEnabled: false,
      setSyncEnabled: (syncEnabled) => set({ syncEnabled }),
      // Ids without summaries — the guest-buffer shape. setSummaries() can't
      // stand in for this: a guest's saved ids have no BE summary yet, which
      // is the whole reason the merge exists.
      setIds: (ids) => set({ ids }),
      setSummaries: (summaries) =>
        set({ summaries, ids: summaries.map((s) => s.id) }),
      save: (eventId, optimistic) => {
        let added = false;
        set((state) => {
          if (state.ids.includes(eventId)) return state;
          added = true;
          const nextIds = [...state.ids, eventId];
          const nextSummaries = optimistic
            ? [optimistic, ...state.summaries]
            : state.summaries;
          return { ids: nextIds, summaries: nextSummaries };
        });
        if (added && get().syncEnabled) {
          postSaveEvent(eventId)
            .then((summary) => {
              // Reconcile: replace the optimistic row with the BE truth so
              // bannerUrl + savedAt match what other clients will see.
              set((state) => ({
                summaries: [
                  summary,
                  ...state.summaries.filter((s) => s.id !== summary.id),
                ],
              }));
            })
            .catch(() => {
              set((state) => ({
                ids: state.ids.filter((x) => x !== eventId),
                summaries: state.summaries.filter((s) => s.id !== eventId),
              }));
            });
        }
      },
      unsave: (eventId) => {
        let removed = false;
        let previous: SavedEventSummary | undefined;
        set((state) => {
          if (!state.ids.includes(eventId)) return state;
          removed = true;
          previous = state.summaries.find((s) => s.id === eventId);
          return {
            ids: state.ids.filter((x) => x !== eventId),
            summaries: state.summaries.filter((s) => s.id !== eventId),
          };
        });
        if (removed && get().syncEnabled) {
          postUnsaveEvent(eventId).catch(() => {
            set((state) =>
              state.ids.includes(eventId)
                ? state
                : {
                    ids: [...state.ids, eventId],
                    summaries: previous
                      ? [previous, ...state.summaries]
                      : state.summaries,
                  },
            );
          });
        }
      },
      toggle: (eventId, optimistic) => {
        const has = get().ids.includes(eventId);
        if (has) get().unsave(eventId);
        else get().save(eventId, optimistic);
      },
      clear: () => set({ ids: [], summaries: [] }),
    }),
    {
      name: "quickpitik-saved-events",
      // Persist IDs only — summaries are rehydrated from BE on auth. Keeping
      // summaries out of storage avoids stale banner/state values surviving
      // across sessions; the small race-log loading flash on refresh is
      // acceptable because the SaveButton's isSaved check uses ids.
      partialize: (state) => ({ ids: state.ids }),
      version: 2,
      migrate: (persisted, fromVersion) => {
        if (!persisted || typeof persisted !== "object") {
          return { ids: [], summaries: [] };
        }
        const ids = (persisted as { ids?: unknown }).ids;
        const cleanIds = Array.isArray(ids)
          ? ids.filter(
              (x): x is string => typeof x === "string" && UUID_RE.test(x),
            )
          : [];
        // v1 → v2: summaries field added; drop any persisted summaries from v1
        // (there were none, but defensively reset) and let AuthHydrator refill
        // from BE.
        void fromVersion;
        return { ids: cleanIds, summaries: [] };
      },
    },
  ),
);
