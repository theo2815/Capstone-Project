import { create } from "zustand";
import { persist } from "zustand/middleware";

// User media types + local persistence. The canonical avatar lives on
// `auth-store.user.avatarUrl` (Q-007 RESOLVED) and selfies are served via
// `useSelfiesList()` (Q-006 RESOLVED). This store remains for transient
// local state and as a typed container for SelfieRef.

export const SELFIE_MAX = 5;

export interface AvatarMedia {
  dataUrl: string;
  uploadedAt: string;
}

export interface SelfieRef {
  id: string;
  dataUrl: string;
  uploadedAt: string;
  isPrimary: boolean;
  // 0–1, and 0 for EVERY selfie while the backend runs AI_API_ENABLED=false —
  // the quality gate is skipped, not failed. Read `qualityTestStatus` before
  // treating this number as meaningful.
  qualityScore: number;
  // "untested" | "passed" (backend V26). "untested" = stored while ai-api was
  // off, so it has never been checked and may not match once search goes live.
  // "rejected" is unreachable: the gate throws before the row is saved, so a
  // rejected selfie is never persisted — the FE reads that off the 4xx envelope
  // at upload time instead. Optional so an older backend response still parses.
  qualityTestStatus?: "untested" | "passed";
}

interface UserMediaState {
  avatar: AvatarMedia | null;
  selfies: SelfieRef[];
  setAvatar: (avatar: AvatarMedia | null) => void;
  addSelfie: (selfie: SelfieRef) => void;
  removeSelfie: (id: string) => void;
  setPrimary: (id: string) => void;
  clear: () => void;
}

export const useUserMediaStore = create<UserMediaState>()(
  persist(
    (set) => ({
      avatar: null,
      selfies: [],
      setAvatar: (avatar) => set({ avatar }),
      addSelfie: (incoming) =>
        set((state) => {
          if (state.selfies.length >= SELFIE_MAX) return state;
          const isFirst = state.selfies.length === 0;
          // First upload is automatically primary; subsequent uploads keep
          // the existing primary unless the caller passed isPrimary: true.
          const promote = isFirst || incoming.isPrimary;
          const others = promote
            ? state.selfies.map((s) => ({ ...s, isPrimary: false }))
            : state.selfies;
          return {
            selfies: [...others, { ...incoming, isPrimary: promote }],
          };
        }),
      removeSelfie: (id) =>
        set((state) => {
          const next = state.selfies.filter((s) => s.id !== id);
          if (next.length === 0) return { selfies: [] };
          if (next.some((s) => s.isPrimary)) return { selfies: next };
          // Removed the primary — promote the most recent remaining selfie.
          const promotedIndex = next.length - 1;
          return {
            selfies: next.map((s, i) => ({
              ...s,
              isPrimary: i === promotedIndex,
            })),
          };
        }),
      setPrimary: (id) =>
        set((state) => ({
          selfies: state.selfies.map((s) => ({
            ...s,
            isPrimary: s.id === id,
          })),
        })),
      clear: () => set({ avatar: null, selfies: [] }),
    }),
    { name: "quickpitik-user-media" },
  ),
);
