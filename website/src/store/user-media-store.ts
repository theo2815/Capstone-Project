import { create } from "zustand";
import { persist } from "zustand/middleware";

// TODO(backend): swap localStorage persist for `/me/avatar` + `/me/selfies` endpoints
// once Spring Boot Phase B exposes them. State shape mirrors the eventual API contract
// (see _journal/2026-05-05-selfie-library-design in the vault).

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
  qualityScore: number; // 0–1, mocked client-side until ai-api wires up
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
