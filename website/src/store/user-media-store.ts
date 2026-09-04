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
  setAvatar: (avatar: AvatarMedia | null) => void;
  clear: () => void;
}

// The selfie slice this store once persisted is gone (2026-08-28 audit): its
// actions had zero callers — the library is BE-backed via useSelfiesList() —
// yet every selfie kept a base64 data URL of the user's face in localStorage,
// the most sensitive artifact the product touches. The next persist write
// drains any old `selfies` key from storage.
export const useUserMediaStore = create<UserMediaState>()(
  persist(
    (set) => ({
      avatar: null,
      setAvatar: (avatar) => set({ avatar }),
      clear: () => set({ avatar: null }),
    }),
    { name: "quickpitik-user-media" },
  ),
);
