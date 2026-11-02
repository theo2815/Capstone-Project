import { create } from "zustand";
import { persist } from "zustand/middleware";
import { FACE_NOTICE_VERSION } from "@/lib/legal/copy";

// The face-matching agreement ticked before the first selfie search or save
// (privacy plan D2: remembered on the device, no server record).
//   Owner:   the device, so guests are covered too.
//   Key:     quickpitik-face-consent (localStorage).
//   Version: FACE_NOTICE_VERSION; raising it asks everyone again.
//   Reset:   never on sign-out, so it is NOT in resetUserScopedStores().
interface FaceConsentState {
  acceptedVersion: number | null;
  accept: () => void;
}

export const useFaceConsentStore = create<FaceConsentState>()(
  persist(
    (set) => ({
      acceptedVersion: null,
      accept: () => set({ acceptedVersion: FACE_NOTICE_VERSION }),
    }),
    { name: "quickpitik-face-consent" },
  ),
);

export const useFaceConsent = () =>
  useFaceConsentStore((s) => s.acceptedVersion === FACE_NOTICE_VERSION);
