import { create } from "zustand";
import { persist } from "zustand/middleware";
import { BIB_NOTICE_VERSION, FACE_NOTICE_VERSION } from "@/lib/legal/copy";

// The agreements ticked before a selfie search or save (privacy plan D2) and
// before a bib search (task BT-1): remembered on the device, no server record.
//   Owner:   the device, so guests are covered too.
//   Keys:    quickpitik-face-consent, quickpitik-bib-consent (localStorage).
//   Version: FACE_NOTICE_VERSION, BIB_NOTICE_VERSION; raising one asks everyone again.
//   Reset:   never on sign-out, so they are NOT in resetUserScopedStores().
interface ConsentState {
  acceptedVersion: number | null;
  accept: () => void;
}

const consentStore = (name: string, version: number) =>
  create<ConsentState>()(
    persist(
      (set) => ({
        acceptedVersion: null,
        accept: () => set({ acceptedVersion: version }),
      }),
      { name },
    ),
  );

export const useFaceConsentStore = consentStore("quickpitik-face-consent", FACE_NOTICE_VERSION);
export const useBibConsentStore = consentStore("quickpitik-bib-consent", BIB_NOTICE_VERSION);

export const useFaceConsent = () =>
  useFaceConsentStore((s) => s.acceptedVersion === FACE_NOTICE_VERSION);
export const useBibConsent = () =>
  useBibConsentStore((s) => s.acceptedVersion === BIB_NOTICE_VERSION);
