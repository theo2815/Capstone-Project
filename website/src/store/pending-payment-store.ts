import { create } from "zustand";
import { persist } from "zustand/middleware";

// The one live QR Ph payment, persisted so a refresh, a closed tab, or an
// accidental modal dismissal never loses the order reference. The backend
// reserves overlapping photos per actor, so there is never more than one.
// Cleared on success, expiry, or a dead return token.
export interface PendingPayment {
  orderId: string;
  imageUrl: string;
  expiresAt: string;
  // Guest orders carry a signed RETURN capability; signed-in orders poll
  // /me/... with the JWT. Branch on this, never on auth state — a guest who
  // logs in mid-checkout still owns a userId-less order.
  returnToken: string | null;
  email: string;
  total: number;
  itemCount: number;
  // Set when the user taps "I've paid" — flips the waiting copy to
  // "confirming" and starts the elapsed clock.
  paidClaimedAt: string | null;
}

interface PendingPaymentState {
  pending: PendingPayment | null;
  set: (p: PendingPayment) => void;
  markPaid: () => void;
  clear: () => void;
}

export const usePendingPaymentStore = create<PendingPaymentState>()(
  persist(
    (set, get) => ({
      pending: null,
      set: (pending) => set({ pending }),
      markPaid: () => {
        const p = get().pending;
        if (p && !p.paidClaimedAt) {
          set({ pending: { ...p, paidClaimedAt: new Date().toISOString() } });
        }
      },
      clear: () => set({ pending: null }),
    }),
    { name: "quickpitik-pending-payment" },
  ),
);
