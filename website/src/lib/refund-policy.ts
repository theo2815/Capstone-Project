// Single source of truth for refund-policy copy. Surfaced in two places:
//   - /events/[slug]?browse=1 — pre-purchase policy disclosure (read-only modal)
//   - /orders receipt — collapsible <details> inside the request modal
// Both consumers render <RefundPolicyContent />; this file owns the words.

export const REFUND_PROCESSING_DAYS: number = 3;
export const REFUND_ELIGIBILITY_DAYS: number = 30;

export interface RefundPolicyBullet {
  kicker: string;
  body: string;
}

export const REFUND_POLICY_BULLETS: ReadonlyArray<RefundPolicyBullet> = [
  {
    kicker: "Eligibility",
    body: `Request a refund within ${REFUND_ELIGIBILITY_DAYS} days of your purchase. After that the order is final.`,
  },
  {
    kicker: "Accepted reasons",
    body: "Wrong runner in the photo, photo quality too low to use, order paid but never delivered, or you were charged twice for the same order. Anything else, pick Other and tell us what happened.",
  },
  {
    kicker: "Review time",
    body: `We review every request within ${REFUND_PROCESSING_DAYS} business days. You'll see the status update on this receipt — no email needed.`,
  },
  {
    kicker: "Where the money goes",
    body: "Approved refunds return to your original payment method. GCash and Maya land within 24 hours; cards take 5–7 business days depending on the bank.",
  },
  {
    kicker: "What we don't refund",
    body: "Photos you've already downloaded and kept past the eligibility window, change-of-mind after 30 days, or photos that match your bib and selfie correctly.",
  },
];
