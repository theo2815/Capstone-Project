// Single source of truth for refund-policy copy. Surfaced in two places:
//   - /events/[slug]?browse=1 — pre-purchase policy disclosure (read-only modal)
//   - /orders receipt — collapsible <details> inside the request modal
// Both consumers render <RefundPolicyContent />; this file owns the words.

import { SUPPORT_EMAIL } from "@/lib/constants";

export const REFUND_PROCESSING_DAYS: number = 3;
// Mirrors the backend's app.platform.payout-clearing-window (P7D): requests
// past it are refused, so a refund never follows a photographer payout.
export const REFUND_ELIGIBILITY_DAYS: number = 7;

export interface RefundPolicyBullet {
  kicker: string;
  body: string;
}

export const REFUND_POLICY_BULLETS: ReadonlyArray<RefundPolicyBullet> = [
  {
    kicker: "Our promise",
    body: "If something goes wrong on our side, we'll make it right. You get a refund when a photo you paid for won't download or the file is broken, a photo is too blurred to recognise anyone in it, you were charged more than once, or checkout failed or charged you the wrong amount.",
  },
  {
    kicker: "Download problems",
    body: "If a photo won't download or the file is broken because of a fault on our side, we'll fix it so you can download it. If we can't, we refund you in full.",
  },
  {
    kicker: "Blurred photos",
    body: "Some motion blur and grain is normal at a live race, and we don't refund for it. If a photo is so blurred that nobody in it can be recognised, ask for a refund. Our team checks every request the same way and makes the final call.",
  },
  {
    kicker: "Charged more than once",
    body: "A double tap, a dropped connection, a slow payment screen or buying the same photo again by mistake can charge you twice. Once our records confirm the extra charge, we refund it.",
  },
  {
    kicker: "Checkout errors",
    body: "If you were charged but your purchase failed or went wrong because of an error on our side, we refund you in full. We may ask for your transaction ID, your e-mail or a screenshot so we can find it.",
  },
  {
    kicker: "What we don't refund",
    body: `Picking the wrong photo, not liking how a photo looks (the angle, the light, your expression), changing your mind about a photo that arrived fine, and the normal imperfections of race-day photos when you can still tell who is in them. After ${REFUND_ELIGIBILITY_DAYS} days, every order is final.`,
  },
  {
    kicker: "How to ask",
    body: `Within ${REFUND_ELIGIBILITY_DAYS} days of buying, open Orders, choose the receipt and tap Request a refund. Pick the photos and the reason, and add a note if it helps. Charged but no order showed up? E-mail ${SUPPORT_EMAIL} with your transaction ID, the e-mail you used and a screenshot if you have one.`,
  },
  {
    kicker: "How long it takes",
    body: `We review every request within ${REFUND_PROCESSING_DAYS} business days and show the status on your receipt. Once we approve it, we send your refund within 24–48 hours as a PayMongo refund link, shown on your receipt with a claim-by date and sent to your e-mail. Open it before that date and choose where the money goes. Your bank or e-wallet (GCash, for example) may then take 3–10 business days to show it.`,
  },
  {
    kicker: "How much comes back",
    body: "We refund what you paid. If your bank, e-wallet or a currency conversion kept a fee that can't be returned, you may get slightly less back.",
  },
  {
    kicker: "Photos bought together",
    body: "Refunds go back through PayMongo, which can only return a whole payment. So we can refund a photo only when it was the only photo in that payment.",
  },
];
