import { readFileSync } from "node:fs";
import path from "node:path";
import { expect, it } from "vitest";

// Runner audit P3-1: design-system hard rule 4 — no interactive or informational
// text under 12 px. Runner-facing files may keep sub-12 px sizes only for the
// counted decorative spots below (drawn mockups, avatar initials, the logo in a
// photo placeholder, cart-count bubbles, the hover hint over a photo).
const SRC = path.join(__dirname, "..");
const ALLOWED: Record<string, number> = {
  "app/runners/page.tsx": 7, // phone mockups in the how-it-works steps
  "app/events/events-browser.tsx": 3, // hero illustration (match card, bib, clock)
  "app/[handle]/events/[slug]/page.tsx": 2, // photographer chip initials and dot
  "app/orders/return/page.tsx": 1, // "QuickPitik" in a photo placeholder
  "components/cart/cart-modal.tsx": 1, // "QuickPitik" in a thumbnail placeholder
  "components/cart/floating-cart.tsx": 2, // count bubbles (aria-hidden)
  "components/cart/checkout-modal.tsx": 0,
  "components/photos/photo-preview-card.tsx": 0,
  "app/orders/page.tsx": 0,
  "app/profile/page.tsx": 0,
  "components/events/photo-mosaic-tile.tsx": 1, // "View →" hover hint (aria-hidden)
  "components/layout/user-menu.tsx": 1, // avatar initials
  "components/events/bib-empty-result.tsx": 0,
  "components/events/bib-search-panels.tsx": 0,
  "components/events/buy-all-bar.tsx": 0,
  "components/events/find-photos-modal.tsx": 0,
  "components/events/save-button.tsx": 0,
  "components/orders/refund-timeline.tsx": 0,
  "components/profile/selfie-library.tsx": 0,
};

it.each(Object.entries(ALLOWED))("%s keeps text at 12 px or more outside its decorative spots", (file, allowed) => {
  const source = readFileSync(path.join(SRC, file), "utf8");
  const small = source.match(/text-\[(?:[0-9]|1[01])px\]/g) ?? [];
  expect(small.length, `${file} sub-12 px sizes`).toBe(allowed);
});
