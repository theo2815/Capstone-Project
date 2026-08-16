// Mocked photographer dashboard data — earnings rollups, pipeline counters,
// and event summary list. All Cebu-flavored to match the wider site's tone.
//
// TODO(backend): replace with `api.get("/me/photographer/earnings|pipeline|events")`
// once Spring Boot Phase F (Photographer Workspace) lands. The endpoints will
// return identical shapes so the dashboard reads stay unchanged.

export interface WeeklyRevenuePoint {
  /** ISO Monday of the week (e.g., "2026-05-04"). */
  weekOf: string;
  /** Peso amount kept by the photographer that week (post-platform-cut). */
  amount: number;
}

export interface PhotographerEarnings {
  lifetimeKept: number;
  thisWeek: number;
  thisMonth: number;
  payoutPending: number;
  /** Next scheduled payout date, ISO. */
  payoutScheduledFor: string | null;
  /** 12 weeks chronological (oldest first, newest last). */
  weeklySeries: ReadonlyArray<WeeklyRevenuePoint>;
  thisWeekSold: number;
  thisMonthSold: number;
}

export interface PhotographerPipeline {
  /** Event ID being actively covered. null when no live event. */
  activeEventId: string | null;
  /** Total photos uploaded for the active event. */
  uploaded: number;
  /** Photos published to the marketplace. */
  live: number;
  /** Pre-formatted activity strings — backend will format these or return ISO. */
  lastUploadDisplay: string | null;
  lastSaleDisplay: string | null;
}

export type EventState = "live" | "upcoming" | "open" | "past";

export interface PhotographerEventSummary {
  id: string;
  slug: string;
  name: string;
  /** Event date, ISO date-only (YYYY-MM-DD). */
  date: string;
  location: string;
  /** Resolved event cover URL — presigned cover_s3_key or legacy banner_url
   *  fallback. Same value runners see on /events tiles; null when the admin
   *  hasn't uploaded a cover yet (EventTile then falls back to its dark
   *  placeholder). */
  bannerUrl?: string | null;
  state: EventState;
  /** Total photos uploaded by this photographer for this event. */
  photoCount: number;
  /** Number of distinct sales (one buyer can buy multiple photos = 1 sale). */
  salesCount: number;
  /** ₱ kept by the photographer for this event (post-platform-cut). */
  revenueKept: number;
}

export const PHOTOGRAPHER_EARNINGS: PhotographerEarnings = {
  lifetimeKept: 0,
  thisWeek: 0,
  thisMonth: 0,
  payoutPending: 0,
  payoutScheduledFor: null,
  thisWeekSold: 0,
  thisMonthSold: 0,
  weeklySeries: [],
};

export const PHOTOGRAPHER_PIPELINE: PhotographerPipeline = {
  activeEventId: null,
  uploaded: 0,
  live: 0,
  lastUploadDisplay: null,
  lastSaleDisplay: null,
};
// The photographer's covered events — a subset of EVENT_CATALOG (most events
// in the catalog are organized by other photographers / organizers in real
// data). Listed newest-first by date.
export const PHOTOGRAPHER_EVENTS: ReadonlyArray<PhotographerEventSummary> = [];

export function getPhotographerEventById(
  id: string,
): PhotographerEventSummary | undefined {
  return PHOTOGRAPHER_EVENTS.find((e) => e.id === id);
}

// ─────────────────────────────────────────────────────────────────────────
// Photographer photo library (dashboard /events/[id]/photos)
//
// Shape oriented to management — status, sales count — distinct from the
// runner-facing MockPhoto on /events/[slug]. Backend will return these from
// `GET /me/photographer/events/{id}/photos` once Spring Boot Phase F lands.
//
// Web uploads do NOT go through AI blur detection (that lives only in the
// desktop app). Photos go straight to "live" on publish; the photographer
// can manually "hide" individual photos from the public gallery.
// ─────────────────────────────────────────────────────────────────────────

export type PhotoStatus = "live" | "hidden";

export interface PhotographerLibraryPhoto {
  id: string;
  bib: string | null;
  status: PhotoStatus;
  salesCount: number;
  /** ISO timestamp of upload. */
  uploadedAt: string;
  /** Tone index for the placeholder tile color (0..3). */
  tone: number;
  /**
   * Aspect span for mosaic grids — `wide` = landscape (row-span-1),
   * `default` = portrait (row-span-2). Mirrors `MockPhoto.span` so the
   * photographer-facing share page can reuse the runner-facing browse
   * layout. Every 7th photo is wide.
   */
  span: "default" | "wide";
  /** Watermarked thumbnail URL from the BE (same image runners see in
   *  /events/[slug]?browse=1). Null while the BE hasn't responded or for
   *  legacy mock-generated rows that never carried one. */
  thumbnailUrl?: string | null;
}

function hashStr(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function mulberry32(seed: number) {
  let s = seed;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function generatePhotographerLibrary(
  event: PhotographerEventSummary,
  cap = 120,
): PhotographerLibraryPhoto[] {
  const totalCount = event.photoCount;
  const visibleCount = Math.min(totalCount, cap);
  if (visibleCount === 0) return [];

  const rng = mulberry32(hashStr(`library-${event.id}`));
  const isActive = event.id === PHOTOGRAPHER_PIPELINE.activeEventId;

  const eventStart = new Date(`${event.date}T04:00:00Z`).getTime();
  // Distribute uploads across a 4-hour race window for active events,
  // pre-event for upcoming, and a single past timestamp for closed events.
  const uploadOffsetMs = (i: number) => {
    if (isActive) return i * 24 * 1000; // ~24s apart, race-day pace
    return -i * 60 * 1000; // historic — older photos uploaded later
  };

  // Roll a sales count weighted toward 0 — most photos don't sell, a few do.
  const rollSales = (): number => {
    const r = rng();
    if (r < 0.82) return 0;
    if (r < 0.96) return 1;
    if (r < 0.995) return 2 + Math.floor(rng() * 3);
    return 6 + Math.floor(rng() * 4);
  };

  const photos: PhotographerLibraryPhoto[] = [];
  for (let i = 0; i < visibleCount; i++) {
    const tagged = rng() < 0.78;
    const bib = tagged
      ? `B-${(1000 + Math.floor(rng() * 8000)).toString().padStart(4, "0")}`
      : null;
    photos.push({
      id: `lib-${event.id}-${i}`,
      bib,
      status: "live",
      salesCount: rollSales(),
      uploadedAt: new Date(eventStart + uploadOffsetMs(i)).toISOString(),
      tone: Math.floor(rng() * 4),
      span: "default",
    });
  }
  // Every 7th photo gets the wide span — same cadence as MockPhoto so the
  // photographer's share-page mosaic feels identical to the runner browse.
  for (let i = 6; i < photos.length; i += 7) {
    photos[i].span = "wide";
  }
  return photos;
}

// ─────────────────────────────────────────────────────────────────────────
// Billing
// ─────────────────────────────────────────────────────────────────────────

// "held" added for the request-based flow: admin can hold a photographer's
// request pending a fix (e.g. update payout account); photographer sees the
// reason and can Withdraw to re-request after addressing it.
export type PayoutStatus = "paid" | "pending" | "scheduled" | "held";

export interface PhotographerPayout {
  id: string;
  /** @deprecated ISO Monday, an artifact of the scheduled-weekly model that
   *  the request-based flow replaced on 2026-05-19. There are no weekly cycles
   *  any more, so "Week of {weekOf}" mislabels a payout — render `requestedAt`.
   *
   *  The BE keeps populating it for THIS app's admin payout queue, which is the
   *  last consumer. The previous note here blamed mobile; that was wrong —
   *  mobile parses the field but never renders it (verified 2026-08-16). */
  weekOf: string;
  /** ISO timestamp of when the photographer asked to be paid. The honest label
   *  for a request-based payout. */
  requestedAt: string;
  amount: number;
  status: PayoutStatus;
  /** ISO timestamp when the payout was settled (paid) or admin approved it.
   *  Null for fresh pending_review requests that haven't been touched yet. */
  settledAt: string | null;
  /** Q-E1 RESOLVED 2026-05-10: enum locked to gcash | maya | gotyme. */
  method: "gcash" | "maya" | "gotyme";
  /** Reference number for paid cycles. null for pending/scheduled. */
  reference: string | null;
  /** Admin's reason when status is "held"; null otherwise. */
  holdReason?: string | null;
}

export interface PhotographerTransaction {
  id: string;
  /** ISO timestamp the runner paid. */
  paidAt: string;
  eventId: string;
  /** Snapshot of the event title at sale time. Null when the event was
   *  hard-deleted — the row then renders the "Event archived" fallback. */
  eventName?: string | null;
  /** Event slug snapshot — kept for future deep-linking from the row. */
  eventSlug?: string | null;
  photoId: string;
  /** Display name of the buyer (mock — backend returns a privacy-safe handle). */
  buyer: string;
  /** ₱ kept by the photographer after platform cut. */
  amountKept: number;
}

// Most recent first. 6 cycles total — 4 paid, 1 pending, 1 scheduled. Adds up
// to ~₱24,850 lifetime when combined with seed earnings on the dashboard.
export const PHOTOGRAPHER_PAYOUTS: ReadonlyArray<PhotographerPayout> = [];

// Most recent first. Mix of buyers and events to populate the ledger
// realistically. Total roughly matches this-week earnings (₱4,250 across 12
// sales = avg ₱354/sale, but real spread varies).
export const PHOTOGRAPHER_TRANSACTIONS: ReadonlyArray<PhotographerTransaction> = [];
