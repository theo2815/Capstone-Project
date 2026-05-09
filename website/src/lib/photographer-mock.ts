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
  payoutScheduledFor: string;
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
  state: EventState;
  /** Total photos uploaded by this photographer for this event. */
  photoCount: number;
  /** Number of distinct sales (one buyer can buy multiple photos = 1 sale). */
  salesCount: number;
  /** ₱ kept by the photographer for this event (post-platform-cut). */
  revenueKept: number;
}

export const PHOTOGRAPHER_EARNINGS: PhotographerEarnings = {
  lifetimeKept: 24850,
  thisWeek: 4250,
  thisMonth: 12400,
  payoutPending: 3800,
  payoutScheduledFor: "2026-05-08",
  thisWeekSold: 12,
  thisMonthSold: 48,
  // Twelve weeks ramping into the current week — Cebu Marathon and SRP Half-
  // Marathon push the last 4 weeks up significantly. Demo-flavored, not random.
  weeklySeries: [
    { weekOf: "2026-02-16", amount: 620 },
    { weekOf: "2026-02-23", amount: 540 },
    { weekOf: "2026-03-02", amount: 880 },
    { weekOf: "2026-03-09", amount: 720 },
    { weekOf: "2026-03-16", amount: 1100 },
    { weekOf: "2026-03-23", amount: 940 },
    { weekOf: "2026-03-30", amount: 1320 },
    { weekOf: "2026-04-06", amount: 1180 },
    { weekOf: "2026-04-13", amount: 2400 },
    { weekOf: "2026-04-20", amount: 1980 },
    { weekOf: "2026-04-27", amount: 3050 },
    { weekOf: "2026-05-04", amount: 4250 },
  ],
};

export const PHOTOGRAPHER_PIPELINE: PhotographerPipeline = {
  activeEventId: "1", // Cebu Marathon (matches EVENT_CATALOG)
  uploaded: 847,
  live: 847,
  lastUploadDisplay: "12 sec ago",
  lastSaleDisplay: "3 min ago",
};

// The photographer's covered events — a subset of EVENT_CATALOG (most events
// in the catalog are organized by other photographers / organizers in real
// data). Listed newest-first by date.
export const PHOTOGRAPHER_EVENTS: ReadonlyArray<PhotographerEventSummary> = [
  {
    id: "u1",
    slug: "cebu-bay-run-2026",
    name: "Cebu Bay Run",
    date: "2026-05-09",
    location: "Mactan Channel Bridge",
    state: "upcoming",
    photoCount: 0,
    salesCount: 0,
    revenueKept: 0,
  },
  {
    id: "1",
    slug: "cebu-marathon-2026",
    name: "Cebu Marathon 2026",
    date: "2026-04-28",
    location: "SRP Boulevard, Cebu City",
    state: "live",
    photoCount: 847,
    salesCount: 12,
    revenueKept: 960,
  },
  {
    id: "3",
    slug: "srp-half-marathon-2026",
    name: "SRP Half-Marathon",
    date: "2026-04-12",
    location: "South Road Properties, Cebu",
    state: "open",
    photoCount: 612,
    salesCount: 4,
    revenueKept: 320,
  },
  {
    id: "5",
    slug: "mactan-coastal-5k-2026",
    name: "Mactan Coastal 5K",
    date: "2026-03-29",
    location: "Mactan, Lapu-Lapu City",
    state: "open",
    photoCount: 480,
    salesCount: 18,
    revenueKept: 1440,
  },
  {
    id: "6",
    slug: "cebu-night-run-2025",
    name: "Cebu City Night Run 2025",
    date: "2025-12-14",
    location: "Cebu Business Park",
    state: "past",
    photoCount: 1820,
    salesCount: 64,
    revenueKept: 5120,
  },
];

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

export type PayoutStatus = "paid" | "pending" | "scheduled";

export interface PhotographerPayout {
  id: string;
  /** ISO Monday of the cycle this payout covers. */
  weekOf: string;
  amount: number;
  status: PayoutStatus;
  /** ISO timestamp when the payout was settled (paid) or is expected (others). */
  settledAt: string;
  /** Q-E1 RESOLVED 2026-05-10: enum locked to gcash | maya | gotyme. */
  method: "gcash" | "maya" | "gotyme";
  /** Reference number for paid cycles. null for pending/scheduled. */
  reference: string | null;
}

export interface PhotographerTransaction {
  id: string;
  /** ISO timestamp the runner paid. */
  paidAt: string;
  eventId: string;
  photoId: string;
  /** Display name of the buyer (mock — backend returns a privacy-safe handle). */
  buyer: string;
  /** ₱ kept by the photographer after platform cut. */
  amountKept: number;
}

// Most recent first. 6 cycles total — 4 paid, 1 pending, 1 scheduled. Adds up
// to ~₱24,850 lifetime when combined with seed earnings on the dashboard.
export const PHOTOGRAPHER_PAYOUTS: ReadonlyArray<PhotographerPayout> = [
  {
    id: "PAY-2026W19-CEBUSTRIDE",
    weekOf: "2026-05-04",
    amount: 3800,
    status: "scheduled",
    settledAt: "2026-05-08T08:00:00.000Z",
    method: "gcash",
    reference: null,
  },
  {
    id: "PAY-2026W18-CEBUSTRIDE",
    weekOf: "2026-04-27",
    amount: 3050,
    status: "pending",
    settledAt: "2026-05-01T08:00:00.000Z",
    method: "gcash",
    reference: null,
  },
  {
    id: "PAY-2026W17-CEBUSTRIDE",
    weekOf: "2026-04-20",
    amount: 1980,
    status: "paid",
    settledAt: "2026-04-24T09:14:00.000Z",
    method: "gcash",
    reference: "GC-A1B2C3D4",
  },
  {
    id: "PAY-2026W16-CEBUSTRIDE",
    weekOf: "2026-04-13",
    amount: 2400,
    status: "paid",
    settledAt: "2026-04-17T09:08:00.000Z",
    method: "gcash",
    reference: "GC-E5F6G7H8",
  },
  {
    id: "PAY-2026W15-CEBUSTRIDE",
    weekOf: "2026-04-06",
    amount: 1180,
    status: "paid",
    settledAt: "2026-04-10T09:21:00.000Z",
    method: "gcash",
    reference: "GC-J9K0L1M2",
  },
  {
    id: "PAY-2026W14-CEBUSTRIDE",
    weekOf: "2026-03-30",
    amount: 1320,
    status: "paid",
    settledAt: "2026-04-03T09:11:00.000Z",
    method: "gcash",
    reference: "GC-N3O4P5Q6",
  },
];

// Most recent first. Mix of buyers and events to populate the ledger
// realistically. Total roughly matches this-week earnings (₱4,250 across 12
// sales = avg ₱354/sale, but real spread varies).
export const PHOTOGRAPHER_TRANSACTIONS: ReadonlyArray<PhotographerTransaction> = [
  {
    id: "TX-2026-1124",
    paidAt: "2026-05-06T09:18:00.000Z",
    eventId: "1",
    photoId: "mock-cm-141",
    buyer: "Aira S.",
    amountKept: 80,
  },
  {
    id: "TX-2026-1119",
    paidAt: "2026-05-05T18:42:00.000Z",
    eventId: "1",
    photoId: "mock-cm-87",
    buyer: "Mark P.",
    amountKept: 80,
  },
  {
    id: "TX-2026-1112",
    paidAt: "2026-05-05T11:03:00.000Z",
    eventId: "1",
    photoId: "mock-cm-52",
    buyer: "Joelle R.",
    amountKept: 320, // bundle of 4
  },
  {
    id: "TX-2026-1108",
    paidAt: "2026-05-04T22:11:00.000Z",
    eventId: "5",
    photoId: "mock-mc-22",
    buyer: "Karl V.",
    amountKept: 80,
  },
  {
    id: "TX-2026-1104",
    paidAt: "2026-05-04T15:55:00.000Z",
    eventId: "1",
    photoId: "mock-cm-203",
    buyer: "Hannah B.",
    amountKept: 80,
  },
  {
    id: "TX-2026-1098",
    paidAt: "2026-05-03T20:17:00.000Z",
    eventId: "5",
    photoId: "mock-mc-89",
    buyer: "Don V.",
    amountKept: 80,
  },
  {
    id: "TX-2026-1092",
    paidAt: "2026-05-03T13:24:00.000Z",
    eventId: "1",
    photoId: "mock-cm-66",
    buyer: "Trish L.",
    amountKept: 240,
  },
  {
    id: "TX-2026-1085",
    paidAt: "2026-05-02T19:00:00.000Z",
    eventId: "3",
    photoId: "mock-srp-77",
    buyer: "Ricky D.",
    amountKept: 80,
  },
  {
    id: "TX-2026-1077",
    paidAt: "2026-05-02T10:48:00.000Z",
    eventId: "5",
    photoId: "mock-mc-44",
    buyer: "Carla M.",
    amountKept: 160,
  },
  {
    id: "TX-2026-1069",
    paidAt: "2026-05-01T17:32:00.000Z",
    eventId: "1",
    photoId: "mock-cm-19",
    buyer: "Dennis O.",
    amountKept: 80,
  },
  {
    id: "TX-2026-1062",
    paidAt: "2026-05-01T12:10:00.000Z",
    eventId: "5",
    photoId: "mock-mc-12",
    buyer: "Patrice U.",
    amountKept: 80,
  },
  {
    id: "TX-2026-1054",
    paidAt: "2026-04-30T21:44:00.000Z",
    eventId: "1",
    photoId: "mock-cm-8",
    buyer: "Lindsay K.",
    amountKept: 80,
  },
];
