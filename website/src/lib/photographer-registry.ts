// Public photographer registry — used to render `/[handle]` and
// `/[handle]/events/[slug]` for ANY photographer (not just the logged-in one).
// The current photographer's preview is resolved from `usePhotographerSettingsStore`
// inside `useResolvedPhotographer` below; everyone else is read from this static
// list.
//
// TODO(backend): replace with `api.get<PhotographerPublic>("/p/{handle}")` once
// Spring Boot Phase F lands. The PhotographerProfile shape is the rough public
// view contract — swap CoverSource for a signed S3 URL field.

import type { EventState } from "@/lib/photographer-mock";
import type { BrandColor } from "@/store/photographer-settings-store";
import type { MockPhoto } from "@/app/events/[slug]/mock-photos";
import type { EventDetail } from "@/types/event";

export type CoverSource =
  | { kind: "gradient"; from: string; to: string }
  | { kind: "image"; url: string };

export interface PhotographerEventCoverage {
  /** matches MOCK_EVENT_DETAILS keys + EventDetail.slug */
  eventSlug: string;
  state: EventState;
  photoCount: number;
  salesCount: number;
}

export interface PhotographerProfile {
  handle: string;
  displayName: string;
  brandColor: BrandColor;
  bio: string;
  city: string;
  /** ISO date the photographer joined the platform. */
  memberSince: string;
  cover: CoverSource | null;
  /** Public watermark text — displayed as a tiny mono chip. Real watermark is
   * the uploaded PNG; this is the photographer's display label. */
  watermarkLabel: string | null;
  events: ReadonlyArray<PhotographerEventCoverage>;
}

export const PHOTOGRAPHER_REGISTRY: ReadonlyArray<PhotographerProfile> = [
  {
    handle: "paksitphotos",
    displayName: "Paksit Photos",
    brandColor: "amber",
    bio: "Cebu-based race photographer. Bridge climbs, finish-line tape, golden-hour bib portraits. Shooting since 2019.",
    city: "Cebu City",
    memberSince: "2024-06-12",
    cover: { kind: "gradient", from: "#D97706", to: "#7C2D12" },
    watermarkLabel: "PAKSIT",
    events: [
      {
        eventSlug: "cebu-marathon-2026",
        state: "live",
        photoCount: 412,
        salesCount: 28,
      },
      {
        eventSlug: "srp-half-marathon-2026",
        state: "open",
        photoCount: 285,
        salesCount: 12,
      },
      {
        eventSlug: "mactan-coastal-5k-2026",
        state: "open",
        photoCount: 198,
        salesCount: 9,
      },
      {
        eventSlug: "cebu-night-run-2025",
        state: "past",
        photoCount: 1120,
        salesCount: 47,
      },
    ],
  },
  {
    handle: "cebustride",
    displayName: "Cebu Stride Co.",
    brandColor: "indigo",
    bio: "Two-photographer team covering Cebu road races since 2022. Action-first, candid, no posed shots.",
    city: "Mandaue",
    memberSince: "2025-01-08",
    cover: { kind: "gradient", from: "#4F46E5", to: "#1E1B4B" },
    watermarkLabel: "CSC",
    events: [
      {
        eventSlug: "mactan-sunset-run-2026",
        state: "open",
        photoCount: 280,
        salesCount: 18,
      },
      {
        eventSlug: "sun-run-cebu-2026",
        state: "past",
        photoCount: 845,
        salesCount: 32,
      },
      {
        eventSlug: "talisay-10k-2025",
        state: "past",
        photoCount: 612,
        salesCount: 21,
      },
    ],
  },
];

const REGISTRY_BY_HANDLE: Record<string, PhotographerProfile> =
  Object.fromEntries(PHOTOGRAPHER_REGISTRY.map((p) => [p.handle, p]));

export function getPhotographerByHandle(
  handle: string,
): PhotographerProfile | null {
  return REGISTRY_BY_HANDLE[handle.trim().toLowerCase()] ?? null;
}

export function getPhotographerCoverage(
  profile: PhotographerProfile,
  eventSlug: string,
): PhotographerEventCoverage | null {
  return profile.events.find((e) => e.eventSlug === eventSlug) ?? null;
}

// Aggregate stats for the public profile hero. Pure derived numbers — backend
// will return these pre-computed in the real `GET /p/{handle}` response.
export function getProfileTotals(
  profile: PhotographerProfile,
): { events: number; photos: number; sales: number } {
  const visibleEvents = profile.events.filter((e) => e.photoCount > 0);
  return {
    events: visibleEvents.length,
    photos: visibleEvents.reduce((sum, e) => sum + e.photoCount, 0),
    sales: visibleEvents.reduce((sum, e) => sum + e.salesCount, 0),
  };
}

// ─────────────────────────────────────────────────────────────────────────
// Photo generator — deterministic per (handle, slug). The shape matches
// `MockPhoto` so `PhotoTile` + `PhotoPreviewCard` reuse unchanged.
// ─────────────────────────────────────────────────────────────────────────

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

function formatRaceTime(minutesFromGun: number): string {
  const total = 4 * 60 + minutesFromGun;
  const h = Math.floor(total / 60) % 24;
  const m = Math.floor(total) % 60;
  return `${h.toString().padStart(2, "0")}:${m.toString().padStart(2, "0")}`;
}

// Generate a deterministic slice of mock photos for one photographer at one
// event. `cap` keeps the rendered grid sane — the real backend serves a
// page of N from the photographer's full upload count.
export function generatePhotographerPhotos(
  handle: string,
  event: EventDetail,
  count: number,
  cap = 60,
): MockPhoto[] {
  const photoCount = Math.min(count, cap);
  const rng = mulberry32(hashStr(`${handle}::${event.slug}`));
  const out: MockPhoto[] = [];

  for (let i = 0; i < photoCount; i++) {
    const taggedRoll = rng();
    const bib =
      taggedRoll < 0.85
        ? `B-${(1000 + Math.floor(rng() * 8000)).toString().padStart(4, "0")}`
        : null;
    const km = pickKmFromCategories(event.categories, rng);
    const minutes = Math.floor(km * (6 + rng() * 2) + (rng() - 0.5) * 12);

    out.push({
      id: `${handle}-${event.slug}-${i}`,
      bib,
      km,
      time: formatRaceTime(minutes),
      tone: Math.floor(rng() * 4),
      span: i > 0 && i % 7 === 0 ? "wide" : "default",
      price: event.pricePerPhoto,
    });
  }

  return out;
}

function pickKmFromCategories(
  categories: string[],
  rng: () => number,
): number {
  const stops = new Set<number>();
  for (const c of categories) {
    if (c.toLowerCase().includes("marathon") || c.includes("42")) {
      [5, 10, 21, 30, 42].forEach((k) => stops.add(k));
    } else if (c.includes("21")) {
      [5, 10, 15, 21].forEach((k) => stops.add(k));
    } else if (c.includes("10")) {
      [2, 5, 10].forEach((k) => stops.add(k));
    } else if (c.includes("5K")) {
      [1, 3, 5].forEach((k) => stops.add(k));
    } else {
      [3, 5].forEach((k) => stops.add(k));
    }
  }
  if (stops.size === 0) {
    return 5;
  }
  const arr = Array.from(stops);
  return arr[Math.floor(rng() * arr.length)];
}
