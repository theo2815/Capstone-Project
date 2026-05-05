import type { EventDetail } from "@/types/event";

export interface MockPhoto {
  id: string;
  bib: string | null;
  km: number | null;
  time: string;
  tone: number;
  span: "default" | "wide";
  price: number;
  imageUrl?: string | null;
  alt?: string;
}

export const DEMO_BIB = "B-4082";

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

export function deriveKmStops(categories: string[]): number[] {
  const stops = new Set<number>();
  for (const c of categories) {
    if (c.toLowerCase().includes("marathon") || c.includes("42")) {
      [5, 10, 21, 30, 42].forEach((k) => stops.add(k));
    } else if (c.includes("21")) {
      [5, 10, 15, 21].forEach((k) => stops.add(k));
    } else if (c.includes("15")) {
      [3, 8, 15].forEach((k) => stops.add(k));
    } else if (c.includes("10")) {
      [2, 5, 10].forEach((k) => stops.add(k));
    } else if (c.includes("5K")) {
      [1, 3, 5].forEach((k) => stops.add(k));
    } else if (c.includes("3K")) {
      [1, 2, 3].forEach((k) => stops.add(k));
    }
  }
  if (stops.size === 0) {
    [3, 5, 10].forEach((k) => stops.add(k));
  }
  return Array.from(stops).sort((a, b) => a - b);
}

function formatRaceTime(minutesFromGun: number): string {
  const startHour = 4;
  const total = startHour * 60 + minutesFromGun;
  const h = Math.floor(total / 60) % 24;
  const m = Math.floor(total) % 60;
  return `${h.toString().padStart(2, "0")}:${m.toString().padStart(2, "0")}`;
}

export function generateMockPhotos(event: EventDetail): MockPhoto[] {
  const rng = mulberry32(hashStr(event.slug));
  const kmStops = deriveKmStops(event.categories);

  const photos: MockPhoto[] = [];
  const targetTotal = 96;
  const demoBibCount = 14;
  const untaggedCount = 12;
  const otherCount = targetTotal - demoBibCount - untaggedCount;

  const otherBibs: string[] = [];
  for (let i = 0; i < 60; i++) {
    const n = 1000 + Math.floor(rng() * 8000);
    if (n === 4082) continue;
    otherBibs.push(`B-${n.toString().padStart(4, "0")}`);
  }

  const minutesAtKm = (km: number | null) => {
    if (km === null) return Math.floor(rng() * 240);
    const paceMinPerKm = 6 + rng() * 2;
    return Math.floor(km * paceMinPerKm + (rng() - 0.5) * 12);
  };

  for (let i = 0; i < demoBibCount; i++) {
    const km = kmStops[i % kmStops.length];
    photos.push({
      id: `${event.id}-demo-${i}`,
      bib: DEMO_BIB,
      km,
      time: formatRaceTime(minutesAtKm(km)),
      tone: i % 4,
      span: "default",
      price: event.pricePerPhoto,
    });
  }

  for (let i = 0; i < otherCount; i++) {
    const bib = otherBibs[Math.floor(rng() * otherBibs.length)] ?? null;
    const km = kmStops[Math.floor(rng() * kmStops.length)];
    photos.push({
      id: `${event.id}-${i}`,
      bib,
      km,
      time: formatRaceTime(minutesAtKm(km)),
      tone: Math.floor(rng() * 4),
      span: "default",
      price: event.pricePerPhoto,
    });
  }

  for (let i = 0; i < untaggedCount; i++) {
    const km = kmStops[Math.floor(rng() * kmStops.length)];
    photos.push({
      id: `${event.id}-crowd-${i}`,
      bib: null,
      km,
      time: formatRaceTime(minutesAtKm(km)),
      tone: Math.floor(rng() * 4),
      span: "default",
      price: event.pricePerPhoto,
    });
  }

  for (let i = photos.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [photos[i], photos[j]] = [photos[j], photos[i]];
  }

  for (let i = 6; i < photos.length; i += 7) {
    photos[i].span = "wide";
  }

  return photos;
}
