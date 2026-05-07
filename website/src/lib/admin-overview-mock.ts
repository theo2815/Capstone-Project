import type { DecisionLogEntry } from "@/store/admin-user-store";

// Mock daily-activity series for /admin Overview's 30-day trend chart.
// TWO modes are exposed:
//   - getDailyUploads30d()  → fully mocked, "platform health" framing
//   - getDailyDecisionsSeed30d() → seed series for daily decisions; the live
//     log from useAdminUserStore is merged in at render time so the chart
//     reflects the admin's real actions.
//
// All series end on TODAY (caller's clock) and span 30 entries inclusive.
// Backend will replace both with `GET /admin/metrics/daily?from=...&to=...`
// in Phase F.

export interface DailyMetric {
  date: string; // YYYY-MM-DD
  amount: number;
}

const DAYS = 30;

function todayIso(): string {
  const d = new Date();
  return formatDate(d);
}

function formatDate(d: Date): string {
  const y = d.getFullYear();
  const m = (d.getMonth() + 1).toString().padStart(2, "0");
  const day = d.getDate().toString().padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function dayOffset(iso: string, n: number): string {
  const d = new Date(iso + "T00:00:00");
  d.setDate(d.getDate() + n);
  return formatDate(d);
}

// Mulberry32 — deterministic seeded RNG so the mock series stays stable across
// re-renders (otherwise the chart would re-shuffle every paint).
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

// 30-day uploads series. Range 100–600. Weekend dip (Sat/Sun ~50% of weekday).
// Slight upward trend over the window.
export function getDailyUploads30d(): DailyMetric[] {
  const today = todayIso();
  const out: DailyMetric[] = [];
  const rng = mulberry32(0x517a9b);
  for (let i = DAYS - 1; i >= 0; i--) {
    const iso = dayOffset(today, -i);
    const d = new Date(iso + "T00:00:00");
    const dow = d.getDay(); // 0 = Sun, 6 = Sat
    const isWeekend = dow === 0 || dow === 6;
    const base = 240 + (DAYS - i) * 4; // gentle uptrend
    const noise = Math.floor(rng() * 220);
    const amount = Math.max(
      80,
      Math.round((base + noise) * (isWeekend ? 0.55 : 1)),
    );
    out.push({ date: iso, amount });
  }
  return out;
}

// 30-day seed for decisions. Range 0–4. Today + last 6 days deliberately set
// to 0 so live `log` entries fill them in; older days carry plausible historic
// counts so the chart never reads as "empty platform".
export function getDailyDecisionsSeed30d(): DailyMetric[] {
  const today = todayIso();
  const out: DailyMetric[] = [];
  const rng = mulberry32(0x91da42);
  for (let i = DAYS - 1; i >= 0; i--) {
    const iso = dayOffset(today, -i);
    const isRecent = i < 7; // last 7 days (incl. today) reserved for live log
    if (isRecent) {
      out.push({ date: iso, amount: 0 });
      continue;
    }
    // Older days: most days 0–2, occasional spike to 3–4.
    const r = rng();
    const amount = r < 0.45 ? 0 : r < 0.8 ? 1 + Math.floor(rng() * 2) : 3 + Math.floor(rng() * 2);
    out.push({ date: iso, amount });
  }
  return out;
}

// Merge live decision log entries into a 30-day seed. Each entry that falls
// inside the window adds 1 to its day's amount. Decisions older than the
// window are dropped silently.
export function mergeDecisionsWithLog(
  seed: ReadonlyArray<DailyMetric>,
  log: ReadonlyArray<DecisionLogEntry>,
): DailyMetric[] {
  if (seed.length === 0) return [];
  const byDate = new Map<string, number>();
  for (const day of seed) byDate.set(day.date, day.amount);
  for (const entry of log) {
    const iso = entry.decidedAt.slice(0, 10);
    if (!byDate.has(iso)) continue;
    byDate.set(iso, (byDate.get(iso) ?? 0) + 1);
  }
  return seed.map((day) => ({
    date: day.date,
    amount: byDate.get(day.date) ?? day.amount,
  }));
}

// Sum helper — used by the chart caption ("47 decisions · last 30 days").
export function totalAmount(series: ReadonlyArray<DailyMetric>): number {
  return series.reduce((sum, d) => sum + d.amount, 0);
}
