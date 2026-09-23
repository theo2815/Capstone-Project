
// Daily-activity series for /admin Overview's 30-day trend chart.
// Backend serves both via `/admin/kpis/trend?days=30`. These helpers return
// zero-filled 30-entry skeletons that the live trend hook can overlay onto
// until the queue-listing hooks finish wiring the live decision log path.

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

function emptySeries(): DailyMetric[] {
  const today = todayIso();
  const out: DailyMetric[] = [];
  for (let i = DAYS - 1; i >= 0; i--) {
    out.push({ date: dayOffset(today, -i), amount: 0 });
  }
  return out;
}

// 30-day uploads series. Live data lives in the backend trend endpoint;
// this returns a zero-filled skeleton until callers swap to the hook.
export function getDailyUploads30d(): DailyMetric[] {
  return emptySeries();
}

// Zero-filled chart skeleton while the backend trend loads.
export function getDailyDecisionsSeed30d(): DailyMetric[] {
  return emptySeries();
}

// Sum helper — used by the chart caption.
export function totalAmount(series: ReadonlyArray<DailyMetric>): number {
  return series.reduce((sum, d) => sum + d.amount, 0);
}
