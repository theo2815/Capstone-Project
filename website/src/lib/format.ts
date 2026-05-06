// Date formatters shared across profile-shell pages (/profile, /account,
// /orders, /dashboard). All return "—" for invalid input so renderers don't
// have to null-check.

export function formatMonthYear(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  const month = d.toLocaleString("en-US", { month: "short" }).toUpperCase();
  const year = d.getFullYear();
  return `${month} ${year}`;
}

// Long form: "APR 28 · 2026". Pass `dateOnly = true` for inputs without a
// time component (YYYY-MM-DD) so the local-tz parse doesn't shift the day.
export function formatLongDate(iso: string, dateOnly = false): string {
  const d = new Date(dateOnly ? `${iso}T00:00:00` : iso);
  if (Number.isNaN(d.getTime())) return "—";
  const month = d.toLocaleString("en-US", { month: "short" }).toUpperCase();
  const day = d.getDate().toString().padStart(2, "0");
  const year = d.getFullYear();
  return `${month} ${day} · ${year}`;
}

// Convenience wrappers that name the call-site intent. They all flow through
// formatLongDate / formatMonthYear so behavior stays consistent.
export const formatMemberSince = formatMonthYear;
export const formatPaidAt = (iso: string) => formatLongDate(iso, false);
export const formatRaceDate = (iso: string) => formatLongDate(iso, true);
