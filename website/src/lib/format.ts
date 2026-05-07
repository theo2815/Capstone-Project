// Date formatters shared across profile-shell pages (/profile, /account,
// /orders, /dashboard). All return "—" for invalid input so renderers don't
// have to null-check. Accepts null/undefined defensively since backend
// payloads can ship partial data.

type MaybeIso = string | null | undefined;

export function formatMonthYear(iso: MaybeIso): string {
  if (typeof iso !== "string" || !iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  const month = d.toLocaleString("en-US", { month: "short" }).toUpperCase();
  const year = d.getFullYear();
  return `${month} ${year}`;
}

// Long form: "APR 28 · 2026". Pass `dateOnly = true` for inputs without a
// time component (YYYY-MM-DD) so the local-tz parse doesn't shift the day.
export function formatLongDate(iso: MaybeIso, dateOnly = false): string {
  if (typeof iso !== "string" || !iso) return "—";
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
export const formatPaidAt = (iso: MaybeIso) => formatLongDate(iso, false);
export const formatRaceDate = (iso: MaybeIso) => formatLongDate(iso, true);
