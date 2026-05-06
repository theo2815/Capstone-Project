import type { PayoutMethod } from "@/store/photographer-settings-store";

// Display formatter for stored digits-only payout numbers.
// GCash/Maya: 11-digit PH mobile -> "09XX XXX XXXX".
// GoTyme: 16-digit bank account -> 4-digit groups.
//
// Mirrors the input-time formatter on /dashboard/settings; lifted from there
// so the billing page hero + How-payouts modal can render the same shape.
export function formatPayoutNumber(method: PayoutMethod, raw: string): string {
  const d = raw.replace(/\D/g, "");
  if (method === "gotyme") {
    const trimmed = d.slice(0, 16);
    return trimmed.replace(/(.{4})(?=.)/g, "$1 ").trim();
  }
  const trimmed = d.slice(0, 11);
  if (trimmed.length <= 4) return trimmed;
  if (trimmed.length <= 7)
    return `${trimmed.slice(0, 4)} ${trimmed.slice(4)}`;
  return `${trimmed.slice(0, 4)} ${trimmed.slice(4, 7)} ${trimmed.slice(7)}`;
}
