import { api } from "@/lib/api";

// Report routes (AA-R9), signed-in only, note 10-500 characters:
//   POST /api/v1/me/reports/events/{eventId}        { note } → ReportResult
//   POST /api/v1/me/reports/photographers/{handle}  { note } → ReportResult
// alreadyReported: this account already has an open report on the target.

export interface ReportResult {
  reported: boolean;
  alreadyReported: boolean;
}

export type ReportTarget =
  | { kind: "event"; id: string }
  | { kind: "photographer"; handle: string };

export function sendReport(target: ReportTarget, note: string): Promise<ReportResult> {
  const path =
    target.kind === "event"
      ? `/me/reports/events/${encodeURIComponent(target.id)}`
      : `/me/reports/photographers/${encodeURIComponent(target.handle)}`;
  return api.post<ReportResult>(path, { note });
}
