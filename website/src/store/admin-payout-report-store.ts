import { create } from "zustand";
import {
  ADMIN_PAYOUT_REPORTS,
  PAYOUT_REPORT_REASON_LABEL,
  type PayoutReport,
  type PayoutReportReason,
} from "@/lib/admin-payout-reports";
import { usePhotographerMessageStore } from "@/store/photographer-message-store";
import { BACKEND_LIVE } from "@/lib/backend-flag";
import {
  acknowledgePayoutReport as apiAcknowledgeReport,
  resolvePayoutReport as apiResolveReport,
} from "@/lib/api-admin";

// Mock-mode store of admin actions on the payout-reports section. Phase G
// wiring fires the matching `api-admin` PATCH in the background after the
// local override is applied — Q-A2 RESOLVED 2026-05-09 means the backend
// writes the photographer-message insert in the same TX. Photographer-side
// submitReport() is photographer-scope; admin actions land here.
//
// acknowledge() and resolve() each fire a side-effect into the
// photographer-message-store so the photographer sees a new inbox row
// reflecting the admin's action. submitReport() is the inverse direction
// (photographer-side write); it does NOT fire an inbox message — the
// photographer already saw the report submission.

function fireBackendReportAction(label: string, p: Promise<unknown>): void {
  if (!BACKEND_LIVE) return;
  void p.catch((err) => {
    console.error(`[admin/reports] ${label} backend call failed`, err);
  });
}

export type PayoutReportDecision = "submitted" | "acknowledged" | "resolved";

export interface PayoutReportLogEntry {
  reportId: string;
  decision: PayoutReportDecision;
  note: string | null;
  decidedAt: string;
}

export interface SubmitReportPayload {
  payoutCycleId: string;
  photographerId: string;
  photographerName: string;
  handle: string | null;
  reason: PayoutReportReason;
  note: string;
}

interface AdminPayoutReportStoreState {
  submissions: PayoutReport[];
  overrides: Record<string, Partial<PayoutReport>>;
  log: PayoutReportLogEntry[];
  submitReport: (payload: SubmitReportPayload) => PayoutReport;
  acknowledge: (id: string, reply: string) => void;
  resolve: (id: string, resolutionNote: string) => void;
  clear: () => void;
}

function makeReportId(): string {
  const slug = Math.random().toString(36).slice(2, 8).toUpperCase();
  return `RPT-RUN-${slug}`;
}

function nowIso(): string {
  return new Date().toISOString();
}

function appendLog(
  prev: PayoutReportLogEntry[],
  entry: PayoutReportLogEntry,
): PayoutReportLogEntry[] {
  return [entry, ...prev].slice(0, 50);
}

// Look up enough metadata to fill the inbox-message body when admin acts on
// a report. We call useAdminPayoutReportStore.getState() ourselves rather
// than passing the full report through the action — keeps the API ergonomic
// (admin UI passes id + reply text only).
function findReport(
  submissions: ReadonlyArray<PayoutReport>,
  overrides: Record<string, Partial<PayoutReport>>,
  id: string,
): PayoutReport | undefined {
  const submitted = submissions.find((r) => r.id === id);
  if (submitted) return { ...submitted, ...overrides[id] };
  const seed = ADMIN_PAYOUT_REPORTS.find((r) => r.id === id);
  if (!seed) return undefined;
  return { ...seed, ...overrides[id] };
}

export const useAdminPayoutReportStore = create<AdminPayoutReportStoreState>(
  (set, get) => ({
    submissions: [],
    overrides: {},
    log: [],
    submitReport: (payload) => {
      const at = nowIso();
      const report: PayoutReport = {
        id: makeReportId(),
        payoutCycleId: payload.payoutCycleId,
        photographerId: payload.photographerId,
        photographerName: payload.photographerName,
        handle: payload.handle,
        reason: payload.reason,
        note: payload.note,
        status: "open",
        reportedAt: at,
        acknowledgedAt: null,
        acknowledgeReply: null,
        resolvedAt: null,
        resolutionNote: null,
      };
      set((s) => ({
        submissions: [report, ...s.submissions],
        log: appendLog(s.log, {
          reportId: report.id,
          decision: "submitted",
          note: PAYOUT_REPORT_REASON_LABEL[payload.reason],
          decidedAt: at,
        }),
      }));
      return report;
    },
    acknowledge: (id, reply) => {
      const at = nowIso();
      const report = findReport(get().submissions, get().overrides, id);
      set((s) => ({
        overrides: {
          ...s.overrides,
          [id]: {
            ...s.overrides[id],
            status: "acknowledged",
            acknowledgedAt: at,
            acknowledgeReply: reply,
          },
        },
        log: appendLog(s.log, {
          reportId: id,
          decision: "acknowledged",
          note: reply,
          decidedAt: at,
        }),
      }));
      if (report) {
        usePhotographerMessageStore.getState().addMessage({
          photographerId: report.photographerId,
          kind: "report_acknowledged",
          title: "Admin acknowledged your report",
          body: reply,
          payoutCycleId: report.payoutCycleId,
          reportId: id,
        });
      }
      fireBackendReportAction("acknowledge", apiAcknowledgeReport(id, reply));
    },
    resolve: (id, resolutionNote) => {
      const at = nowIso();
      const report = findReport(get().submissions, get().overrides, id);
      set((s) => ({
        overrides: {
          ...s.overrides,
          [id]: {
            ...s.overrides[id],
            status: "resolved",
            resolvedAt: at,
            resolutionNote,
          },
        },
        log: appendLog(s.log, {
          reportId: id,
          decision: "resolved",
          note: resolutionNote,
          decidedAt: at,
        }),
      }));
      if (report) {
        usePhotographerMessageStore.getState().addMessage({
          photographerId: report.photographerId,
          kind: "report_resolved",
          title: "Your report was resolved",
          body: resolutionNote,
          payoutCycleId: report.payoutCycleId,
          reportId: id,
        });
      }
      fireBackendReportAction(
        "resolve",
        apiResolveReport(id, resolutionNote),
      );
    },
    clear: () => set({ submissions: [], overrides: {}, log: [] }),
  }),
);
