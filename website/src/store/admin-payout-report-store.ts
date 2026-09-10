import { create } from "zustand";
import { useToastStore } from "@/store/toast-store";
import {
  PAYOUT_REPORT_REASON_LABEL,
  type PayoutReport,
  type PayoutReportReason,
} from "@/lib/admin-payout-reports";
import {
  acknowledgePayoutReport as apiAcknowledgeReport,
  resolvePayoutReport as apiResolveReport,
} from "@/lib/api-admin";

// Admin decisions are mirrored locally only after backend confirmation.
// Photographer-side submitReport() is separate from these admin actions.
//
// acknowledge() and resolve() photographer-side notifications fire BE-side
// via AdminPayoutReportService → AdminDecisionLogService.pushMessage
// (kinds payout_report_acknowledged / payout_report_resolved). The
// photographer reads them via GET /me/photographer/messages.
// submitReport() is the inverse direction (photographer-side write); it
// does NOT push an inbox message — the photographer already saw the
// submission they themselves filed.

function toastBackendFailure(label: string, error: unknown): void {
  console.error(`[admin/reports] ${label} failed`, error);
  useToastStore.getState().showToast({
    kind: "error",
    message: `Couldn't ${label} payout report — please try again.`,
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
  acknowledge: (id: string, reply: string) => Promise<boolean>;
  resolve: (id: string, resolutionNote: string) => Promise<boolean>;
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

export const useAdminPayoutReportStore = create<AdminPayoutReportStoreState>(
  (set) => ({
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
    acknowledge: async (id, reply) => {
      try {
        await apiAcknowledgeReport(id, reply);
        const at = nowIso();
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
        return true;
      } catch (error) {
        toastBackendFailure("acknowledge", error);
        return false;
      }
    },
    resolve: async (id, resolutionNote) => {
      try {
        await apiResolveReport(id, resolutionNote);
        const at = nowIso();
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
        return true;
      } catch (error) {
        toastBackendFailure("resolve", error);
        return false;
      }
    },
    clear: () => set({ submissions: [], overrides: {}, log: [] }),
  }),
);
