"use client";

import { useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Slab } from "@/components/profile-shell";
import { Kicker } from "@/components/ui/kicker";
import {
  AdminStatusPill,
  type AdminStatusPillTone,
} from "@/components/admin/admin-status-pill";
import { AdminDetailDrawer } from "@/components/admin/admin-detail-drawer";
import {
  PayoutReportActionModal,
  type PayoutReportActionMode,
} from "@/components/admin/payout-report-action-modal";
import {
  PAYOUT_REPORT_REASON_LABEL,
  PAYOUT_REPORT_STATUS_LABEL,
  mergeReportsWithOverrides,
  type PayoutReport,
  type PayoutReportStatus,
} from "@/lib/admin-payout-reports";
import { useAdminPayoutReportStore } from "@/store/admin-payout-report-store";
import {
  mergePayoutsWithOverrides,
  useAdminPayoutStore,
} from "@/store/admin-payout-store";
import {
  useAdminPayoutReports,
  useAdminPayouts,
} from "@/hooks/use-admin-data";
import { useToast } from "@/hooks/use-toast";
import { useUrlState } from "@/hooks/use-url-state";
import { formatLongDate } from "@/lib/format";
import { formatPrice } from "@/lib/utils";
import { cn } from "@/lib/utils";

function statusTone(status: PayoutReportStatus): AdminStatusPillTone {
  switch (status) {
    case "open":
      return "amber";
    case "acknowledged":
      return "ink";
    case "resolved":
      return "fresh";
  }
}

// /admin/payouts second section — runs below the existing PayoutsQueue and
// surfaces photographer-filed reports about specific cycles. State machine
// is open → acknowledged → resolved; admin actions on a report fire an
// inbox message back to the photographer (handled inside the store).
//
// Drawer + action modal mirror the Disputes pattern: row click opens a
// focus-mode drawer; drawer footer has the status-conditional verbs;
// each verb opens a stacked modal that captures the textarea + commits.
//
// Reports default-sorted: open first, then acknowledged, then resolved;
// within each bucket, newest reportedAt first.

export function PayoutReportsSection() {
  // A-1 followup for reports: read from BE. The local submissions array in
  // Zustand only ever contains reports filed in *this* browser session
  // (photographer-side), so a fresh admin tab saw "No reports yet" even
  // when reports existed on the server. overrides still layer for instant
  // feedback after acknowledge/resolve.
  const overrides = useAdminPayoutReportStore((s) => s.overrides);
  const acknowledge = useAdminPayoutReportStore((s) => s.acknowledge);
  const resolve = useAdminPayoutReportStore((s) => s.resolve);
  const payoutOverrides = useAdminPayoutStore((s) => s.overrides);
  const serverReports = useAdminPayoutReports() ?? [];
  const serverPayouts = useAdminPayouts() ?? [];
  const queryClient = useQueryClient();
  const { showToast } = useToast();

  function refreshAfterAction() {
    // Local override already gives instant feedback; this kicks a refetch so
    // BE state becomes the source of truth (and the photographer's
    // /dashboard/billing reports cache, if mounted, also catches up).
    queryClient.invalidateQueries({
      queryKey: ["admin", "payouts", "reports"],
    });
    queryClient.invalidateQueries({
      queryKey: ["photographer", "payouts", "reports"],
    });
  }

  const [reportRowId, setReportRowId] = useUrlState<string>("report", "");
  const [actionMode, setActionMode] = useState<PayoutReportActionMode | null>(
    null,
  );

  const reports = useMemo(() => {
    const all = mergeReportsWithOverrides(serverReports, overrides);
    const order: Record<PayoutReportStatus, number> = {
      open: 0,
      acknowledged: 1,
      resolved: 2,
    };
    return [...all].sort((a, b) => {
      const so = order[a.status] - order[b.status];
      if (so !== 0) return so;
      return b.reportedAt.localeCompare(a.reportedAt);
    });
  }, [serverReports, overrides]);

  const openCount = reports.filter((r) => r.status === "open").length;

  const cyclesById = useMemo(() => {
    const merged = mergePayoutsWithOverrides(serverPayouts, payoutOverrides);
    const map = new Map<string, (typeof merged)[number]>();
    for (const c of merged) map.set(c.id, c);
    return map;
  }, [serverPayouts, payoutOverrides]);

  const openReport = useMemo(
    () => (reportRowId ? reports.find((r) => r.id === reportRowId) ?? null : null),
    [reportRowId, reports],
  );

  function handleAcknowledge(note: string) {
    if (!openReport) return;
    acknowledge(openReport.id, note);
    refreshAfterAction();
    setActionMode(null);
    showToast({
      kind: "success",
      message: `Acknowledged · ${openReport.id}`,
    });
  }

  function handleResolve(note: string) {
    if (!openReport) return;
    resolve(openReport.id, note);
    refreshAfterAction();
    setActionMode(null);
    setReportRowId("");
    showToast({
      kind: "success",
      message: `Resolved · ${openReport.id}`,
    });
  }

  return (
    <Slab
      id="reports"
      number="05"
      title="Reports from photographers"
      caption="Filed against payouts they didn't receive or expect"
      trailing={
        openCount > 0
          ? `${openCount} open · ${reports.length} total`
          : `${reports.length} total`
      }
    >
      {reports.length === 0 ? (
        <p className="font-sans text-sm text-slate-soft">
          No photographer reports yet.
        </p>
      ) : (
        <ul className="space-y-3">
          {reports.map((report) => (
            <li key={report.id}>
              <ReportRow
                report={report}
                cycleAmount={cyclesById.get(report.payoutCycleId)?.amount}
                onOpen={() => setReportRowId(report.id)}
              />
            </li>
          ))}
        </ul>
      )}

      {openReport && (
        <AdminDetailDrawer
          open={true}
          onClose={() => setReportRowId("")}
          escDisabled={actionMode !== null}
          kicker={`Report · ${openReport.id}`}
          title={openReport.photographerName}
          rightHeader={
            <AdminStatusPill
              label={PAYOUT_REPORT_STATUS_LABEL[openReport.status]}
              tone={statusTone(openReport.status)}
            />
          }
          subtitle={
            <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft tnum">
              {openReport.handle ? `@${openReport.handle}` : "—"}
              <span className="text-slate-soft"> · </span>
              {PAYOUT_REPORT_REASON_LABEL[openReport.reason]}
              <span className="text-slate-soft"> · </span>
              {formatLongDate(openReport.reportedAt)}
            </p>
          }
          actions={
            <ReportActions
              report={openReport}
              onAcknowledge={() => setActionMode("acknowledge")}
              onResolve={() => setActionMode("resolve")}
            />
          }
        >
          <ReportDetailBody
            report={openReport}
            cycleAmount={cyclesById.get(openReport.payoutCycleId)?.amount}
          />
        </AdminDetailDrawer>
      )}

      {openReport && actionMode && (
        <PayoutReportActionModal
          open={true}
          mode={actionMode}
          report={openReport}
          onClose={() => setActionMode(null)}
          onSubmit={
            actionMode === "acknowledge" ? handleAcknowledge : handleResolve
          }
        />
      )}
    </Slab>
  );
}

interface ReportRowProps {
  report: PayoutReport;
  cycleAmount: number | undefined;
  onOpen: () => void;
}

function ReportRow({ report, cycleAmount, onOpen }: ReportRowProps) {
  return (
    <button
      type="button"
      onClick={onOpen}
      className="w-full text-left rounded-2xl border border-line bg-bone hover:bg-bone-deep/60 transition-colors px-4 py-4 md:px-5 md:py-5"
    >
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-3 flex-wrap">
            <AdminStatusPill
              label={PAYOUT_REPORT_STATUS_LABEL[report.status]}
              tone={statusTone(report.status)}
            />
            <Kicker tone="soft" tnum>
              {report.id}
            </Kicker>
          </div>
          <p className="font-display text-base md:text-lg text-ink mt-2">
            {report.photographerName}
            {report.handle && (
              <span className="font-mono uppercase tracking-[0.22em] text-[12px] text-slate-soft ml-2">
                @{report.handle}
              </span>
            )}
          </p>
          <p className="font-sans text-sm text-ink-soft mt-1.5">
            <span className="text-ink">
              {PAYOUT_REPORT_REASON_LABEL[report.reason]}
            </span>
            <span className="text-slate-soft"> · </span>
            <span className="font-mono tnum">{report.payoutCycleId}</span>
            {cycleAmount !== undefined && (
              <>
                <span className="text-slate-soft"> · </span>
                <span className="font-mono tnum">{formatPrice(cycleAmount)}</span>
              </>
            )}
          </p>
          {report.note && (
            <p className="font-sans text-sm text-slate mt-2 line-clamp-2">
              {report.note}
            </p>
          )}
        </div>
        <Kicker tone="soft" tnum className="shrink-0">
          {formatLongDate(report.reportedAt)}
        </Kicker>
      </div>
    </button>
  );
}

interface ReportActionsProps {
  report: PayoutReport;
  onAcknowledge: () => void;
  onResolve: () => void;
}

function ReportActions({ report, onAcknowledge, onResolve }: ReportActionsProps) {
  if (report.status === "resolved") {
    return (
      <p className="font-sans text-sm text-slate-soft">
        Resolved. No further actions.
      </p>
    );
  }
  return (
    <>
      {report.status === "open" && (
        <button
          type="button"
          onClick={onAcknowledge}
          className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate hover:text-ink transition-colors px-4 py-2"
        >
          Acknowledge…
        </button>
      )}
      <button
        type="button"
        onClick={onResolve}
        className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-ink border border-line hover:bg-ink hover:text-bone hover:border-ink transition-colors rounded-full px-5 py-2"
      >
        Resolve…
      </button>
    </>
  );
}

interface ReportDetailBodyProps {
  report: PayoutReport;
  cycleAmount: number | undefined;
}

function ReportDetailBody({ report, cycleAmount }: ReportDetailBodyProps) {
  return (
    <div className="space-y-10">
      <section>
        <Kicker as="p" tone="soft" className="mb-3">
          Cycle
        </Kicker>
        <div className="rounded-2xl border border-line bg-bone-deep p-4">
          <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-ink tnum">
            {report.payoutCycleId}
          </p>
          {cycleAmount !== undefined && (
            <Kicker as="p" tone="soft" tnum className="mt-2">
              Amount · {formatPrice(cycleAmount)}
            </Kicker>
          )}
        </div>
      </section>

      <section>
        <Kicker as="p" tone="soft" className="mb-3">
          Reason
        </Kicker>
        <p className="font-sans text-base text-ink">
          {PAYOUT_REPORT_REASON_LABEL[report.reason]}
        </p>
        {report.note && (
          <p className="font-sans text-sm text-ink-soft mt-3 leading-relaxed">
            {report.note}
          </p>
        )}
      </section>

      <section>
        <Kicker as="p" tone="soft" className="mb-3">
          Timeline
        </Kicker>
        <ol className="space-y-3">
          <TimelineRow
            label="Filed"
            at={report.reportedAt}
            note={`By ${report.photographerName}`}
            active
          />
          <TimelineRow
            label="Acknowledged"
            at={report.acknowledgedAt}
            note={report.acknowledgeReply ?? null}
            active={report.status !== "open"}
          />
          <TimelineRow
            label="Resolved"
            at={report.resolvedAt}
            note={report.resolutionNote ?? null}
            active={report.status === "resolved"}
          />
        </ol>
      </section>
    </div>
  );
}

interface TimelineRowProps {
  label: string;
  at: string | null;
  note: string | null;
  active: boolean;
}

function TimelineRow({ label, at, note, active }: TimelineRowProps) {
  return (
    <li
      className={cn(
        "rounded-xl border px-4 py-3",
        active ? "border-line bg-bone-deep" : "border-line/60 bg-bone",
      )}
    >
      <div className="flex items-baseline justify-between gap-3">
        <Kicker tone={active ? "default" : "soft"}>{label}</Kicker>
        {at && (
          <Kicker as="time" tone="soft" tnum>
            {formatLongDate(at)}
          </Kicker>
        )}
      </div>
      {note ? (
        <p
          className={cn(
            "font-sans text-sm mt-2 leading-relaxed",
            active ? "text-ink-soft" : "text-slate-soft",
          )}
        >
          {note}
        </p>
      ) : !at ? (
        <p className="font-sans text-sm text-slate-soft mt-2">Pending.</p>
      ) : null}
    </li>
  );
}
