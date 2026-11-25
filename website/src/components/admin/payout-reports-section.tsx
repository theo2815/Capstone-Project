"use client";

import { LoadError } from "@/components/ui/load-error";
import { Skeleton } from "@/components/ui/skeleton";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { useAdminPayoutById, useAdminPayouts, EMPTY_PAYOUTS } from "@/hooks/use-admin-payouts";
import { ADMIN_PAYOUT_STATUS_LABEL } from "@/lib/admin-payouts";
import { useAdminKpis } from "@/hooks/use-admin-data";
import { useQueueKeyboardNav } from "@/hooks/use-admin-keyboard";
import { Slab } from "@/components/profile-shell";
import { Kicker } from "@/components/ui/kicker";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { BTN_PRIMARY, BTN_SECONDARY, BTN_SIZE, ROW_OPEN_BUTTON } from "@/components/ui/button-styles";
import { AdminStatusTabs } from "@/components/admin/admin-status-tabs";
import { ROUTES } from "@/lib/constants";
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
  type PayoutReport,
  type PayoutReportStatus,
} from "@/lib/admin-payout-reports";
import { useAdminPayoutReportsPage, usePayoutReportActions } from "@/hooks/use-payout-reports";
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
// inbox message back to the photographer (handled by the backend).
//
// Drawer + action modal mirror the Disputes pattern: row click opens a
// focus-mode drawer; drawer footer has the status-conditional verbs;
// each verb opens a stacked modal that captures the textarea + commits.
//
// One status tab at a time (Open by default), paged by the server, newest first.

const STATUSES = Object.keys(PAYOUT_REPORT_STATUS_LABEL) as PayoutReportStatus[];
const parseStatus = (raw: string): PayoutReportStatus =>
  STATUSES.includes(raw as PayoutReportStatus) ? (raw as PayoutReportStatus) : "open";

// keyScope: the second list on its page (Payouts), so J/K/Enter work here only while focus is inside it.
export function PayoutReportsSection({ keyScope = false }: { keyScope?: boolean } = {}) {
  const { acknowledge, resolve, isPending, pendingAction } = usePayoutReportActions();
  const [status, setStatus] = useUrlState<PayoutReportStatus>("reportStatus", "open", { parse: parseStatus });
  const list = useAdminPayoutReportsPage(status);
  const reports = list.items;
  const serverPayouts = useAdminPayouts().data ?? EMPTY_PAYOUTS;
  const openCount = useAdminKpis()?.openPayoutReports;
  // FU-20: in the Inbox the tabs keep to one row and the legend follows the list, so the first report sits higher.
  const inInbox = usePathname() === ROUTES.ADMIN_INBOX;
  const { showToast } = useToast();

  const [reportRowId, setReportRowId] = useUrlState<string>("report", "");
  const submitting = isPending(reportRowId);
  const [actionMode, setActionMode] = useState<PayoutReportActionMode | null>(
    null,
  );

  const cyclesById = useMemo(() => {
    const merged = serverPayouts;
    const map = new Map<string, (typeof merged)[number]>();
    for (const c of merged) map.set(c.id, c);
    return map;
  }, [serverPayouts]);

  // The report the admin clicked: a refresh that drops it from this tab must not
  // close the drawer (and a half-typed reply) under them.
  const [picked, setPicked] = useState<PayoutReport | null>(null);
  const openReport = useMemo(
    () => (reportRowId
      ? reports.find((r) => r.id === reportRowId) ?? (picked?.id === reportRowId ? picked : null)
      : null),
    [reportRowId, reports, picked],
  );

  const scopeRef = useRef<HTMLDivElement>(null);
  const rowIds = useMemo(() => reports.map((r) => r.id), [reports]);
  const { focusedId, setFocusedId } = useQueueKeyboardNav({
    rowIds,
    disabled: !!reportRowId,
    onActivate: (id) => {
      setPicked(reports.find((r) => r.id === id) ?? null);
      setReportRowId(id);
    },
    scopeRef: keyScope ? scopeRef : undefined,
  });
  useEffect(() => { if (reportRowId) setFocusedId(reportRowId); }, [reportRowId, setFocusedId]);

  async function handleAcknowledge(note: string) {
    if (!openReport || submitting) return;
    const ok = await acknowledge(openReport.id, note);
    if (!ok) return;
    setActionMode(null);
    // The report has moved to the Acknowledged tab, so it leaves this list.
    setReportRowId("");
    showToast({
      kind: "success",
      message: "Reply sent. The report is now under Acknowledged.",
    });
  }

  async function handleResolve(note: string) {
    if (!openReport || submitting) return;
    const ok = await resolve(openReport.id, note);
    if (!ok) return;
    setActionMode(null);
    setReportRowId("");
    showToast({
      kind: "success",
      message: "Report resolved. The photographer was sent your note.",
    });
  }

  const legend = (
    <p className={`font-sans text-sm text-slate-soft ${inInbox ? "mt-6" : "mt-3"}`}>
      Open: not answered yet. Acknowledged: you replied and are still working on it. Resolved: closed.
    </p>
  );

  return (
    <Slab
      titleStyle="heading"
      id="payout-reports"
      number="06"
      title="Reports from photographers"
      caption="Filed against payouts they didn't receive or expect"
      trailing={openCount !== undefined ? <span className="tnum">{openCount} open</span> : undefined}
    >
      {/* Inside the Slab, so the Slab stays a sibling of the slabs before it (its first: styles). */}
      <div ref={scopeRef} data-kbd-scope={keyScope ? "" : undefined}>
      <AdminStatusTabs
        label="Report status"
        tabs={STATUSES.map((key) => ({
          key,
          label: PAYOUT_REPORT_STATUS_LABEL[key],
          count: key === "open" ? openCount : undefined,
        }))}
        value={status}
        onChange={setStatus}
        scroll={inInbox}
      />
      {!inInbox && legend}
      <div className="mt-6">
      {list.isLoading ? <div aria-busy="true" className="space-y-3">{Array.from({ length: 3 }, (_, i) => <Skeleton key={i} className="h-14 w-full rounded-xl" />)}</div> :
        list.error && reports.length === 0 ? (
          <LoadError what="payout reports" onRetry={list.refetch} className="text-sm text-error" />
        ) : reports.length === 0 ? (
        <p className="font-sans text-sm text-slate-soft">
          {status === "open"
            ? "No open reports. You're caught up."
            : `No ${PAYOUT_REPORT_STATUS_LABEL[status].toLowerCase()} reports.`}
        </p>
      ) : (
        <>
          <ul className="space-y-3">
            {reports.map((report) => (
              <li key={report.id}>
                <ReportRow
                  report={report}
                  focused={focusedId === report.id}
                  cycleAmount={cyclesById.get(report.payoutCycleId)?.amount}
                  onOpen={() => { setPicked(report); setReportRowId(report.id); }}
                />
              </li>
            ))}
          </ul>
          <LoadMoreButton
            shown={reports.length}
            total={list.total}
            increment={50}
            onLoadMore={list.fetchNextPage}
            isLoading={list.isFetchingNextPage}
          />
        </>
      )}
      </div>
      {inInbox && legend}

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
            <Kicker as="p" tone="soft" className="tnum break-words">
              {openReport.handle ? `@${openReport.handle}` : "—"}
              <span className="text-slate-soft"> · </span>
              {PAYOUT_REPORT_REASON_LABEL[openReport.reason]}
              <span className="text-slate-soft"> · </span>
              {formatLongDate(openReport.reportedAt)}
            </Kicker>
          }
          actions={
            <ReportActions
              report={openReport}
              submitting={submitting}
              onAcknowledge={() => setActionMode("acknowledge")}
              onResolve={() => setActionMode("resolve")}
            />
          }
        >
          {submitting && <p role="status">{pendingAction(reportRowId)}</p>}
          <ReportDetailBody
            report={openReport}
            listedAmount={cyclesById.get(openReport.payoutCycleId)?.amount}
          />
        </AdminDetailDrawer>
      )}

      {openReport && actionMode && (
        <PayoutReportActionModal
          open={true}
          submitting={submitting}
          mode={actionMode}
          report={openReport}
          onClose={() => setActionMode(null)}
          onSubmit={
            actionMode === "acknowledge" ? handleAcknowledge : handleResolve
          }
        />
      )}
      </div>
    </Slab>
  );
}

interface ReportRowProps {
  report: PayoutReport;
  cycleAmount: number | undefined;
  onOpen: () => void;
  focused: boolean;
}

function CycleLink({ cycleId }: { cycleId: string }) {
  return (
    <Link
      href={`${ROUTES.ADMIN_PAYOUTS}?row=${encodeURIComponent(cycleId)}`}
      className="relative z-10 font-mono tnum underline underline-offset-2 hover:text-ink"
    >
      {cycleId}
    </Link>
  );
}

// AX-4 row pattern: the title button covers the card; the cycle link sits above it.
function ReportRow({ report, cycleAmount, onOpen, focused }: ReportRowProps) {
  return (
    <article data-row-id={report.id} className={cn("relative rounded-2xl border border-line bg-bone hover:bg-bone-deep/60 transition-colors px-4 py-4 md:px-5 md:py-5", focused && "ring-2 ring-ink ring-offset-2 ring-offset-bone")}>
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
          <h3 className="font-display text-base md:text-lg text-ink mt-2">
            <button type="button" onClick={onOpen} aria-label={`${report.photographerName}, report ${report.id}`} className={ROW_OPEN_BUTTON}>
              {report.photographerName}
            </button>
            {/* The space lets the handle wrap as a whole word after the name button. */}
            {report.handle && " "}
            {report.handle && (
              <Kicker as="span" tone="soft" className="ml-1 break-words">
                @{report.handle}
              </Kicker>
            )}
          </h3>
          <p className="font-sans text-sm text-ink-soft mt-1.5 break-words">
            <span className="text-ink">
              {PAYOUT_REPORT_REASON_LABEL[report.reason]}
            </span>
            <span className="text-slate-soft"> · </span>
            <CycleLink cycleId={report.payoutCycleId} />
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
    </article>
  );
}

interface ReportActionsProps {
  report: PayoutReport;
  submitting: boolean;
  onAcknowledge: () => void;
  onResolve: () => void;
}

function ReportActions({ report, submitting, onAcknowledge, onResolve }: ReportActionsProps) {
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
          disabled={submitting}
          className={cn(BTN_SECONDARY, BTN_SIZE.sm)}
        >
          Acknowledge…
        </button>
      )}
      <button
        type="button"
        onClick={onResolve}
        disabled={submitting}
        className={cn(BTN_PRIMARY, BTN_SIZE.sm)}
      >
        Resolve…
      </button>
    </>
  );
}

interface ReportDetailBodyProps {
  report: PayoutReport;
  listedAmount: number | undefined;
}

function ReportDetailBody({ report, listedAmount }: ReportDetailBodyProps) {
  // C-33, C-34: the list holds the newest 200 payouts only, so the open report reads its own by id.
  const payout = useAdminPayoutById(report.payoutCycleId).data;
  const cycleAmount = payout?.amount ?? listedAmount;
  return (
    <div className="space-y-10">
      <section>
        <Kicker as="p" tone="soft" className="mb-3">
          Payout
        </Kicker>
        <div className="rounded-2xl border border-line bg-bone-deep p-4">
          <p className="text-[14px] text-ink break-words">
            <CycleLink cycleId={report.payoutCycleId} />
          </p>
          {cycleAmount !== undefined && (
            <Kicker as="p" tone="soft" tnum className="mt-2">
              Amount · {formatPrice(cycleAmount)}
            </Kicker>
          )}
          {payout && (
            <Kicker as="p" tone="soft" className="mt-2">
              Status · {ADMIN_PAYOUT_STATUS_LABEL[payout.status] ?? payout.status}
            </Kicker>
          )}
          {payout?.paidAt && (
            <Kicker as="p" tone="soft" tnum className="mt-2">
              Paid on · {formatLongDate(payout.paidAt)}
            </Kicker>
          )}
          {payout?.paymentReference && (
            <Kicker as="p" tone="soft" tnum className="mt-2 break-all">
              Reference · {payout.paymentReference}
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
