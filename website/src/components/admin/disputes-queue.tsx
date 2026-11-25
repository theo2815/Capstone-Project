"use client";

import { LoadError } from "@/components/ui/load-error";
import { Skeleton } from "@/components/ui/skeleton";
import { useEffect, useMemo, useRef, useState, useSyncExternalStore, type ReactNode } from "react";
import { Kicker } from "@/components/ui/kicker";
import { BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { AdminDisputeCard } from "@/components/admin/admin-dispute-card";
import { RefundRun } from "@/components/admin/refund-run";
import { AdminDetailDrawer } from "@/components/admin/admin-detail-drawer";
import {
  PhotoPreviewCard,
  type PhotoPreviewItem,
} from "@/components/photos/photo-preview-card";
import { AdminStatusPill } from "@/components/admin/admin-status-pill";
import { AdminStatusTabs } from "@/components/admin/admin-status-tabs";
import { AdminDisputeActivityList } from "@/components/admin/admin-dispute-activity-list";
import { AdminDisputeActionAside } from "@/components/admin/admin-dispute-action-aside";
import { useAdminDispute, useAdminDisputePage } from "@/hooks/use-admin-disputes";
import { useAdminKpis } from "@/hooks/use-admin-data";
import { useUrlState } from "@/hooks/use-url-state";
import { useQueueKeyboardNav } from "@/hooks/use-admin-keyboard";
import {
  type Dispute,
  type DisputeResolution,
  DISPUTE_PAYMENT_LABEL,
  DISPUTE_REASON_LABEL,
  DISPUTE_RESOLUTION_LABEL,
  DISPUTE_STATUS_LABEL,
  disputeStatusTone,
} from "@/lib/admin-disputes";
import type { AdminDisputeView } from "@/lib/api-admin";
import { LogoPlaceholder } from "@/components/ui/logo-placeholder";
import { formatLongDate, sentenceCaseDate } from "@/lib/format";
import { formatPrice } from "@/lib/utils";

// The server's views (no status filter in the UI on purpose; the server keeps `status=`).
type QueueView = Extract<AdminDisputeView, "needs_me" | "overdue" | "refund_run" | "closed" | "all">;
const VIEWS: readonly { value: QueueView; label: string; empty: string }[] = [
  { value: "needs_me", label: "Needs you", empty: "No disputes waiting. You're caught up." },
  { value: "overdue", label: "Overdue", empty: "Nothing is overdue." },
  { value: "refund_run", label: "Refund run", empty: "No approved refunds are waiting to be paid." },
  { value: "closed", label: "Closed", empty: "No closed disputes yet." },
  { value: "all", label: "All", empty: "No disputes yet." },
];
const parseView = (raw: string): QueueView =>
  VIEWS.some((v) => v.value === raw) ? (raw as QueueView) : "needs_me";
const SEARCH_DEBOUNCE_MS = 300;
// The server's page size, for the Load more count.
const PAGE_INCREMENT = 50;

// Phones (below md) get the actions inside the drawer's scroll, not a footer that takes half the screen (DSP7);
// their buttons stay in view in a row stuck to the bottom of that scroll.
// shortcut: crossing 768 px remounts the aside and the drawer body, closing an open dispute modal or photo preview; fine for rotate and resize.
const MD_UP = "(min-width: 768px)";
const subscribeMdUp = (onChange: () => void) => {
  const mq = window.matchMedia?.(MD_UP);
  mq?.addEventListener("change", onChange);
  return () => mq?.removeEventListener("change", onChange);
};
const useMdUp = () => useSyncExternalStore(subscribeMdUp, () => window.matchMedia?.(MD_UP).matches ?? true, () => true);

// Disputes queue body, shared by /admin/disputes and /admin/inbox. The
// server sorts and pages each view ("Needs you" is due time first); the
// header stays on the dedicated page.
//
// Card click → drawer with full detail (claim, photo, parties, activity
// log) and the shared <AdminDisputeActionAside> in the footer. URL state
// via ?row=<disputeId>; the old /admin/disputes/[id] route redirects here.
// The aside owns its modals and reports when one is open, so the drawer's
// ESC stands down.

export function DisputesQueue() {
  const [view, setView] = useUrlState<QueueView>("view", "needs_me", { parse: parseView });
  const [q, setQ] = useUrlState<string>("q", "");
  // FU-18: a runner's "Their disputes" link narrows the list to that runner, until the admin searches for something else.
  const [runner, setRunner] = useUrlState<string>("runner", "");
  // The box holds what is typed; the URL (and the request) get it trimmed, after a pause.
  const [qDraft, setQDraft] = useState(q);
  const [qSeen, setQSeen] = useState(q);
  if (q !== qSeen) {
    // q changed from outside (back button, a link): show it in the box.
    setQSeen(q);
    if (q !== qDraft.trim()) setQDraft(q);
  }
  useEffect(() => {
    const next = qDraft.trim();
    if (next === q) return;
    const timer = setTimeout(() => {
      if (runner) setRunner("");
      setQ(next);
    }, SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [qDraft, q, setQ, runner, setRunner]);

  // Each view and search is its own list, so a new one starts at the first page.
  const list = useAdminDisputePage({ view, q: view === "refund_run" ? undefined : q || undefined, runner: view === "refund_run" ? undefined : runner || undefined });
  const items = list.items;
  const mdUp = useMdUp();
  // Counts come from the polled KPIs; a field an older backend leaves out shows no count.
  const kpis = useAdminKpis();
  const counts: Partial<Record<QueueView, number>> = {
    needs_me: kpis?.disputesNeedingAdmin ?? kpis?.openDisputes,
    overdue: kpis?.disputesOverdue,
  };

  const [rowId, setRowId] = useUrlState<string>("row", "");
  // A decision can return after the admin opened another dispute, or left the queue; only the decided one's drawer closes.
  const rowIdRef = useRef(rowId);
  useEffect(() => {
    rowIdRef.current = rowId;
    return () => {
      rowIdRef.current = "";
    };
  }, [rowId]);
  // Any aside dialog (resolve, refund confirm, deny, escalate, refund panel) is open; the drawer must ignore Esc.
  const [asideDialogOpen, setAsideDialogOpen] = useState(false);

  const byId = useMemo(() => {
    const map = new Map<string, Dispute>();
    for (const d of items) map.set(d.id, d);
    return map;
  }, [items]);

  // A ?row= link to a dispute outside the loaded list is fetched by id.
  const fallbackId = !list.isLoading && !list.error && rowId && !byId.has(rowId) ? rowId : "";
  const fallback = useAdminDispute(fallbackId);

  const openRow = useMemo<Dispute | null>(() => {
    if (!rowId) return null;
    return byId.get(rowId) ?? (fallbackId ? fallback.data ?? null : null);
  }, [rowId, byId, fallbackId, fallback.data]);

  // Close only when the server says the dispute does not exist.
  useEffect(() => {
    if (fallbackId && fallback.isSuccess && fallback.data === null) setRowId("");
  }, [fallbackId, fallback.isSuccess, fallback.data, setRowId]);

  const drawerEscDisabled = asideDialogOpen;

  // Keyboard nav walks the rows on this page, in server order.
  // Resolve/Deny/Escalate aren't part of the E/R/H/S verb whitelist, so
  // the drawer doesn't register verbs — keystrokes inside the drawer fall
  // through to ESC + the click-only footer buttons.
  const rowIds = useMemo(() => items.map((d) => d.id), [items]);
  const queueNavDisabled = openRow !== null || drawerEscDisabled;
  const { focusedId, setFocusedId } = useQueueKeyboardNav({
    rowIds,
    disabled: queueNavDisabled,
    onActivate: (id) => {
      setFocusedId(id);
      setRowId(id);
    },
  });

  useEffect(() => {
    if (rowId) setFocusedId(rowId);
  }, [rowId, setFocusedId]);

  function handleOpenRow(id: string) {
    setFocusedId(id);
    setRowId(id);
  }

  const emptyText = q
    ? `No disputes match “${q}”.`
    : VIEWS.find((v) => v.value === view)?.empty;

  // On phones the aside wraps the drawer body, so its buttons can stay stuck to the bottom of the scroll.
  const actionsAside = (body?: ReactNode) => openRow && (
    <AdminDisputeActionAside
      key={openRow.id}
      dispute={openRow}
      runnerHandle={openRow.runnerHandle}
      onDialogOpenChange={setAsideDialogOpen}
      onDecided={() => {
        if (rowIdRef.current === openRow.id) setRowId("");
      }}
      className={body ? "mb-10" : ""}
      body={body}
    />
  );

  return (
    <>
      <AdminStatusTabs
        label="Dispute views"
        tabs={VIEWS.map((v) => ({ key: v.value, label: v.label, count: counts[v.value] }))}
        value={view}
        onChange={setView}
      />
      <div className="pt-6 md:pt-8 space-y-6" aria-busy={list.isFetching}>
        <p className="font-sans text-sm text-slate">
          Needs you: waiting for your decision. Overdue: past the due date on the dispute. Refund run: approved QR Ph
          refunds you still have to pay in PayMongo.
        </p>
        {list.error != null && items.length > 0 && (
          <div role="alert" className="flex flex-wrap items-center gap-3">
            <p className="font-sans text-sm text-error">Couldn&apos;t refresh.</p>
            <button type="button" className={`${BTN_SECONDARY} ${BTN_SIZE.sm}`} onClick={list.refetch}>
              Retry
            </button>
          </div>
        )}
        {view !== "refund_run" && (
          <div className="lg:w-80">
            <label htmlFor="dispute-search" className="sr-only">
              Search disputes
            </label>
            <input
              id="dispute-search"
              type="search"
              value={qDraft}
              onChange={(e) => setQDraft(e.target.value)}
              placeholder="Runner, photographer, event, or an ID"
              className="w-full rounded-2xl border border-line bg-surface px-5 py-3 font-sans text-sm text-ink placeholder:text-slate-soft focus:outline-none focus:ring-2 focus:ring-fresh focus:border-fresh"
            />
          </div>
        )}

        {view === "refund_run" ? (
          <RefundRun onOpen={handleOpenRow} focusedId={focusedId} />
        ) : list.isLoading ? (
          <div aria-busy="true" className="space-y-4">{Array.from({ length: 3 }, (_, i) => <Skeleton key={i} className="h-28 w-full rounded-2xl" />)}</div>
        ) : list.error && items.length === 0 ? (
          <LoadError what="disputes" onRetry={list.refetch} className="text-sm text-error" />
        ) : items.length === 0 ? (
          <p className="font-sans text-sm text-slate-soft">{emptyText}</p>
        ) : (
          <>
            <ul className="space-y-4">
              {items.map((row) => (
                <li key={row.id}>
                  <AdminDisputeCard
                    dispute={row}
                    focused={focusedId === row.id}
                    onOpen={() => handleOpenRow(row.id)}
                  />
                </li>
              ))}
            </ul>
            {/* Like the old pager: only once the list is longer than one server page. */}
            {list.total > PAGE_INCREMENT && (
              <LoadMoreButton
                shown={items.length}
                total={list.total}
                increment={PAGE_INCREMENT}
                onLoadMore={list.fetchNextPage}
                isLoading={list.isFetchingNextPage}
                label="Load more"
              />
            )}
          </>
        )}
      </div>

      {openRow && (
        <AdminDetailDrawer
          open={true}
          onClose={() => setRowId("")}
          escDisabled={drawerEscDisabled}
          kicker={`Dispute · ${openRow.id}`}
          title={
            <>
              @{openRow.runnerHandle}
              <span className="text-slate-soft"> vs </span>
              @{openRow.photographerHandle}
            </>
          }
          rightHeader={
            <AdminStatusPill
              label={DISPUTE_STATUS_LABEL[openRow.status]}
              tone={disputeStatusTone(openRow.status)}
            />
          }
          subtitle={<DisputeSubtitle dispute={openRow} />}
          actions={
            // Full width (the footer row is right-aligned) and its own scroll, so "Mark refunded" stays reachable on short screens; p-1 keeps focus outlines from being cut off.
            mdUp ? <div className="w-full max-h-[50vh] overflow-y-auto p-1">{actionsAside()}</div> : undefined
          }
        >
          {mdUp ? <DisputeDetailBody dispute={openRow} /> : actionsAside(<DisputeDetailBody dispute={openRow} />)}
        </AdminDetailDrawer>
      )}

    </>
  );
}

function DisputeSubtitle({ dispute }: { dispute: Dispute }) {
  const eventName = dispute.eventName ?? "Event archived";
  return (
    <p className="font-sans text-sm text-slate-soft tnum break-words">
      {eventName}
      <span className="text-slate-soft"> · </span>
      Order total {formatPrice(dispute.orderSnapshot.total)}
      <span className="text-slate-soft"> · </span>
      Reported {sentenceCaseDate(formatLongDate(dispute.reportedAt))}
    </p>
  );
}

function DisputeDetailBody({ dispute }: { dispute: Dispute }) {
  const eventName = dispute.eventName ?? "Event archived";
  const paymentLabel =
    DISPUTE_PAYMENT_LABEL[dispute.orderSnapshot.paymentMethod] ??
    dispute.orderSnapshot.paymentMethod;
  const thumbnailUrl = dispute.photoSnapshot.thumbnailUrl;
  const [previewOpen, setPreviewOpen] = useState(false);

  // Build a PhotoPreviewItem from the dispute's photoSnapshot so the
  // existing lightbox component can render without a separate fetch. We
  // reuse the watermarked thumbnail URL — admin reviews against the same
  // image the runner saw, surfaced full-screen so blur / framing issues are
  // judgable. `time` is borrowed for the kmMark (best contextual signal
  // we have without a separate captured-at field on photoSnapshot).
  const previewPhoto: PhotoPreviewItem = useMemo(() => {
    const kmLabel =
      dispute.photoSnapshot.kmMark === null
        ? "—"
        : `${dispute.photoSnapshot.kmMark} km`;
    return {
      id: dispute.photoId,
      bib: dispute.photoSnapshot.bib,
      time: kmLabel,
      tone: 0,
      price: 0,
      imageUrl: thumbnailUrl ?? null,
      alt: dispute.photoSnapshot.alt || `Disputed photo ${dispute.photoId}`,
    };
  }, [dispute, thumbnailUrl]);

  return (
    <div className="space-y-10">
      <section>
        <Kicker as="p" tone="soft" className="mb-3">
          Claim
        </Kicker>
        <Kicker as="p" tone="soft">
          Reason: {DISPUTE_REASON_LABEL[dispute.reason]}
        </Kicker>
        <p className="font-sans text-sm md:text-base text-ink-soft mt-3 whitespace-pre-line">
          {dispute.note}
        </p>
      </section>

      <section>
        <Kicker as="p" tone="soft" className="mb-3">
          Photo
        </Kicker>
        <div className="grid grid-cols-1 sm:grid-cols-[10rem_1fr] gap-5 items-start">
          <DisputePhotoThumb
            thumbnailUrl={thumbnailUrl}
            alt={dispute.photoSnapshot.alt}
            onClick={() => setPreviewOpen(true)}
          />
          <div className="space-y-3">
            {dispute.photoSnapshot.alt && (
              <p className="font-sans text-sm text-ink-soft">
                {dispute.photoSnapshot.alt}
              </p>
            )}
            <dl className="grid grid-cols-2 gap-y-3 gap-x-6">
              <FieldRow label="Photo id" value={dispute.photoId} mono />
              <FieldRow label="Event" value={eventName} mono={false} />
            </dl>
          </div>
        </div>
      </section>

      <section>
        <Kicker as="p" tone="soft" className="mb-3">
          Order
        </Kicker>
        <dl className="grid grid-cols-1 sm:grid-cols-2 gap-y-4 gap-x-8">
          <FieldRow label="Order id" value={dispute.orderId} mono />
          <FieldRow
            label="Order total"
            value={formatPrice(dispute.orderSnapshot.total)}
            mono
          />
          <FieldRow label="Method" value={paymentLabel} mono={false} />
          <FieldRow
            label="Paid at"
            value={sentenceCaseDate(formatLongDate(dispute.orderSnapshot.paidAt))}
            mono
          />
        </dl>
      </section>

      <DecisionFacts dispute={dispute} />

      <section>
        <Kicker as="p" tone="soft" className="mb-3">
          Parties
        </Kicker>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="rounded-2xl border border-line bg-bone-deep p-4">
            <Kicker as="p" tone="soft">
              Runner
            </Kicker>
            <p className="font-display text-lg text-ink mt-2 break-words">
              @{dispute.runnerHandle}
            </p>
          </div>
          <div className="rounded-2xl border border-line bg-bone-deep p-4">
            <Kicker as="p" tone="soft">
              Photographer
            </Kicker>
            <p className="font-display text-lg text-ink mt-2 break-words">
              @{dispute.photographerHandle}
            </p>
          </div>
        </div>
      </section>

      <section>
        <Kicker as="p" tone="soft" className="mb-3">
          Activity
        </Kicker>
        {(dispute.status === "resolved" || dispute.status === "denied") &&
          (dispute.activity ?? []).length === 0 && (
            <ResolutionNote
              resolution={dispute.resolution}
              refundAmount={dispute.refundAmount}
              resolvedAt={dispute.resolvedAt}
            />
          )}
        <AdminDisputeActivityList
          activity={dispute.activity}
        />
      </section>

      {previewOpen && (
        <PhotoPreviewCard
          mode="review"
          photo={previewPhoto}
          eventName={eventName}
          index={1}
          total={1}
          footerLabel="Admin · Read-only preview"
          onClose={() => setPreviewOpen(false)}
        />
      )}
    </div>
  );
}

// What the admin needs beside the decision. A field the server did not send (an older response) shows no row;
// null means the server looked and there is nothing: no download record, or a deleted runner.
function DecisionFacts({ dispute: d }: { dispute: Dispute }) {
  const h = d.runnerHistory;
  return (
    <section>
      <Kicker as="p" tone="soft" className="mb-3">
        Decision facts
      </Kicker>
      <dl className="grid grid-cols-1 sm:grid-cols-2 gap-y-4 gap-x-8">
        {d.itemChargeAmount != null && (
          <FieldRow label="This photo's charge" value={formatPrice(d.itemChargeAmount)} mono />
        )}
        {d.paymentPhotoCount != null && (
          <FieldRow label="Payment covered" value={`${d.paymentPhotoCount} ${d.paymentPhotoCount === 1 ? "photo" : "photos"}`} mono />
        )}
        {d.refundable != null && (
          <FieldRow
            label="Can it be refunded?"
            value={d.refundable ? "Yes" : `No: ${d.refundBlockedMessage ?? "see the payment"}`}
            mono={false}
          />
        )}
        {d.downloadCount !== undefined && (
          <FieldRow
            label="Downloads started"
            value={
              d.downloadCount === null
                ? "No download record"
                : d.downloadCount === 0
                  ? "No downloads yet"
                  : `${d.downloadCount}${d.lastDownloadedAt ? ` · last ${sentenceCaseDate(formatLongDate(d.lastDownloadedAt))}` : ""}`
            }
            help="Counts download starts; retries and share attempts count again. Gallery and receipt previews are not counted."
            mono={d.downloadCount !== null && d.downloadCount > 0}
          />
        )}
        {h !== undefined && (
          <FieldRow
            label="This runner's other requests"
            value={
              h === null
                ? "Runner deleted"
                : `${h.total} total · ${h.refunded} refunded · ${h.denied} declined · ${h.withdrawn} withdrawn · ${h.open} open`
            }
            mono={false}
          />
        )}
      </dl>
    </section>
  );
}

function DisputePhotoThumb({
  thumbnailUrl,
  alt,
  onClick,
}: {
  thumbnailUrl: string | undefined;
  alt: string;
  onClick?: () => void;
}) {
  // Clickable thumbnail opens the shared <PhotoPreviewCard> in review mode
  // so admins can judge framing / blur / wrong-runner claims at full size.
  if (thumbnailUrl) {
    return (
      <button
        type="button"
        onClick={onClick}
        disabled={!onClick}
        aria-label={`Open ${alt || "disputed photo"} at full size`}
        className="group aspect-[4/3] w-full rounded-2xl border border-line bg-bone-deep overflow-hidden focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone enabled:cursor-zoom-in"
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={thumbnailUrl}
          alt={alt}
          className="size-full object-cover transition-transform duration-300 group-hover:scale-[1.02]"
          loading="lazy"
        />
      </button>
    );
  }
  // Fallback for ghost rows (photo deleted, presigned URL not minted).
  return <LogoPlaceholder size="md" label={alt} className="aspect-[4/3] rounded-2xl border border-line" />;
}

function FieldRow({
  label,
  value,
  mono,
  help,
}: {
  label: string;
  value: string;
  mono: boolean;
  help?: string;
}) {
  return (
    <div>
      <Kicker as="dt" tone="soft">
        {label}
      </Kicker>
      <dd
        className={`mt-1 break-words ${
          mono
            ? "font-mono text-[13px] text-ink tnum"
            : "font-sans text-sm text-ink tnum"
        }`}
      >
        {value}
      </dd>
      {help && <p className="mt-1 font-sans text-xs text-slate-soft">{help}</p>}
    </div>
  );
}

function ResolutionNote({
  resolution,
  refundAmount,
  resolvedAt,
}: {
  resolution: DisputeResolution | null;
  refundAmount: number | null;
  resolvedAt: string | null;
}) {
  if (!resolution) return null;
  return (
    <div className="rounded-xl border border-line bg-bone-deep p-4 mb-4">
      <Kicker as="p" tone="soft">
        Closed before live decision logging started
      </Kicker>
      <p className="font-sans text-sm text-ink-soft mt-2">
        {DISPUTE_RESOLUTION_LABEL[resolution]}
        {refundAmount !== null && ` · ${formatPrice(refundAmount)} refunded`}
        {resolvedAt && ` · ${sentenceCaseDate(formatLongDate(resolvedAt))}`}
      </p>
    </div>
  );
}
