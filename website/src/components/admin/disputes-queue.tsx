"use client";

import { useEffect, useMemo, useState } from "react";
import { Kicker } from "@/components/ui/kicker";
import { BTN_GHOST, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { AdminDisputeCard } from "@/components/admin/admin-dispute-card";
import { RefundRun } from "@/components/admin/refund-run";
import { AdminDetailDrawer } from "@/components/admin/admin-detail-drawer";
import {
  PhotoPreviewCard,
  type PhotoPreviewItem,
} from "@/components/photos/photo-preview-card";
import { AdminStatusPill } from "@/components/admin/admin-status-pill";
import { AdminDisputeActivityList } from "@/components/admin/admin-dispute-activity-list";
import { AdminDisputeActionAside } from "@/components/admin/admin-dispute-action-aside";
import { useAdminDispute, useAdminDisputePage, EMPTY_DISPUTES } from "@/hooks/use-admin-disputes";
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
import { syntheticCoverGradient } from "@/lib/admin-photographer-view";
import { formatLongDate } from "@/lib/format";
import { cn, formatPrice } from "@/lib/utils";

// The server's views (no status filter in the UI on purpose; the server keeps `status=`).
type QueueView = Extract<AdminDisputeView, "needs_me" | "overdue" | "refund_run" | "closed" | "all">;
const VIEWS: readonly { value: QueueView; label: string; empty: string }[] = [
  { value: "needs_me", label: "Needs you", empty: "Nothing needs you right now." },
  { value: "overdue", label: "Overdue", empty: "Nothing is overdue." },
  { value: "refund_run", label: "Refund run", empty: "No approved refunds are waiting to be paid." },
  { value: "closed", label: "Closed", empty: "No closed disputes yet." },
  { value: "all", label: "All", empty: "No disputes yet." },
];
const parseView = (raw: string): QueueView =>
  VIEWS.some((v) => v.value === raw) ? (raw as QueueView) : "needs_me";
const SEARCH_DEBOUNCE_MS = 300;
// The server's page size when the request names none (PaginationParams.DEFAULT_LIMIT).
const SERVER_PAGE_SIZE = 50;

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
    const timer = setTimeout(() => setQ(next), SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [qDraft, q, setQ]);

  // A new view or search starts at the first page.
  const pageScope = `${view}|${q}`;
  const [paging, setPaging] = useState({ scope: pageScope, offset: 0 });
  const offset = paging.scope === pageScope ? paging.offset : 0;
  const goTo = (next: number) => setPaging({ scope: pageScope, offset: next });

  const query = useAdminDisputePage({ view, q: view === "refund_run" ? undefined : q || undefined, offset });
  const items = query.data?.items ?? EMPTY_DISPUTES;
  const total = query.data?.total ?? 0;
  const limit = query.data?.limit || SERVER_PAGE_SIZE;
  // Rows left this page (decided, or the view shrank): step back to the last page that has rows.
  if (query.isSuccess && !query.isPlaceholderData && items.length === 0 && offset > 0) {
    goTo(Math.floor(Math.max(total - 1, 0) / limit) * limit);
  }

  const [rowId, setRowId] = useUrlState<string>("row", "");
  // Any aside dialog (resolve, refund confirm, deny, escalate, refund panel) is open; the drawer must ignore Esc.
  const [asideDialogOpen, setAsideDialogOpen] = useState(false);

  const byId = useMemo(() => {
    const map = new Map<string, Dispute>();
    for (const d of items) map.set(d.id, d);
    return map;
  }, [items]);

  // A ?row= link to a dispute outside the loaded list is fetched by id.
  const fallbackId = query.isSuccess && rowId && !byId.has(rowId) ? rowId : "";
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

  if (query.isPending) return <p role="status" className="py-6">Loading disputes…</p>;
  if (query.isError && !query.data) return (
    <div role="alert" className="space-y-4 py-6">
      <p>Couldn&apos;t load disputes.</p>
      <button type="button" className={`${BTN_SECONDARY} ${BTN_SIZE.sm}`} onClick={() => void query.refetch()}>
        Try again
      </button>
    </div>
  );

  const emptyText = q
    ? `No disputes match “${q}”.`
    : VIEWS.find((v) => v.value === view)?.empty;

  return (
    <>
      <div className="pt-6 md:pt-8 space-y-6">
        {/* Side by side from lg only: at md the admin rail leaves too narrow a column. */}
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div role="tablist" aria-label="Dispute views" className="flex flex-wrap gap-2">
            {VIEWS.map((v) => {
              const active = view === v.value;
              return (
                <button
                  key={v.value}
                  type="button"
                  role="tab"
                  aria-selected={active}
                  onClick={() => setView(v.value)}
                  className={cn(
                    "font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] rounded-full px-4 py-2 transition-colors",
                    active
                      ? "bg-ink text-bone border border-ink"
                      : "bg-bone text-slate border border-line hover:text-ink hover:border-ink",
                  )}
                >
                  {v.label}
                </button>
              );
            })}
          </div>
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
        </div>

        {view === "refund_run" ? (
          <RefundRun onOpen={handleOpenRow} />
        ) : items.length === 0 ? (
          <p className="font-sans text-sm text-slate-soft">{emptyText}</p>
        ) : (
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
        )}

        {view !== "refund_run" && total > limit && (
          <div className="flex flex-wrap items-center gap-2">
            <button type="button" className={cn(BTN_GHOST, BTN_SIZE.sm)} disabled={offset === 0} onClick={() => goTo(Math.max(0, offset - limit))}>← Previous</button>
            <span className="font-mono tnum text-sm text-slate">{offset + 1}–{offset + items.length} of {total}</span>
            <button type="button" className={cn(BTN_GHOST, BTN_SIZE.sm)} disabled={offset + items.length >= total} onClick={() => goTo(offset + items.length)}>Next →</button>
          </div>
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
            <div className="w-full max-h-[50vh] overflow-y-auto p-1">
              <AdminDisputeActionAside
                key={openRow.id}
                dispute={openRow}
                runnerHandle={openRow.runnerHandle}
                onDialogOpenChange={setAsideDialogOpen}
                className=""
              />
            </div>
          }
        >
          <DisputeDetailBody dispute={openRow} />
        </AdminDetailDrawer>
      )}

    </>
  );
}

function DisputeSubtitle({ dispute }: { dispute: Dispute }) {
  const eventName = dispute.eventName ?? "Event archived";
  return (
    <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft tnum">
      {eventName}
      <span className="text-slate-soft"> · </span>
      Order total {formatPrice(dispute.orderSnapshot.total)}
      <span className="text-slate-soft"> · </span>
      Reported {formatLongDate(dispute.reportedAt)}
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
        <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft">
          Reason: {DISPUTE_REASON_LABEL[dispute.reason]}
        </p>
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
            disputeId={dispute.id}
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
            value={formatLongDate(dispute.orderSnapshot.paidAt)}
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
            <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft">
              Runner
            </p>
            <p className="font-display text-lg text-ink mt-2">
              @{dispute.runnerHandle}
            </p>
          </div>
          <div className="rounded-2xl border border-line bg-bone-deep p-4">
            <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft">
              Photographer
            </p>
            <p className="font-display text-lg text-ink mt-2">
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
                  : `${d.downloadCount}${d.lastDownloadedAt ? ` · last ${formatLongDate(d.lastDownloadedAt)}` : ""}`
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
  disputeId,
  alt,
  onClick,
}: {
  thumbnailUrl: string | undefined;
  disputeId: string;
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
  const cover = syntheticCoverGradient(disputeId);
  return (
    <div
      className="aspect-[4/3] rounded-2xl border border-line"
      style={{
        background: `linear-gradient(135deg, ${cover.from}, ${cover.to})`,
      }}
      aria-label={alt}
    />
  );
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
      <dt className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft">
        {label}
      </dt>
      <dd
        className={`mt-1 ${
          mono
            ? "font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-ink tnum"
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
      <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft">
        Closed before live decision logging started
      </p>
      <p className="font-sans text-sm text-ink-soft mt-2">
        {DISPUTE_RESOLUTION_LABEL[resolution]}
        {refundAmount !== null && ` · ${formatPrice(refundAmount)} refunded`}
        {resolvedAt && ` · ${formatLongDate(resolvedAt)}`}
      </p>
    </div>
  );
}
