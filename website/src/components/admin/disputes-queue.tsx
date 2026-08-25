"use client";

import { useEffect, useMemo, useState } from "react";
import { Slab } from "@/components/profile-shell";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { Kicker } from "@/components/ui/kicker";
import { PAGE_SIZE } from "@/lib/pagination-config";
import { AdminDisputeCard } from "@/components/admin/admin-dispute-card";
import { AdminDetailDrawer } from "@/components/admin/admin-detail-drawer";
import {
  PhotoPreviewCard,
  type PhotoPreviewItem,
} from "@/components/photos/photo-preview-card";
import { AdminStatusPill, type AdminStatusPillTone } from "@/components/admin/admin-status-pill";
import { AdminDisputeActivityList } from "@/components/admin/admin-dispute-activity-list";
import { AdminResolveDisputeModal } from "@/components/admin/admin-resolve-dispute-modal";
import { AdminRefundConfirmModal } from "@/components/admin/admin-refund-confirm-modal";
import { AdminDenyDisputeModal } from "@/components/admin/admin-deny-dispute-modal";
import { AdminEscalateModal } from "@/components/admin/admin-escalate-modal";
import {
  useAdminDisputeStore,
  mergeDisputesWithOverrides,
} from "@/store/admin-dispute-store";
import { useAdminDisputes } from "@/hooks/use-admin-data";
import { useUrlState } from "@/hooks/use-url-state";
import { useToast } from "@/hooks/use-toast";
import { useQueueKeyboardNav } from "@/hooks/use-admin-keyboard";
import {
  type Dispute,
  type DisputeResolution,
  type DisputeStatus,
  DISPUTE_REASON_LABEL,
  DISPUTE_RESOLUTION_LABEL,
} from "@/lib/admin-disputes";
import { syntheticCoverGradient } from "@/lib/admin-photographer-view";
import { formatLongDate } from "@/lib/format";
import { formatPrice } from "@/lib/utils";

const SEVEN_DAYS_MS = 7 * 24 * 60 * 60 * 1000;

const STATUS_LABEL: Record<DisputeStatus, string> = {
  open: "Open",
  resolved: "Resolved",
  denied: "Denied",
  escalated: "Escalated",
};

function statusTone(status: DisputeStatus): AdminStatusPillTone {
  switch (status) {
    case "open":
      return "amber";
    case "resolved":
      return "fresh";
    case "denied":
      return "ink";
    case "escalated":
      return "slate";
  }
}

const PAYMENT_LABEL: Record<string, string> = {
  gcash: "GCash",
  maya: "Maya",
  paymaya: "PayMaya",
  card: "Card",
  grabpay: "GrabPay",
};

interface PendingResolve {
  resolution: DisputeResolution;
  refundAmount: number | null;
  note: string;
}

// Disputes queue body — extracted from /admin/disputes/page.tsx so the
// universal /admin/inbox can render the same four slabs (Open / Resolved
// / Denied / Escalated) under the chip filter. Header stays on the
// dedicated page; this component renders only the slab stack.
//
// Phase 3 adds a focus-mode drawer. Card click → drawer with full
// detail (claim, photo, parties, activity log) and Resolve/Deny/Escalate
// in the footer. URL state via ?row=<disputeId>. The dedicated route
// /admin/disputes/[id] still works as a deep-link backup. Modal stack
// (resolve → refund-confirm) is owned at queue level so the drawer's ESC
// can stand down via escDisabled while any modal is on top.

export function DisputesQueue() {
  const overrides = useAdminDisputeStore((s) => s.overrides);
  const submissions = useAdminDisputeStore((s) => s.submissions);
  const resolve = useAdminDisputeStore((s) => s.resolve);
  const deny = useAdminDisputeStore((s) => s.deny);
  const escalate = useAdminDisputeStore((s) => s.escalate);
  const serverDisputes = useAdminDisputes() ?? [];
  const { showToast } = useToast();

  const [rowId, setRowId] = useUrlState<string>("row", "");
  const [resolveOpen, setResolveOpen] = useState(false);
  const [denyOpen, setDenyOpen] = useState(false);
  const [escalateOpen, setEscalateOpen] = useState(false);
  const [pendingResolve, setPendingResolve] = useState<PendingResolve | null>(
    null,
  );

  const effective = useMemo(
    () => mergeDisputesWithOverrides(serverDisputes, overrides, submissions),
    [serverDisputes, overrides, submissions],
  );

  const byId = useMemo(() => {
    const map = new Map<string, Dispute>();
    for (const d of effective) map.set(d.id, d);
    return map;
  }, [effective]);

  const openRow = useMemo<Dispute | null>(() => {
    if (!rowId) return null;
    return byId.get(rowId) ?? null;
  }, [rowId, byId]);

  useEffect(() => {
    if (rowId && !openRow) setRowId("");
  }, [rowId, openRow, setRowId]);

  const open = useMemo(
    () =>
      [...effective.filter((d) => d.status === "open")].sort((a, b) =>
        a.reportedAt.localeCompare(b.reportedAt),
      ),
    [effective],
  );
  const resolved = useMemo(
    () =>
      [...effective.filter((d) => d.status === "resolved")].sort((a, b) =>
        (b.resolvedAt ?? "").localeCompare(a.resolvedAt ?? ""),
      ),
    [effective],
  );
  const denied = useMemo(
    () =>
      [...effective.filter((d) => d.status === "denied")].sort((a, b) =>
        (b.resolvedAt ?? "").localeCompare(a.resolvedAt ?? ""),
      ),
    [effective],
  );
  const escalated = useMemo(
    () =>
      [...effective.filter((d) => d.status === "escalated")].sort((a, b) =>
        b.reportedAt.localeCompare(a.reportedAt),
      ),
    [effective],
  );

  const drawerEscDisabled =
    resolveOpen || denyOpen || escalateOpen || pendingResolve !== null;

  // Pagination — load-more model. Open keeps full visibility because
  // that's where the triage flow + J/K nav land. Read-only history slabs
  // start at 10 and grow by 10 per click.
  const [resolvedLoaded, setResolvedLoaded] = useState(PAGE_SIZE.ADMIN_INITIAL);
  const [deniedLoaded, setDeniedLoaded] = useState(PAGE_SIZE.ADMIN_INITIAL);
  const [escalatedLoaded, setEscalatedLoaded] = useState(
    PAGE_SIZE.ADMIN_INITIAL,
  );
  const resolvedVisible = useMemo(
    () => resolved.slice(0, resolvedLoaded),
    [resolved, resolvedLoaded],
  );
  const deniedVisible = useMemo(
    () => denied.slice(0, deniedLoaded),
    [denied, deniedLoaded],
  );
  const escalatedVisible = useMemo(
    () => escalated.slice(0, escalatedLoaded),
    [escalated, escalatedLoaded],
  );

  // Keyboard nav iterates through every dispute slab in render order.
  // Resolve/Deny/Escalate aren't part of the E/R/H/S verb whitelist, so
  // the drawer doesn't register verbs — keystrokes inside the drawer fall
  // through to ESC + the click-only footer buttons. Use the visible
  // slices for paginated slabs so J/K only visits rows that are currently
  // rendered.
  const rowIds = useMemo(
    () =>
      [
        ...open,
        ...resolvedVisible,
        ...deniedVisible,
        ...escalatedVisible,
      ].map((d) => d.id),
    [open, resolvedVisible, deniedVisible, escalatedVisible],
  );
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

  function handleContinueResolve(args: PendingResolve) {
    if (!openRow) return;
    if (args.resolution === "deny") {
      resolve(openRow.id, {
        resolution: "deny",
        refundAmount: null,
        reason: args.note,
      });
      setResolveOpen(false);
      showToast({
        kind: "info",
        message: `Closed without refund · ${openRow.id}`,
      });
      return;
    }
    setPendingResolve(args);
  }

  function handleConfirmRefund() {
    if (!openRow || !pendingResolve) return;
    resolve(openRow.id, {
      resolution: pendingResolve.resolution,
      refundAmount: pendingResolve.refundAmount,
      reason: pendingResolve.note,
    });
    setPendingResolve(null);
    setResolveOpen(false);
    showToast({ kind: "success", message: `Refunded · ${openRow.id}` });
  }

  function handleCancelRefund() {
    setPendingResolve(null);
  }

  function handleDeny(reason: string) {
    if (!openRow) return;
    deny(openRow.id, reason);
    setDenyOpen(false);
    showToast({ kind: "info", message: `Denied · ${openRow.id}` });
  }

  function handleEscalate(note: string | null) {
    if (!openRow) return;
    escalate(openRow.id, note);
    setEscalateOpen(false);
    showToast({ kind: "info", message: `Escalated · ${openRow.id}` });
  }

  return (
    <>
      <DisputeSlab
        id="open"
        number="01"
        title="Open"
        caption="Awaiting decision"
        totalCount={open.length}
        rows={open}
        empty="No open disputes — queue is clear."
        onOpenRow={handleOpenRow}
        focusedId={focusedId}
      />
      <DisputeSlab
        id="resolved"
        number="02"
        title="Resolved"
        caption="Closed with refund"
        totalCount={resolved.length}
        rows={resolvedVisible}
        empty="No resolved disputes yet."
        onOpenRow={handleOpenRow}
        focusedId={focusedId}
        loadMore={{
          shown: resolvedVisible.length,
          total: resolved.length,
          increment: PAGE_SIZE.ADMIN_INCREMENT,
          onLoadMore: () =>
            setResolvedLoaded((c) => c + PAGE_SIZE.ADMIN_INCREMENT),
        }}
      />
      <DisputeSlab
        id="denied"
        number="03"
        title="Denied"
        caption="Closed without refund"
        totalCount={denied.length}
        rows={deniedVisible}
        empty="No denied disputes."
        onOpenRow={handleOpenRow}
        focusedId={focusedId}
        loadMore={{
          shown: deniedVisible.length,
          total: denied.length,
          increment: PAGE_SIZE.ADMIN_INCREMENT,
          onLoadMore: () =>
            setDeniedLoaded((c) => c + PAGE_SIZE.ADMIN_INCREMENT),
        }}
      />
      <DisputeSlab
        id="escalated"
        number="04"
        title="Escalated"
        caption="Pushed to higher review"
        totalCount={escalated.length}
        rows={escalatedVisible}
        empty="No escalations."
        onOpenRow={handleOpenRow}
        focusedId={focusedId}
        loadMore={{
          shown: escalatedVisible.length,
          total: escalated.length,
          increment: PAGE_SIZE.ADMIN_INCREMENT,
          onLoadMore: () =>
            setEscalatedLoaded((c) => c + PAGE_SIZE.ADMIN_INCREMENT),
        }}
      />

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
              label={STATUS_LABEL[openRow.status]}
              tone={statusTone(openRow.status)}
            />
          }
          subtitle={<DisputeSubtitle dispute={openRow} />}
          actions={
            <DisputeDrawerActions
              dispute={openRow}
              onResolve={() => setResolveOpen(true)}
              onDeny={() => setDenyOpen(true)}
              onEscalate={() => setEscalateOpen(true)}
            />
          }
        >
          <DisputeDetailBody dispute={openRow} />
        </AdminDetailDrawer>
      )}

      {openRow && (
        <>
          <AdminResolveDisputeModal
            open={resolveOpen}
            escDisabled={pendingResolve !== null}
            onClose={() => {
              setResolveOpen(false);
              setPendingResolve(null);
            }}
            onContinue={handleContinueResolve}
            disputeId={openRow.id}
            orderTotal={openRow.orderSnapshot.total}
          />
          {pendingResolve &&
            pendingResolve.resolution !== "deny" &&
            pendingResolve.refundAmount !== null && (
              <AdminRefundConfirmModal
                open={true}
                onCancel={handleCancelRefund}
                onConfirm={handleConfirmRefund}
                amount={pendingResolve.refundAmount}
                runnerHandle={openRow.runnerHandle}
                isFull={pendingResolve.resolution === "refund_full"}
              />
            )}
          <AdminDenyDisputeModal
            open={denyOpen}
            onClose={() => setDenyOpen(false)}
            onSubmit={handleDeny}
            disputeId={openRow.id}
          />
          <AdminEscalateModal
            open={escalateOpen}
            onClose={() => setEscalateOpen(false)}
            onSubmit={handleEscalate}
            targetLabel={`Escalate ${openRow.id}`}
            body="Push this dispute to the next review tier. The runner sees the dispute remains open while higher-level review takes place."
          />
        </>
      )}
    </>
  );
}

export function useOpenDisputesCount(): number {
  const overrides = useAdminDisputeStore((s) => s.overrides);
  const submissions = useAdminDisputeStore((s) => s.submissions);
  const serverDisputes = useAdminDisputes() ?? [];
  return useMemo(
    () =>
      mergeDisputesWithOverrides(serverDisputes, overrides, submissions).filter(
        (d) => d.status === "open",
      ).length,
    [serverDisputes, overrides, submissions],
  );
}

export function useRefundedThisWeekTotal(): number {
  const overrides = useAdminDisputeStore((s) => s.overrides);
  const submissions = useAdminDisputeStore((s) => s.submissions);
  const serverDisputes = useAdminDisputes() ?? [];
  return useMemo(() => {
    const cutoff = Date.now() - SEVEN_DAYS_MS;
    return mergeDisputesWithOverrides(serverDisputes, overrides, submissions)
      .filter((d) => {
        if (d.status !== "resolved" || d.refundAmount === null) return false;
        const t = new Date(d.resolvedAt ?? "").getTime();
        return Number.isFinite(t) && t >= cutoff;
      })
      .reduce((acc, d) => acc + (d.refundAmount ?? 0), 0);
  }, [serverDisputes, overrides, submissions]);
}

function DisputeSlab({
  id,
  number,
  title,
  caption,
  rows,
  totalCount,
  empty,
  onOpenRow,
  focusedId,
  loadMore,
}: {
  id: string;
  number: string;
  title: string;
  caption: string;
  rows: Dispute[];
  totalCount: number;
  empty: string;
  onOpenRow: (id: string) => void;
  focusedId: string | null;
  loadMore?: {
    shown: number;
    total: number;
    increment: number;
    onLoadMore: () => void;
  };
}) {
  const noun = totalCount === 1 ? "dispute" : "disputes";
  return (
    <Slab
      id={id}
      number={number}
      title={title}
      caption={caption}
      trailing={`${totalCount} ${noun}`}
    >
      {totalCount === 0 ? (
        <p className="font-sans text-sm text-slate-soft">{empty}</p>
      ) : (
        <>
          <ul className="space-y-4">
            {rows.map((row) => (
              <li key={row.id}>
                <AdminDisputeCard
                  dispute={row}
                  focused={focusedId === row.id}
                  onOpen={() => onOpenRow(row.id)}
                />
              </li>
            ))}
          </ul>
          {loadMore && (
            <LoadMoreButton
              shown={loadMore.shown}
              total={loadMore.total}
              increment={loadMore.increment}
              onLoadMore={loadMore.onLoadMore}
            />
          )}
        </>
      )}
    </Slab>
  );
}

function DisputeSubtitle({ dispute }: { dispute: Dispute }) {
  const eventName = dispute.eventName ?? "Event archived";
  return (
    <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft tnum">
      {eventName}
      <span className="text-slate-soft"> · </span>
      {formatPrice(dispute.orderSnapshot.total)}
      <span className="text-slate-soft"> · </span>
      Reported {formatLongDate(dispute.reportedAt)}
    </p>
  );
}

function DisputeDrawerActions({
  dispute,
  onResolve,
  onDeny,
  onEscalate,
}: {
  dispute: Dispute;
  onResolve: () => void;
  onDeny: () => void;
  onEscalate: () => void;
}) {
  const isOpen = dispute.status === "open";
  const isDenied = dispute.status === "denied";
  const escalateAllowed = isOpen || isDenied;

  if (!isOpen && !escalateAllowed) {
    return (
      <p className="font-sans text-sm text-slate-soft">
        This dispute is closed. No further actions are available.
      </p>
    );
  }

  return (
    <>
      {escalateAllowed && (
        <button
          type="button"
          onClick={onEscalate}
          className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate hover:text-ink transition-colors px-4 py-2"
        >
          Escalate…
        </button>
      )}
      {isOpen && (
        <button
          type="button"
          onClick={onDeny}
          className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-ink border border-line hover:bg-ink hover:text-surface hover:border-ink transition-colors rounded-full px-5 py-2"
        >
          Deny…
        </button>
      )}
      {isOpen && (
        <button
          type="button"
          onClick={onResolve}
          className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-surface bg-fresh hover:bg-fresh-deep transition-colors rounded-full px-5 py-2"
        >
          Resolve…
        </button>
      )}
    </>
  );
}

function DisputeDetailBody({ dispute }: { dispute: Dispute }) {
  const eventName = dispute.eventName ?? "Event archived";
  const paymentLabel =
    PAYMENT_LABEL[dispute.orderSnapshot.paymentMethod] ??
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
        <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft">
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
            label="Total"
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

      <section>
        <Kicker as="p" tone="soft" className="mb-3">
          Parties
        </Kicker>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="rounded-2xl border border-line bg-bone-deep p-4">
            <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft">
              Runner
            </p>
            <p className="font-display text-lg text-ink mt-2">
              @{dispute.runnerHandle}
            </p>
          </div>
          <div className="rounded-2xl border border-line bg-bone-deep p-4">
            <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft">
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
        {dispute.status !== "open" &&
          (dispute.activity ?? []).length === 0 && (
            <ResolutionNote
              resolution={dispute.resolution}
              refundAmount={dispute.refundAmount}
              resolvedAt={dispute.resolvedAt}
            />
          )}
        <AdminDisputeActivityList
          disputeId={dispute.id}
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
}: {
  label: string;
  value: string;
  mono: boolean;
}) {
  return (
    <div>
      <dt className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft">
        {label}
      </dt>
      <dd
        className={`mt-1 ${
          mono
            ? "font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-ink tnum"
            : "font-sans text-sm text-ink"
        }`}
      >
        {value}
      </dd>
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
      <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft">
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
