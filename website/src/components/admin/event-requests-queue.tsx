"use client";

import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Slab } from "@/components/profile-shell";
import { AdminFormModal } from "@/components/admin/admin-form-modal";
import { AdminTextarea } from "@/components/admin/admin-form-fields";
import { Kicker } from "@/components/ui/kicker";
import { Skeleton } from "@/components/ui/skeleton";
import {
  BTN_PRIMARY,
  BTN_SECONDARY,
  BTN_SIZE,
} from "@/components/ui/button-styles";
import { useAdminEvents } from "@/hooks/use-admin-data";
import { useToast } from "@/hooks/use-toast";
import { ApiError } from "@/lib/api";
import {
  approveAdminEvent,
  rejectAdminEvent,
  type AdminEventRow,
} from "@/lib/api-admin";
import { formatLongDate } from "@/lib/format";
import { describePricing } from "@/lib/photographer-events";
import { cn } from "@/lib/utils";

// Event requests queue (V46) — photographer-owned events waiting on an
// admin: a new submission (the event is a draft nobody can see) or a pricing
// change parked on a live event (the gallery keeps its current settings until
// approved). Approve / Send back land on the same two BE endpoints; the
// service decides which state it is in. Price edits before approval go
// through the existing /admin/events board.

export function EventRequestsQueue() {
  const rows = useAdminEvents({ review: "queue" });
  const queryClient = useQueryClient();
  const { showToast } = useToast();
  const [rejecting, setRejecting] = useState<AdminEventRow | null>(null);
  const [reason, setReason] = useState("");
  const [busyId, setBusyId] = useState<string | null>(null);

  async function run(row: AdminEventRow, action: () => Promise<unknown>, ok: string) {
    setBusyId(row.id);
    try {
      await action();
      await queryClient.invalidateQueries({ queryKey: ["admin", "events"] });
      showToast({ kind: "success", message: ok });
    } catch (err) {
      showToast({
        kind: "error",
        message:
          err instanceof ApiError
            ? (err.errors[0]?.message ?? err.message)
            : "Couldn't update the event. Try again.",
      });
    } finally {
      setBusyId(null);
    }
  }

  function submitReject() {
    const row = rejecting;
    const text = reason.trim();
    if (!row || !text) return;
    setRejecting(null);
    setReason("");
    void run(
      row,
      () => rejectAdminEvent(row.id, text),
      row.reviewStatus === "change_pending"
        ? `Change declined — ${row.name} keeps its current pricing.`
        : `${row.name} sent back to the photographer.`,
    );
  }

  const waiting = rows?.length ?? 0;

  return (
    <Slab
      id="event-requests"
      number="01"
      title="Event requests"
      caption="Photographer-owned events · new submissions and pricing changes"
      trailing={rows ? `${waiting} waiting` : undefined}
    >
      {rows === null ? (
        <div className="space-y-4">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-28 w-full rounded-2xl" />
          ))}
        </div>
      ) : rows.length === 0 ? (
        <div className="border border-dashed border-line rounded-2xl p-8 md:p-12 text-center">
          <p className="font-display text-2xl md:text-3xl font-medium tracking-tight text-ink">
            Nothing waiting.
          </p>
          <p className="font-sans text-base text-ink-soft mt-3 max-w-sm mx-auto">
            New photographer events and pricing change requests land here.
          </p>
        </div>
      ) : (
        <ul className="divide-y divide-line border-y border-line">
          {rows.map((row) => (
            <RequestRow
              key={row.id}
              row={row}
              busy={busyId === row.id}
              onApprove={() =>
                run(
                  row,
                  () => approveAdminEvent(row.id),
                  row.reviewStatus === "change_pending"
                    ? `Change applied to ${row.name}.`
                    : `${row.name} is live. Uploads are open.`,
                )
              }
              onReject={() => setRejecting(row)}
            />
          ))}
        </ul>
      )}

      <AdminFormModal
        open={rejecting !== null}
        onClose={() => {
          setRejecting(null);
          setReason("");
        }}
        onSubmit={submitReject}
        title={
          rejecting?.reviewStatus === "change_pending"
            ? "Decline the change"
            : "Send back for fixes"
        }
        intro={
          rejecting
            ? `${rejecting.createdByName ?? "The photographer"} sees your reason in their inbox.`
            : undefined
        }
        submitLabel={
          rejecting?.reviewStatus === "change_pending" ? "Decline" : "Send back"
        }
        submitDisabled={reason.trim().length === 0}
      >
        <AdminTextarea
          id="event-reject-reason"
          label="Reason (max 500)"
          value={reason}
          onChange={setReason}
          maxLength={500}
          placeholder="What should they change before resubmitting?"
        />
      </AdminFormModal>
    </Slab>
  );
}

function RequestRow({
  row,
  busy,
  onApprove,
  onReject,
}: {
  row: AdminEventRow;
  busy: boolean;
  onApprove: () => void;
  onReject: () => void;
}) {
  const isChange = row.reviewStatus === "change_pending";
  const current = describePricing({
    pricingMode: row.pricingMode,
    pricePerPhoto: row.pricePerPhoto ?? 0,
    watermarkPolicy: row.watermarkPolicy,
  });
  const owner = row.createdByHandle
    ? `@${row.createdByHandle}`
    : (row.createdByName ?? "Photographer");

  return (
    <li className="py-5 md:py-6 flex flex-col md:flex-row md:items-start gap-4 md:gap-8">
      <div className="flex-1 min-w-0">
        <Kicker as="p" tone="soft" tnum>
          {isChange ? "Pricing change · live event" : "New event"} · {owner}
        </Kicker>
        <p className="font-display text-xl md:text-2xl font-medium tracking-tight text-ink mt-2 truncate">
          {row.name}
        </p>
        <p className="font-sans text-sm text-ink-soft mt-1 tnum">
          {formatLongDate(row.date, true)} · {row.location} ·{" "}
          {row.visibility === "unlisted" ? "Unlisted" : "Public"}
        </p>
        {isChange && row.pendingChange ? (
          <p className="font-sans text-sm text-ink mt-3 tnum">
            <span className="text-slate">Now</span> {current}
            <span className="text-slate-soft mx-2">→</span>
            <span className="text-slate">Requested</span>{" "}
            {describePricing(row.pendingChange)}
          </p>
        ) : (
          <p className="font-sans text-sm text-ink mt-3 tnum">{current}</p>
        )}
        {row.description && !isChange && (
          <p className="font-sans text-sm text-slate mt-2 line-clamp-2 max-w-prose">
            {row.description}
          </p>
        )}
      </div>
      <div className="flex gap-2 shrink-0">
        <button
          type="button"
          onClick={onApprove}
          disabled={busy}
          className={cn(BTN_PRIMARY, BTN_SIZE.sm)}
        >
          {busy ? "Working…" : "Approve"}
        </button>
        <button
          type="button"
          onClick={onReject}
          disabled={busy}
          className={cn(BTN_SECONDARY, BTN_SIZE.sm)}
        >
          {isChange ? "Decline" : "Send back"}
        </button>
      </div>
    </li>
  );
}
