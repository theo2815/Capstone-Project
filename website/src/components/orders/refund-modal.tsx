"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Modal } from "@/components/ui/modal";
import { Kicker } from "@/components/ui/kicker";
import { FieldError } from "@/components/ui/field-error";
import { CharCounter } from "@/components/ui/char-counter";
import {
  Dropdown,
  DropdownItem,
} from "@/components/ui/dropdown";
import { RefundPolicyContent } from "@/components/orders/refund-policy-content";
import { getDisputableePhotoIds } from "@/lib/refund-helpers";
import {
  DISPUTE_REASON_LABEL,
  type DisputeReason,
} from "@/lib/admin-disputes";
import { REFUND_PROCESSING_DAYS } from "@/lib/refund-policy";
import { useToast } from "@/hooks/use-toast";
import type { MockOrder } from "@/store/orders-store";
import type { RunnerDispute, RunnerDisputeStatus } from "@/lib/api-orders";
import { formatLongDate } from "@/lib/format";
import { formatPrice, cn } from "@/lib/utils";
import { submitOrderRefund } from "@/lib/api-orders";
import { ApiError } from "@/lib/api";

const NOTE_MAX = 500;
const REASON_ORDER: ReadonlyArray<DisputeReason> = [
  "wrong_runner",
  "low_quality",
  "not_received",
  "duplicate_charge",
  "other",
];

type RefundModalProps =
  | {
      mode: "policy";
      isOpen: boolean;
      onClose: () => void;
    }
  | {
      mode: "request";
      isOpen: boolean;
      onClose: () => void;
      order: MockOrder;
      eventName: string;
    };

export function RefundModal(props: RefundModalProps) {
  if (props.mode === "policy") {
    return (
      <Modal
        isOpen={props.isOpen}
        onClose={props.onClose}
        title="Refund Policy"
      >
        <p className="font-sans text-sm md:text-base text-ink-soft mb-6 max-w-md">
          A few rules so you know what to expect. Same policy whether you
          submit from a receipt or ask in advance.
        </p>
        <RefundPolicyContent />
        <div className="mt-8 flex justify-end">
          <button
            type="button"
            onClick={props.onClose}
            className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] border border-line rounded-full px-6 py-3 text-ink hover:bg-ink hover:text-bone transition-colors"
          >
            Got it
          </button>
        </div>
      </Modal>
    );
  }

  return <RefundRequestModal {...props} />;
}

interface RequestProps {
  isOpen: boolean;
  onClose: () => void;
  order: MockOrder;
  eventName: string;
}

function RefundRequestModal({
  isOpen,
  onClose,
  order,
  eventName,
}: RequestProps) {
  const queryClient = useQueryClient();
  const { showToast } = useToast();

  const orderDisputes = useMemo<RunnerDispute[]>(
    () => order.disputes ?? [],
    [order.disputes],
  );
  const disputableIds = useMemo(
    () => getDisputableePhotoIds(order),
    [order],
  );

  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [reason, setReason] = useState<DisputeReason | null>(null);
  const [note, setNote] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [photosError, setPhotosError] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const noteId = useId();
  const noteCounterId = useId();
  const photoErrorId = useId();
  const wasOpen = useRef(false);

  // Reset state on each open. Modal mount is conditional in consumer
  // (`{refundOrder && <RefundModal ... />}`), but be defensive.
  useEffect(() => {
    if (isOpen && !wasOpen.current) {
      setSelected(new Set());
      setReason(null);
      setNote("");
      setPhotosError(null);
      setSubmitError(null);
      setIsSubmitting(false);
    }
    wasOpen.current = isOpen;
  }, [isOpen]);

  function togglePhoto(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
    if (photosError) setPhotosError(null);
  }

  const canSubmit = selected.size > 0 && reason !== null && !isSubmitting;
  const noPhotosLeft = disputableIds.length === 0;

  async function handleSubmit() {
    if (selected.size === 0) {
      setPhotosError("Pick at least one photo to refund.");
      return;
    }
    if (!reason) return;

    setIsSubmitting(true);
    setSubmitError(null);

    const photoIds = Array.from(selected);
    const trimmedNote = note.trim();

    try {
      // Server creates the canonical Dispute records + admin queue rows.
      // We then invalidate the orders cache so the receipt chip + photo
      // locks come from the freshly-fetched order.disputes payload — no
      // local-store mirror to drift out of sync with admin actions.
      await submitOrderRefund({
        orderId: order.id,
        photoIds,
        reason,
        note: trimmedNote,
      });

      await queryClient.invalidateQueries({ queryKey: ["me", "orders"] });

      const count = photoIds.length;
      showToast({
        kind: "success",
        message: `Refund request sent for ${count} photo${count === 1 ? "" : "s"}. We'll review within ${REFUND_PROCESSING_DAYS} business day${REFUND_PROCESSING_DAYS === 1 ? "" : "s"}.`,
        duration: 5000,
      });
      onClose();
    } catch (err) {
      const message =
        err instanceof ApiError
          ? err.message
          : "We couldn't send your request. Try again in a moment.";
      setSubmitError(message);
      setIsSubmitting(false);
    }
  }

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Request a refund">
      <div className="space-y-6">
        <div>
          <Kicker as="p" tnum>
            Order · {order.id} · {formatLongDate(order.paidAt)}
          </Kicker>
          <p className="mt-2 font-sans text-sm text-ink-soft">
            <span className="font-display text-base text-ink tnum">
              {formatPrice(order.total)}
            </span>{" "}
            paid for{" "}
            <span className="font-mono tnum">{order.photoIds.length}</span>{" "}
            photo{order.photoIds.length === 1 ? "" : "s"} from{" "}
            <span className="text-ink">{eventName}</span>.
          </p>
        </div>

        <details className="group border border-line rounded-xl">
          <summary className="cursor-pointer list-none px-4 py-3 flex items-center justify-between">
            <Kicker as="span" tone="default">
              Read our refund policy
            </Kicker>
            <span
              aria-hidden="true"
              className="font-mono text-slate transition-transform group-open:rotate-90"
            >
              →
            </span>
          </summary>
          <div className="px-4 pb-5 pt-1 border-t border-line/60">
            <RefundPolicyContent />
          </div>
        </details>

        <fieldset>
          <legend className="block">
            <Kicker as="span">
              Pick the photos to refund
            </Kicker>
          </legend>
          <ul
            className="mt-3 border border-line rounded-xl divide-y divide-line"
            aria-describedby={photosError ? photoErrorId : undefined}
          >
            {order.photoIds.map((id) => {
              const lockedDispute = orderDisputes.find(
                (d) =>
                  d.photoId === id &&
                  d.status !== "denied" &&
                  d.status !== "withdrawn",
              );
              const isLocked = !!lockedDispute;
              const isChecked = selected.has(id);
              return (
                <li key={id} className="px-4 py-3">
                  <label
                    className={cn(
                      "flex items-center gap-3 cursor-pointer",
                      isLocked && "cursor-not-allowed opacity-60",
                    )}
                  >
                    <input
                      type="checkbox"
                      checked={isChecked}
                      disabled={isLocked || isSubmitting}
                      onChange={() => togglePhoto(id)}
                      className="size-4 accent-ink rounded border-line"
                    />
                    <span className="flex-1 min-w-0 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
                      <span className="font-mono text-sm text-ink tnum truncate">
                        {id.replace(/^mock-/, "").toUpperCase()}
                      </span>
                      {isLocked && (
                        <Kicker tone="soft">
                          Refund {refundChipLabel(lockedDispute.status)}
                        </Kicker>
                      )}
                    </span>
                  </label>
                </li>
              );
            })}
          </ul>
          <FieldError
            id={photoErrorId}
            message={photosError}
            density="tight"
          />
          {noPhotosLeft && !photosError && (
            <p className="mt-3 font-sans text-sm text-slate">
              Every photo on this order is already in a refund flow.
            </p>
          )}
        </fieldset>

        <div>
          <Kicker as="p" className="mb-3">
            Reason
          </Kicker>
          <Dropdown
            ariaLabel="Refund reason"
            trigger={
              <span
                className={cn(
                  "inline-flex w-full items-center justify-between gap-3 rounded-xl border border-line px-4 py-3 font-sans text-sm transition-colors hover:border-ink/40",
                  reason ? "text-ink" : "text-slate",
                )}
              >
                <span className="truncate">
                  {reason
                    ? DISPUTE_REASON_LABEL[reason]
                    : "Choose a reason…"}
                </span>
                <svg
                  className="size-3 shrink-0 text-slate"
                  viewBox="0 0 16 16"
                  fill="none"
                  aria-hidden="true"
                >
                  <path
                    d="M4 6 L8 10 L12 6"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </span>
            }
            panelClassName="z-[70]"
          >
            {REASON_ORDER.map((r) => (
              <DropdownItem
                key={r}
                onClick={() => setReason(r)}
                active={reason === r}
              >
                {DISPUTE_REASON_LABEL[r]}
              </DropdownItem>
            ))}
          </Dropdown>
        </div>

        <div>
          <label htmlFor={noteId} className="block">
            <Kicker as="span">Note (optional)</Kicker>
          </label>
          <textarea
            id={noteId}
            value={note}
            onChange={(e) => setNote(e.target.value.slice(0, NOTE_MAX))}
            disabled={isSubmitting}
            rows={4}
            placeholder="Tell us anything that'll help us review faster."
            aria-describedby={noteCounterId}
            className="mt-3 block w-full rounded-xl border border-line bg-bone-deep/40 px-4 py-3 font-sans text-sm text-ink placeholder:text-slate-soft focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone resize-none"
          />
          <CharCounter
            id={noteCounterId}
            current={note.length}
            max={NOTE_MAX}
            className="mt-1.5"
          />
        </div>

        {submitError && (
          <p className="font-sans text-sm text-error" role="alert">
            {submitError}
          </p>
        )}

        <div className="flex flex-wrap items-center justify-end gap-x-6 gap-y-3 pt-2 border-t border-line/60">
          <button
            type="button"
            onClick={onClose}
            disabled={isSubmitting}
            className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate hover:text-ink transition-colors disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleSubmit}
            disabled={!canSubmit}
            className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] border border-line rounded-full px-6 py-3 text-ink hover:bg-ink hover:text-bone transition-colors disabled:opacity-50 disabled:hover:bg-transparent disabled:hover:text-ink"
          >
            {isSubmitting ? "Sending…" : "Send request"}
          </button>
        </div>
      </div>
    </Modal>
  );
}

function refundChipLabel(status: RunnerDisputeStatus): string {
  switch (status) {
    case "open":
      return "pending";
    case "escalated":
      return "in review";
    case "resolved":
      return "approved";
    case "denied":
      return "denied";
    case "withdrawn":
      return "withdrawn";
    default:
      return "in flight";
  }
}
