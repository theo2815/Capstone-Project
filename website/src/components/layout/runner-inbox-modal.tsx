"use client";

import { useRouter } from "next/navigation";
import { Modal } from "@/components/ui/modal";
import { Kicker } from "@/components/ui/kicker";
import {
  RUNNER_MESSAGE_KIND_LABEL,
  getRunnerUnreadCount,
  resolveRunnerMessageTitle,
  type RunnerMessage,
  type RunnerMessageKind,
} from "@/lib/runner-messages";
import { useMyRunnerMessages } from "@/hooks/use-inbox-messages";
import { useConfirmationStore } from "@/store/confirmation-store";
import { useConfirmation } from "@/hooks/use-confirmation";
import { ROUTES } from "@/lib/constants";
import { formatLongDate } from "@/lib/format";
import { BTN_GHOST, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { cn } from "@/lib/utils";

interface RunnerInboxModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const KIND_TONE: Record<RunnerMessageKind, string> = {
  dispute_resolved: "text-fresh",
  dispute_denied: "text-error",
  dispute_escalated: "text-warning",
  admin_message: "text-ink",
  account_suspended: "text-error",
  account_unsuspended: "text-fresh",
  photo_removed: "text-warning",
};

export function RunnerInboxModal({ isOpen, onClose }: RunnerInboxModalProps) {
  const router = useRouter();
  const { messages, total, hasMore, loadMore, markRead, markAllRead, remove, pending, loading, loadingMore, error, actionError, refetch } =
    useMyRunnerMessages(isOpen);
  const { confirm } = useConfirmation();
  const confirmationOpen = useConfirmationStore((s) => s.active !== null);

  const unreadCount = getRunnerUnreadCount(messages);
  const totalLabel = total ?? messages.length;

  async function handleRemove(message: RunnerMessage) {
    const ok = await confirm({
      title: "Remove this notification?",
      message:
        "This message will be cleared from your inbox. Admin still has the underlying record.",
      confirmLabel: "Remove",
      cancelLabel: "Cancel",
      danger: true,
    });
    if (!ok) return;
    await remove(message.id);
  }

  // Clicking a row marks read + navigates. Order-bearing messages (every
  // dispute outcome) deep-link to /orders?expand={orderId} so the runner
  // lands on the affected receipt with the timeline already open. Order-
  // less kinds (admin_message, account_suspended) only mark read.
  async function handleRowClick(message: RunnerMessage) {
    if (pending) return;
    if (message.readAt === null && !(await markRead(message.id))) return;
    if (message.orderId) {
      router.push(`${ROUTES.ORDERS}?expand=${encodeURIComponent(message.orderId)}`);
      onClose();
    }
  }

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Inbox" escDisabled={confirmationOpen}>
      <div className="flex flex-wrap items-baseline justify-between gap-2 mb-5">
        <Kicker as="p" tone="soft" tnum>
          {unreadCount > 0
            ? `${unreadCount} unread · ${totalLabel} total`
            : `${totalLabel} total`}
        </Kicker>
        {unreadCount > 0 && (
          <button
            type="button"
            disabled={pending}
            onClick={() => void markAllRead()}
            className={cn(BTN_GHOST, BTN_SIZE.sm)}
          >
            Mark all read
          </button>
        )}
      </div>

      {actionError && <p role="alert" className="mb-4 text-sm text-error">{actionError}</p>}
      {error && (
        <div role="alert" className="mb-4 text-sm text-error">
          <p>{error}</p>
          <button type="button" disabled={loading || pending} onClick={() => void refetch()} className={cn(BTN_GHOST, BTN_SIZE.sm)}>
            Retry
          </button>
        </div>
      )}
      {loading && messages.length === 0 ? (
        <p role="status" className="py-8 text-center text-sm text-slate">Loading inbox…</p>
      ) : messages.length === 0 ? (!error && (
        <p className="font-sans text-sm text-slate-soft py-8 text-center">
          No messages yet. Updates on your refund requests will land here.
        </p>
      )) : (
        <div className="max-h-[60vh] overflow-y-auto -mx-2">
          <ul className="divide-y divide-line">
            {messages.map((m) => (
              <InboxRow
                key={m.id}
                message={m}
                pending={pending}
                onClick={() => handleRowClick(m)}
                onRemove={() => handleRemove(m)}
              />
            ))}
          </ul>
          {hasMore && (
            <div className="mt-3 flex justify-center">
              <button
                type="button"
                disabled={loading || pending}
                onClick={() => void loadMore()}
                className={cn(BTN_GHOST, BTN_SIZE.sm)}
              >
                {loadingMore ? "Loading…" : "Load older"}
              </button>
            </div>
          )}
        </div>
      )}

      <div className="mt-6 flex justify-end">
        <button
          type="button"
          onClick={onClose}
          className={cn(BTN_SECONDARY, BTN_SIZE.sm)}
        >
          Close
        </button>
      </div>
    </Modal>
  );
}

interface InboxRowProps {
  message: RunnerMessage;
  pending: boolean;
  onClick: () => void;
  onRemove: () => void;
}

function InboxRow({ message, pending, onClick, onRemove }: InboxRowProps) {
  const isUnread = message.readAt === null;
  const title = resolveRunnerMessageTitle(message);
  const clickable = isUnread || !!message.orderId;

  return (
    <li
      className={cn(
        "px-2 py-4 transition-colors",
        isUnread
          ? "bg-bone-deep/40 hover:bg-bone-deep"
          : message.orderId
            ? "hover:bg-bone-deep/40"
            : "",
      )}
    >
      <button
        type="button"
        disabled={pending || !clickable}
        onClick={onClick}
        className={cn("block w-full text-left rounded focus-visible:outline-fresh", pending && "opacity-50 cursor-not-allowed")}
      >
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <Kicker
            tone="soft"
            className={cn("flex items-center gap-2", KIND_TONE[message.kind])}
          >
            {isUnread && (
              <span
                className="size-1.5 rounded-full bg-fresh shrink-0"
                aria-label="Unread"
              />
            )}
            {RUNNER_MESSAGE_KIND_LABEL[message.kind]}
          </Kicker>
          <Kicker as="time" tone="soft" tnum>
            {formatLongDate(message.createdAt)}
          </Kicker>
        </div>
        <p
          className={cn(
            "font-sans text-base mt-2",
            isUnread ? "text-ink font-semibold" : "text-ink-soft",
          )}
        >
          {title}
        </p>
        <p className="font-sans text-sm text-slate mt-1.5 leading-relaxed">
          {message.body}
        </p>
      </button>
      <div className="mt-3 flex items-center justify-between">
        {message.orderId ? (
          <button type="button" disabled={pending} onClick={onClick} className={cn(BTN_GHOST, BTN_SIZE.sm)}>
            View receipt →
          </button>
        ) : (
          <span />
        )}
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onRemove();
          }}
          disabled={pending}
          aria-label="Remove notification"
          className={cn(BTN_GHOST, BTN_SIZE.sm, "hover:text-error")}
        >
          Remove
        </button>
      </div>
    </li>
  );
}
