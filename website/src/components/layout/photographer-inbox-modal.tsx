"use client";

import { Modal } from "@/components/ui/modal";
import { Kicker } from "@/components/ui/kicker";
import {
  MESSAGE_KIND_LABEL,
  getUnreadCount,
  resolveMessageTitle,
  type PhotographerMessage,
  type PhotographerMessageKind,
} from "@/lib/photographer-messages";
import { useMyPhotographerMessages } from "@/hooks/use-inbox-messages";
import { useConfirmationStore } from "@/store/confirmation-store";
import { useConfirmation } from "@/hooks/use-confirmation";
import { formatLongDate } from "@/lib/format";
import { BTN_GHOST, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { cn } from "@/lib/utils";

interface PhotographerInboxModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const KIND_TONE: Record<PhotographerMessageKind, string> = {
  verification_approved: "text-fresh",
  verification_rejected: "text-warning",
  verification_reset: "text-warning",
  event_approved: "text-fresh",
  event_rejected: "text-warning",
  event_change_approved: "text-fresh",
  event_change_rejected: "text-warning",
  suspended: "text-error",
  unsuspended: "text-fresh",
  force_edit: "text-slate",
  dispute_resolved: "text-fresh",
  dispute_denied: "text-slate",
  dispute_escalated: "text-warning",
  payout_approved: "text-fresh",
  payout_held: "text-warning",
  payout_paid: "text-fresh",
  payout_report_acknowledged: "text-slate",
  payout_report_resolved: "text-fresh",
  admin_message: "text-ink",
};

export function PhotographerInboxModal({
  isOpen,
  onClose,
}: PhotographerInboxModalProps) {
  const { messages, total, hasMore, loadMore, markRead, markAllRead, remove, pending, loading, loadingMore, error, actionError, refetch } =
    useMyPhotographerMessages(isOpen);
  const { confirm } = useConfirmation();
  const confirmationOpen = useConfirmationStore((s) => s.active !== null);

  const unreadCount = getUnreadCount(messages);
  const totalLabel = total ?? messages.length;

  async function handleRemove(message: PhotographerMessage) {
    const ok = await confirm({
      title: "Remove this notification?",
      message:
        "This message will be cleared from your inbox. Admin still has the underlying record on the decision log.",
      confirmLabel: "Remove",
      cancelLabel: "Cancel",
      danger: true,
    });
    if (!ok) return;
    await remove(message.id);
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
          No messages yet. Admin actions on your account will land here.
        </p>
      )) : (
        <div className="max-h-[60vh] overflow-y-auto -mx-2">
          <ul className="divide-y divide-line">
            {messages.map((m) => (
              <InboxRow
                key={m.id}
                message={m}
                pending={pending}
                onMarkRead={() => {
                  if (m.readAt === null) void markRead(m.id);
                }}
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
  message: PhotographerMessage;
  pending: boolean;
  onMarkRead: () => void;
  onRemove: () => void;
}

function InboxRow({ message, pending, onMarkRead, onRemove }: InboxRowProps) {
  const isUnread = message.readAt === null;
  const title = resolveMessageTitle(message);

  return (
    <li
      className={cn(
        "px-2 py-4 transition-colors",
        isUnread ? "bg-bone-deep/40 hover:bg-bone-deep" : "hover:bg-bone-deep/40",
      )}
    >
      <button
        type="button"
        disabled={pending || !isUnread}
        onClick={onMarkRead}
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
            {MESSAGE_KIND_LABEL[message.kind]}
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
      <div className="mt-3 flex items-center justify-end">
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
