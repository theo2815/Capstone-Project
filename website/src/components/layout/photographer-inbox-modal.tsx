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
import { useMyPhotographerMessages } from "@/lib/me-photographer-messages-data";
import { useConfirmation } from "@/hooks/use-confirmation";
import { formatLongDate } from "@/lib/format";
import { cn } from "@/lib/utils";

interface PhotographerInboxModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const KIND_TONE: Record<PhotographerMessageKind, string> = {
  verification_approved: "text-fresh",
  verification_rejected: "text-warning",
  verification_reset: "text-warning",
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
  const { messages, markRead, markAllRead, remove } =
    useMyPhotographerMessages(isOpen);
  const { confirm } = useConfirmation();

  const unreadCount = getUnreadCount(messages);

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
    void remove(message.id);
  }

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Inbox">
      <div className="flex items-baseline justify-between mb-5">
        <Kicker as="p" tone="soft" tnum>
          {unreadCount > 0
            ? `${unreadCount} unread · ${messages.length} total`
            : `${messages.length} total`}
        </Kicker>
        {unreadCount > 0 && (
          <button
            type="button"
            onClick={() => void markAllRead()}
            className="font-mono uppercase tracking-[0.22em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate hover:text-ink transition-colors"
          >
            Mark all read
          </button>
        )}
      </div>

      {messages.length === 0 ? (
        <p className="font-sans text-sm text-slate-soft py-8 text-center">
          No messages yet. Admin actions on your account will land here.
        </p>
      ) : (
        <ul className="max-h-[60vh] overflow-y-auto -mx-2 divide-y divide-line">
          {messages.map((m) => (
            <InboxRow
              key={m.id}
              message={m}
              onMarkRead={() => {
                if (m.readAt === null) void markRead(m.id);
              }}
              onRemove={() => handleRemove(m)}
            />
          ))}
        </ul>
      )}

      <div className="mt-6 flex justify-end">
        <button
          type="button"
          onClick={onClose}
          className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-ink border border-line hover:bg-ink hover:text-bone hover:border-ink transition-colors rounded-full px-5 py-2"
        >
          Close
        </button>
      </div>
    </Modal>
  );
}

interface InboxRowProps {
  message: PhotographerMessage;
  onMarkRead: () => void;
  onRemove: () => void;
}

function InboxRow({ message, onMarkRead, onRemove }: InboxRowProps) {
  const isUnread = message.readAt === null;
  const title = resolveMessageTitle(message);

  return (
    <li
      onClick={onMarkRead}
      className={cn(
        "px-2 py-4 cursor-pointer transition-colors",
        isUnread ? "bg-bone-deep/40 hover:bg-bone-deep" : "hover:bg-bone-deep/40",
      )}
    >
      <div className="flex items-baseline justify-between gap-3">
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
      <div className="mt-3 flex items-center justify-end">
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onRemove();
          }}
          aria-label="Remove notification"
          className="font-mono uppercase tracking-[0.22em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft hover:text-error transition-colors"
        >
          Remove
        </button>
      </div>
    </li>
  );
}
