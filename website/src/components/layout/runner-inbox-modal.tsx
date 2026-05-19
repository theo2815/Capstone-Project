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
import { useMyRunnerMessages } from "@/lib/me-runner-messages-data";
import { useConfirmation } from "@/hooks/use-confirmation";
import { ROUTES } from "@/lib/constants";
import { formatLongDate } from "@/lib/format";
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
};

export function RunnerInboxModal({ isOpen, onClose }: RunnerInboxModalProps) {
  const router = useRouter();
  const { messages, markRead, markAllRead, remove } =
    useMyRunnerMessages(isOpen);
  const { confirm } = useConfirmation();

  const unreadCount = getRunnerUnreadCount(messages);

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
    void remove(message.id);
  }

  // Clicking a row marks read + navigates. Order-bearing messages (every
  // dispute outcome) deep-link to /orders?expand={orderId} so the runner
  // lands on the affected receipt with the timeline already open. Order-
  // less kinds (admin_message, account_suspended) only mark read.
  function handleRowClick(message: RunnerMessage) {
    if (message.readAt === null) void markRead(message.id);
    if (message.orderId) {
      router.push(`${ROUTES.ORDERS}?expand=${encodeURIComponent(message.orderId)}`);
      onClose();
    }
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
          No messages yet. Updates on your refund requests will land here.
        </p>
      ) : (
        <ul className="max-h-[60vh] overflow-y-auto -mx-2 divide-y divide-line">
          {messages.map((m) => (
            <InboxRow
              key={m.id}
              message={m}
              onClick={() => handleRowClick(m)}
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
  message: RunnerMessage;
  onClick: () => void;
  onRemove: () => void;
}

function InboxRow({ message, onClick, onRemove }: InboxRowProps) {
  const isUnread = message.readAt === null;
  const title = resolveRunnerMessageTitle(message);
  const clickable = isUnread || !!message.orderId;

  return (
    <li
      onClick={clickable ? onClick : undefined}
      className={cn(
        "px-2 py-4 transition-colors",
        clickable && "cursor-pointer",
        isUnread
          ? "bg-bone-deep/40 hover:bg-bone-deep"
          : message.orderId
            ? "hover:bg-bone-deep/40"
            : "",
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
      <div className="mt-3 flex items-center justify-between">
        {message.orderId ? (
          <Kicker tone="soft" className="text-fresh">
            View receipt →
          </Kicker>
        ) : (
          <span />
        )}
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
