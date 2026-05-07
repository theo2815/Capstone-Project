"use client";

import { useMemo } from "react";
import Link from "next/link";
import { Modal } from "@/components/ui/modal";
import { Kicker } from "@/components/ui/kicker";
import {
  MESSAGE_KIND_LABEL,
  getEffectiveMessages,
  type PhotographerMessage,
  type PhotographerMessageKind,
} from "@/lib/photographer-messages";
import {
  markAllEffectiveRead,
  usePhotographerMessageStore,
} from "@/store/photographer-message-store";
import { useConfirmation } from "@/hooks/use-confirmation";
import { ROUTES } from "@/lib/constants";
import { formatLongDate } from "@/lib/format";
import { cn } from "@/lib/utils";

interface PhotographerInboxModalProps {
  isOpen: boolean;
  onClose: () => void;
  photographerId: string;
}

const KIND_TONE: Record<PhotographerMessageKind, string> = {
  payout_held: "text-warning",
  report_acknowledged: "text-slate",
  report_resolved: "text-fresh",
  verification_approved: "text-fresh",
  verification_rejected: "text-warning",
  account_suspended: "text-error",
  account_unsuspended: "text-fresh",
  admin_message: "text-ink",
};

export function PhotographerInboxModal({
  isOpen,
  onClose,
  photographerId,
}: PhotographerInboxModalProps) {
  const submissions = usePhotographerMessageStore((s) => s.submissions);
  const overrides = usePhotographerMessageStore((s) => s.overrides);
  const markRead = usePhotographerMessageStore((s) => s.markRead);
  const removeMessage = usePhotographerMessageStore((s) => s.removeMessage);
  const { confirm } = useConfirmation();

  const messages = useMemo(() => {
    const all = getEffectiveMessages(submissions, overrides);
    return all
      .filter((m) => m.photographerId === photographerId)
      .sort((a, b) => b.createdAt.localeCompare(a.createdAt));
  }, [submissions, overrides, photographerId]);

  const unreadCount = messages.filter((m) => m.readAt === null).length;

  function handleMarkAll() {
    markAllEffectiveRead(messages);
  }

  async function handleRemove(message: PhotographerMessage) {
    const ok = await confirm({
      title: "Remove this notification?",
      message:
        "This message will be cleared from your inbox. Admin still has the underlying record on the payout cycle.",
      confirmLabel: "Remove",
      cancelLabel: "Cancel",
      danger: true,
    });
    if (!ok) return;
    removeMessage(message.id);
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
            onClick={handleMarkAll}
            className="font-mono uppercase tracking-[0.22em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate hover:text-ink transition-colors"
          >
            Mark all read
          </button>
        )}
      </div>

      {messages.length === 0 ? (
        <p className="font-sans text-sm text-slate-soft py-8 text-center">
          No messages yet. Admin actions on your payouts will land here.
        </p>
      ) : (
        <ul className="max-h-[60vh] overflow-y-auto -mx-2 divide-y divide-line">
          {messages.map((m) => (
            <InboxRow
              key={m.id}
              message={m}
              onMarkRead={() => {
                if (m.readAt === null) markRead(m.id);
              }}
              onRemove={() => handleRemove(m)}
              onViewCycle={onClose}
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
  onViewCycle: () => void;
}

function InboxRow({ message, onMarkRead, onRemove, onViewCycle }: InboxRowProps) {
  const isUnread = message.readAt === null;

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
        {message.title}
      </p>
      <p className="font-sans text-sm text-slate mt-1.5 leading-relaxed">
        {message.body}
      </p>
      <div className="mt-3 flex items-center justify-between gap-3">
        {message.cta ? (
          <Link
            href={message.cta.href}
            onClick={(e) => {
              e.stopPropagation();
              onViewCycle();
            }}
            className="inline-flex items-center gap-1"
          >
            <Kicker
              tone="active"
              className="underline decoration-fresh/40 underline-offset-4 decoration-1 hover:decoration-fresh"
            >
              {message.cta.label}
            </Kicker>
            <span aria-hidden="true" className="text-fresh">
              →
            </span>
          </Link>
        ) : message.payoutCycleId ? (
          <Link
            href={`${ROUTES.DASHBOARD_BILLING}#cycle-${message.payoutCycleId}`}
            onClick={(e) => {
              e.stopPropagation();
              onViewCycle();
            }}
            className="inline-flex items-center gap-1"
          >
            <Kicker
              tone="active"
              className="underline decoration-fresh/40 underline-offset-4 decoration-1 hover:decoration-fresh"
            >
              View cycle
            </Kicker>
            <span aria-hidden="true" className="text-fresh">
              →
            </span>
          </Link>
        ) : (
          <span aria-hidden="true" />
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
