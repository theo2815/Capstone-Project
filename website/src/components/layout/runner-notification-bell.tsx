"use client";

import { useState } from "react";
import { useAuth } from "@/hooks/use-auth";
import { getRunnerUnreadCount } from "@/lib/runner-messages";
import { useMyRunnerMessages } from "@/lib/me-runner-messages-data";
import { useRunnerNotificationsWs } from "@/hooks/use-runner-notifications-ws";
import { RunnerInboxModal } from "@/components/layout/runner-inbox-modal";

// Bell + unread-count badge mounted in <SiteHeader> for authed runners.
// Mirrors NotificationBell (photographer) — reads from
// GET /me/runner/messages, WS-pushes via /ws/me/runner/notifications.
//
// Renders nothing for non-runner roles, when not authenticated, or when
// the runner has zero messages. Stays mounted while the modal is open so
// last-message removal doesn't yank the modal mid-interaction.

export function RunnerNotificationBell() {
  const { user, isAuthenticated, isLoading } = useAuth();
  const isRunner = user?.role === "RUNNER";
  const enabled = isAuthenticated && isRunner;
  // Mount BEFORE the early return so the WS stays alive even when the
  // bell hides itself (zero messages, modal closed). Without this a fresh
  // runner never receives the WS push for their first message.
  useRunnerNotificationsWs(enabled);
  const { messages } = useMyRunnerMessages(enabled);
  const [open, setOpen] = useState(false);

  const unread = getRunnerUnreadCount(messages);
  const total = messages.length;

  if (isLoading || !isAuthenticated || !isRunner) return null;
  if (total === 0 && !open) return null;

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="relative grid size-9 place-items-center rounded-full border border-line text-slate hover:text-ink hover:border-ink transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
        aria-label={
          unread > 0 ? `Inbox · ${unread} unread` : "Inbox · no unread messages"
        }
      >
        <BellIcon />
        {unread > 0 && (
          <span
            aria-hidden="true"
            className="absolute -top-1 -right-1 grid h-[18px] min-w-[18px] place-items-center rounded-full bg-error px-1 font-mono tnum text-[12px] leading-none text-bone"
          >
            {unread > 9 ? "9+" : unread}
          </span>
        )}
      </button>
      <RunnerInboxModal isOpen={open} onClose={() => setOpen(false)} />
    </>
  );
}

function BellIcon() {
  return (
    <svg
      viewBox="0 0 16 16"
      className="size-4"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M3.5 12 L12.5 12 C12.5 12 11.5 11 11.5 9 L11.5 7 C11.5 5 10 3.5 8 3.5 C6 3.5 4.5 5 4.5 7 L4.5 9 C4.5 11 3.5 12 3.5 12 Z" />
      <path d="M6.5 12 C6.5 13 7 13.5 8 13.5 C9 13.5 9.5 13 9.5 12" />
    </svg>
  );
}
