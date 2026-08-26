"use client";

import { useState } from "react";
import { useAuth } from "@/hooks/use-auth";
import { useEffectiveRole } from "@/hooks/use-effective-role";
import { getUnreadCount } from "@/lib/photographer-messages";
import { useMyPhotographerMessages } from "@/lib/me-photographer-messages-data";
import { usePhotographerNotificationsWs } from "@/hooks/use-photographer-notifications-ws";
import { PhotographerInboxModal } from "@/components/layout/photographer-inbox-modal";

// Bell + unread-count badge mounted in <SiteHeader>. Renders nothing for
// non-photographer roles, when not authenticated, or when the photographer
// has zero messages.
//
// Reads from GET /me/photographer/messages via useMyPhotographerMessages
// — every admin action (approve/reject/suspend/unsuspend/reset/
// dispute resolve|deny|escalate/payout approve|hold|paid/report ack|resolve
// /admin DM) writes a row server-side via AdminDecisionLogService.pushMessage,
// so the badge surfaces them all without any per-action FE wiring.
//
// Real-time path: usePhotographerNotificationsWs opens a WebSocket to
// /ws/me/photographer/notifications and feeds every push into the same
// store the bell renders from. While the WS is up, the REST poll relaxes
// to 5min; on disconnect it snaps back to 30s. Both paths land in
// applyPush which dedupes by id, so cross-path collisions are harmless.

export function NotificationBell() {
  const { isAuthenticated, isLoading } = useAuth();
  // Effective role: hide the photographer inbox bell (and silence its WS/poll)
  // when a photographer is in runner view — no photographer chrome in runner
  // mode. The runner bell stays gated on the TRUE role (the runner inbox is
  // RUNNER-only server-side), so a photographer in runner view simply has no
  // bell, which is correct — they have no runner inbox.
  const effectiveRole = useEffectiveRole();
  const isPhotographer = effectiveRole === "PHOTOGRAPHER";
  const enabled = isAuthenticated && isPhotographer;
  // Mount BEFORE the early return — the WS must stay alive even when the
  // bell hides itself (zero messages, modal closed). Without this, a fresh
  // photographer would never receive the WS push for their first message.
  usePhotographerNotificationsWs(enabled);
  const { messages } = useMyPhotographerMessages(enabled);
  const [open, setOpen] = useState(false);

  const unread = getUnreadCount(messages);
  const total = messages.length;

  if (isLoading || !isAuthenticated || !isPhotographer) return null;
  // Stay mounted while the modal is open even if the inbox empties out
  // mid-session (last-message removal) — otherwise the modal unmounts
  // abruptly. Bell hides on next paint after the modal closes.
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
      <PhotographerInboxModal isOpen={open} onClose={() => setOpen(false)} />
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
