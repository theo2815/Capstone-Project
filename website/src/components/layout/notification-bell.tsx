"use client";

import { useMemo, useState } from "react";
import { useAuth } from "@/hooks/use-auth";
import { resolveCurrentPhotographer } from "@/lib/current-photographer";
import {
  getEffectiveMessages,
  getUnreadCount,
} from "@/lib/photographer-messages";
import { usePhotographerMessageStore } from "@/store/photographer-message-store";
import { PhotographerInboxModal } from "@/components/layout/photographer-inbox-modal";

// Bell + unread-count badge mounted in <SiteHeader>. Renders nothing for
// non-photographer roles, when not authenticated, or when the photographer
// has zero messages (fully empty inbox). SiteHeader stays role-blind — all
// gating lives here.
//
// Unread count is derived from getEffectiveMessages so seed messages with
// readAt=null contribute to the badge until the user opens the inbox and
// marks them read.

export function NotificationBell() {
  const { user, isAuthenticated, isLoading } = useAuth();
  const submissions = usePhotographerMessageStore((s) => s.submissions);
  const overrides = usePhotographerMessageStore((s) => s.overrides);
  const [open, setOpen] = useState(false);

  const photographer = resolveCurrentPhotographer(user);

  const { unread, total } = useMemo(() => {
    if (!photographer) return { unread: 0, total: 0 };
    const all = getEffectiveMessages(submissions, overrides);
    const mine = all.filter((m) => m.photographerId === photographer.id);
    return {
      unread: getUnreadCount(mine, photographer.id),
      total: mine.length,
    };
  }, [submissions, overrides, photographer]);

  if (isLoading || !isAuthenticated || !photographer) return null;
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
      <PhotographerInboxModal
        isOpen={open}
        onClose={() => setOpen(false)}
        photographerId={photographer.id}
      />
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
