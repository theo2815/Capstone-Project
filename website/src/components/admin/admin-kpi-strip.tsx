"use client";

import Link from "next/link";
import { Kicker } from "@/components/ui/kicker";
import { Skeleton } from "@/components/ui/skeleton";
import { ROUTES } from "@/lib/constants";
import { cn } from "@/lib/utils";
import { useAdminKpisQuery } from "@/hooks/use-admin-data";
import {
  ADMIN_QUEUES,
  useAdminAttentionTarget,
  useAdminQueueCounts,
} from "@/lib/admin-queues";
import { isInboxQueueType } from "@/components/admin/admin-inbox-filter-chips";

// Compact at-a-glance metrics row above the inbox queue: every queue in
// ADMIN_QUEUES order, with its count from GET /admin/kpis. Queues with an
// Inbox chip link into /admin/inbox?type=…; the rest link to their own page.
// A count the backend does not send hides its entry. While the KPIs load it
// shows a skeleton, never numbers worked out in the browser; if they fail,
// the strip is left out.
//
// The attention queue (the first with work) carries the single breathing
// fresh dot, the same one the rail lights (one-fresh-per-viewport).
export function AdminKpiStrip() {
  const { data: kpis, isPending } = useAdminKpisQuery();
  const { byKey } = useAdminQueueCounts();
  const attention = useAdminAttentionTarget();
  if (!isPending && !kpis) return null;

  return (
    <nav
      aria-label="Queue counts"
      aria-busy={isPending}
      className="border-y border-line"
    >
      {/* One scrolling row on phones so the queue starts within the first screen. */}
      <ul className="flex flex-nowrap md:flex-wrap overflow-x-auto md:overflow-visible [scrollbar-width:none] [&::-webkit-scrollbar]:hidden items-baseline gap-x-8 md:gap-x-12 gap-y-4 py-4 md:py-5 [&>li]:shrink-0 [&>li]:whitespace-nowrap [mask-image:linear-gradient(to_right,black_85%,transparent)] md:[mask-image:none] after:content-[''] after:w-[15%] after:shrink-0 md:after:hidden">
        {isPending
          ? ADMIN_QUEUES.map((queue) => (
              <li key={queue.key}>
                <Skeleton className="h-6 w-28" />
              </li>
            ))
          : ADMIN_QUEUES.map((queue) => {
              const count = byKey[queue.key];
              if (count === null) return null;
              const labelText = count === 1 ? queue.singularLabel : queue.label;
              const href = isInboxQueueType(queue.inboxType)
                ? `${ROUTES.ADMIN_INBOX}?type=${queue.inboxType}`
                : queue.href;
              return (
                <li key={queue.key}>
                  <Link
                    href={href}
                    aria-label={`${count} ${labelText.toLowerCase()}`}
                    className="group inline-flex items-center gap-2.5 min-h-[44px]"
                  >
                    <span
                      className={cn(
                        "font-display text-2xl tnum leading-none transition-colors",
                        count === 0 ? "text-slate-soft" : "text-ink",
                      )}
                    >
                      {count}
                    </span>
                    <Kicker
                      tone={count === 0 ? "soft" : "default"}
                      className="group-hover:text-ink transition-colors"
                    >
                      {labelText}
                    </Kicker>
                    {attention?.key === queue.key && (
                      <span
                        aria-hidden="true"
                        className="size-1.5 rounded-full bg-fresh breathe"
                      />
                    )}
                  </Link>
                </li>
              );
            })}
      </ul>
    </nav>
  );
}
