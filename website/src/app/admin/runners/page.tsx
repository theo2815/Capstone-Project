"use client";

import { useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { Kicker } from "@/components/ui/kicker";
import { Skeleton } from "@/components/ui/skeleton";
import { AdminStatusPill } from "@/components/admin/admin-status-pill";
import { AdminSuspendModal, RUNNER_SUSPEND_REASONS } from "@/components/admin/admin-suspend-modal";
import { AdminMessageModal } from "@/components/admin/admin-message-modal";
import { useAdminRunnersPage, useAdminUserActions } from "@/hooks/use-admin-users";
import { useSearchFocusShortcut } from "@/hooks/use-admin-keyboard";
import { useUrlState } from "@/hooks/use-url-state";
import { useToast } from "@/hooks/use-toast";
import type { AdminUserRow } from "@/lib/admin-user-registry";
import { formatMemberSince, sentenceCaseDate } from "@/lib/format";
import { BTN_DANGER, BTN_GHOST, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { cn } from "@/lib/utils";

// AX-41 (M1): find a runner on the server, then suspend, unsuspend or message
// them with the same modals and actions the photographer detail page uses.
export default function AdminRunnersPage() {
  const [query, setQuery] = useUrlState<string>("q", "", { debounceMs: 300 });
  // The server search follows the URL, which trails typing by the 300 ms debounce.
  const q = (useSearchParams().get("q") ?? "").trim();
  const list = useAdminRunnersPage(q);
  const failed = !!list.error && list.items.length === 0;
  const searchInputRef = useRef<HTMLInputElement | null>(null);
  useSearchFocusShortcut({ active: true, inputRef: searchInputRef });

  const { suspend, unsuspend, message, messageRecoveryNotice, isPending, pendingAction } = useAdminUserActions();
  const { showToast } = useToast();
  // The last row stays after close, so a closing modal keeps its name and a draft stays with its runner.
  const [suspendRow, setSuspendRow] = useState<AdminUserRow | null>(null);
  const [messageRow, setMessageRow] = useState<AdminUserRow | null>(null);
  const [openDialog, setOpenDialog] = useState<"suspend" | "message" | null>(null);
  const close = () => setOpenDialog(null);

  async function handleSuspend(reason: string) {
    const row = suspendRow!;
    const ok = await suspend(row.userId, reason);
    if (ok) {
      close();
      showToast({ kind: "success", message: `Suspended · ${row.name}` });
    }
    return ok;
  }

  async function handleUnsuspend(row: AdminUserRow) {
    if (await unsuspend(row.userId)) showToast({ kind: "success", message: `Unsuspended · ${row.name}` });
  }

  async function handleMessage(payload: { subject: string; body: string }) {
    const row = messageRow!;
    const ok = await message(row.userId, payload);
    if (ok) {
      close();
      showToast({ kind: "success", message: `Message sent · ${row.name}` });
    }
    return ok;
  }

  return (
    <>
      <header className="pb-8 md:pb-12 border-b border-line">
        <Kicker as="p">Runners</Kicker>
        <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-[1.05] text-ink mt-3">
          Runners.
        </h1>
        <p className="font-sans text-sm md:text-base text-ink-soft mt-3 max-w-xl">
          Find a runner by name or email, then suspend or message them.
        </p>
      </header>

      <div className="mt-8 md:mt-10">
        <label htmlFor="runner-search" className="sr-only">
          Search runners
        </label>
        <div className="relative">
          <input
            id="runner-search"
            ref={searchInputRef}
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search by name or email"
            className="w-full rounded-2xl border border-line bg-surface px-5 py-3 pr-12 [@media(hover:none)]:pr-5 font-sans text-sm text-ink placeholder:text-slate-soft focus:outline-none focus:ring-2 focus:ring-fresh focus:border-fresh"
          />
          <kbd aria-hidden className="pointer-events-none absolute right-4 top-1/2 -translate-y-1/2 font-mono text-[12px] text-slate-soft rounded border border-line px-1.5 py-0.5 [@media(hover:none)]:hidden">
            /
          </kbd>
        </div>
        <Kicker as="p" tone="soft" role="status" aria-live="polite" className="mt-3 tnum">
          {q.length > 0 && !list.isLoading && !failed
            ? `${list.total} match${list.total === 1 ? "" : "es"}`
            : ""}
        </Kicker>
      </div>

      {failed ? (
        <div role="alert" className="mt-10 space-y-3">
          <p className="font-sans text-sm text-error">
            Couldn&apos;t load runners. Please try again.
          </p>
          <button
            type="button"
            onClick={list.refetch}
            disabled={list.isFetching}
            className={cn(BTN_SECONDARY, BTN_SIZE.sm)}
          >
            Retry
          </button>
        </div>
      ) : list.isLoading ? (
        <div className="mt-6 space-y-4" aria-busy="true">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="h-24 w-full rounded-2xl" />
          ))}
        </div>
      ) : list.items.length === 0 ? (
        <p className="mt-6 font-sans text-sm text-slate-soft">
          {q ? "No runner matches this search." : "No runners yet."}
        </p>
      ) : (
        <>
          <ul className="mt-6 space-y-4">
            {list.items.map((row) => {
              const busy = isPending([row.userId]);
              const suspended = row.suspendedAt !== null;
              return (
                <li key={row.userId} className="rounded-2xl border border-line bg-bone p-5 md:p-6">
                  <div className="flex items-start gap-4 flex-wrap">
                    <div className="min-w-0 flex-1">
                      <h3 className="font-display text-lg md:text-xl font-medium text-ink break-words">{row.name}</h3>
                      <p className="font-sans text-sm text-slate mt-1 break-all">{row.email}</p>
                      <p className="font-sans text-sm text-slate-soft mt-1 tnum">
                        Joined {sentenceCaseDate(formatMemberSince(row.createdAt))}
                      </p>
                    </div>
                    <AdminStatusPill label={suspended ? "Suspended" : "Active"} tone={suspended ? "ink" : "slate"} />
                  </div>
                  {suspended && row.suspensionReason && (
                    <p className="font-sans text-sm text-ink-soft mt-3 break-words">Reason: {row.suspensionReason}</p>
                  )}
                  <div className="mt-4 flex flex-wrap gap-3" aria-busy={busy}>
                    {suspended ? (
                      <button type="button" disabled={busy} onClick={() => handleUnsuspend(row)} className={cn(BTN_SECONDARY, BTN_SIZE.sm)}>
                        {pendingAction(row.userId) === "Unsuspending…" ? "Unsuspending…" : "Unsuspend"}
                      </button>
                    ) : (
                      <button type="button" disabled={busy} onClick={() => { setSuspendRow(row); setOpenDialog("suspend"); }} className={cn(BTN_DANGER, BTN_SIZE.sm)}>
                        Suspend…
                      </button>
                    )}
                    <button type="button" disabled={busy} onClick={() => { setMessageRow(row); setOpenDialog("message"); }} className={cn(BTN_GHOST, BTN_SIZE.sm)}>
                      Send message…
                    </button>
                  </div>
                </li>
              );
            })}
          </ul>
          <LoadMoreButton
            shown={list.items.length}
            total={list.total}
            increment={50}
            onLoadMore={list.fetchNextPage}
            isLoading={list.isFetchingNextPage}
            label="Load more"
          />
        </>
      )}

      {suspendRow && (
        <AdminSuspendModal
          open={openDialog === "suspend"}
          submitting={isPending([suspendRow.userId])}
          onClose={close}
          onSubmit={handleSuspend}
          photographerName={suspendRow.name}
          reasons={RUNNER_SUSPEND_REASONS}
          intro={`${suspendRow.name} will be signed out and unable to sign in until you unsuspend. They get a message with the reason, and the reason is logged.`}
        />
      )}
      {messageRow && (
        <AdminMessageModal
          key={messageRow.userId}
          open={openDialog === "message"}
          submitting={isPending([messageRow.userId])}
          recoveryNotice={messageRecoveryNotice(messageRow.userId)}
          onClose={close}
          onSubmit={handleMessage}
          photographerName={messageRow.name}
        />
      )}
    </>
  );
}
