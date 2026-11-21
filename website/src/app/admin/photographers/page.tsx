"use client";

import { useRef } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { Kicker } from "@/components/ui/kicker";
import { AdminPhotographerCard } from "@/components/admin/admin-photographer-card";
import { AdminStatusTabs } from "@/components/admin/admin-status-tabs";
import { useAdminPhotographersPage } from "@/hooks/use-admin-users";
import { useAdminKpis } from "@/hooks/use-admin-data";
import type { AdminUserStatusFilter } from "@/lib/api-admin";
import { photographerHref } from "@/lib/admin-photographer-view";
import {
  useQueueKeyboardNav,
  useSearchFocusShortcut,
} from "@/hooks/use-admin-keyboard";
import { useUrlState } from "@/hooks/use-url-state";
import { Skeleton } from "@/components/ui/skeleton";
import { BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { cn } from "@/lib/utils";

const TABS: ReadonlyArray<{ key: AdminUserStatusFilter; label: string; empty: string }> = [
  { key: "pending", label: "Pending", empty: "Queue is clear." },
  { key: "approved", label: "Approved", empty: "No approved photographers yet." },
  { key: "incomplete", label: "Incomplete", empty: "No incomplete profiles." },
  { key: "rejected", label: "Rejected", empty: "No rejected photographers. A profile rejected now goes back to Incomplete." },
  { key: "suspended", label: "Suspended", empty: "No active suspensions." },
];

// Admin Photographers directory: one status tab at a time, paged and searched
// on the server. Click any row to land on its detail page.
export default function AdminPhotographersPage() {
  const [tabRaw, setTab] = useUrlState<string>("status", "pending");
  const tab = TABS.find((t) => t.key === tabRaw) ?? TABS[0];
  // In the URL so the search survives opening a profile and pressing Back.
  const [query, setQuery] = useUrlState<string>("q", "", { debounceMs: 300 });
  // The server search follows the URL, which trails typing by the 300 ms debounce.
  const q = (useSearchParams().get("q") ?? "").trim();
  const list = useAdminPhotographersPage(tab.key, q);
  const kpis = useAdminKpis();
  // The suspended KPI counts runners and admins too, so that tab and Incomplete show no count.
  const counts: Partial<Record<AdminUserStatusFilter, number>> = {
    pending: kpis?.pendingVerifications,
    approved: kpis?.approvedPhotographers,
  };
  // A failed background refresh keeps the loaded list; the error shows only with nothing to show.
  const failed = !!list.error && list.items.length === 0;
  const router = useRouter();
  const searchInputRef = useRef<HTMLInputElement | null>(null);

  const { focusedId, setFocusedId } = useQueueKeyboardNav({
    rowIds: list.items.map((u) => u.userId),
    disabled: false,
    onActivate: (id) => {
      const row = list.items.find((u) => u.userId === id);
      if (!row) return;
      setFocusedId(id);
      router.push(photographerHref(row));
    },
  });
  useSearchFocusShortcut({ active: true, inputRef: searchInputRef });

  return (
    <>
      <Header />

      <div className="mt-8 md:mt-10">
        <label htmlFor="photographer-search" className="sr-only">
          Search photographers
        </label>
        <div className="relative">
          <input
            id="photographer-search"
            ref={searchInputRef}
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search by name, brand, handle, or email"
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

      <AdminStatusTabs
        label="Photographer status"
        tabs={TABS.map((t) => ({ key: t.key, label: t.label, count: counts[t.key] }))}
        value={tab.key}
        onChange={setTab}
      />

      {failed ? (
        <div role="alert" className="mt-10 space-y-3">
          <p className="font-sans text-sm text-error">
            Couldn&apos;t load photographers. Please try again.
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
          {q ? "No photographers in this tab match this search. Try the other tabs." : tab.empty}
        </p>
      ) : (
        <>
          <ul className="mt-6 space-y-4">
            {list.items.map((row) => (
              <li key={row.userId}>
                <AdminPhotographerCard row={row} focused={focusedId === row.userId} />
              </li>
            ))}
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
    </>
  );
}

function Header() {
  return (
    <header className="pb-8 md:pb-12 border-b border-line">
      <Kicker as="p">Photographers</Kicker>
      <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-[1.05] text-ink mt-3">
        Photographers.
      </h1>
      <p className="font-sans text-sm md:text-base text-ink-soft mt-3 max-w-xl">
        Every photographer, by status.
      </p>
    </header>
  );
}
