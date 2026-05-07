"use client";

import { useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Slab } from "@/components/profile-shell";
import { Pagination } from "@/components/ui/pagination";
import { AdminPhotographerCard } from "@/components/admin/admin-photographer-card";
import {
  ADMIN_USER_SEED,
  type AdminUserRow,
} from "@/lib/admin-user-registry";
import { useAdminUserStore } from "@/store/admin-user-store";
import { ROUTES } from "@/lib/constants";
import {
  useQueueKeyboardNav,
  useSearchFocusShortcut,
} from "@/hooks/use-admin-keyboard";
import { usePagination } from "@/hooks/use-pagination";

// Phase 2a admin Photographers directory. Four slabs (Pending / Approved /
// Incomplete / Suspended) of <AdminPhotographerCard>s. Search box filters
// across all slabs. Click any row to land on /admin/photographers/[handle].
//
// Demo loop: admin actions on the detail page write to admin-user-store
// overrides → this directory rebuilds the merged seed and rows migrate
// between slabs without a refresh.

export default function AdminPhotographersPage() {
  // Subscribe to stable underlying state — never `getXxx()` selectors
  // (Phase 1 infinite-loop trap).
  const overrides = useAdminUserStore((s) => s.overrides);
  const [query, setQuery] = useState("");
  const router = useRouter();
  const searchInputRef = useRef<HTMLInputElement | null>(null);

  const effective = useMemo<AdminUserRow[]>(
    () =>
      ADMIN_USER_SEED.map((row) => {
        const patch = overrides[row.userId];
        return patch ? { ...row, ...patch } : row;
      }),
    [overrides],
  );

  const photographers = useMemo(
    () => effective.filter((u) => u.role === "PHOTOGRAPHER"),
    [effective],
  );

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (q.length === 0) return photographers;
    return photographers.filter((u) => {
      return (
        u.name.toLowerCase().includes(q) ||
        (u.brandName ?? "").toLowerCase().includes(q) ||
        (u.handle ?? "").toLowerCase().includes(q) ||
        u.email.toLowerCase().includes(q)
      );
    });
  }, [photographers, query]);

  const sorted = useMemo(
    () =>
      [...filtered].sort((a, b) => b.createdAt.localeCompare(a.createdAt)),
    [filtered],
  );

  const pending = sorted.filter(
    (u) => u.verificationStatus === "pending" && u.suspendedAt === null,
  );
  const approved = sorted.filter(
    (u) => u.verificationStatus === "approved" && u.suspendedAt === null,
  );
  const incomplete = sorted.filter(
    (u) => u.verificationStatus === "incomplete" && u.suspendedAt === null,
  );
  const suspended = sorted.filter((u) => u.suspendedAt !== null);

  const totalAll = photographers.length;
  const pendingAllCount = photographers.filter(
    (u) => u.verificationStatus === "pending" && u.suspendedAt === null,
  ).length;

  // Phase 7 pagination: read-only / non-actionable slabs only. Pending
  // (the head triage slab) keeps full visibility for J/K nav + bulk-bar
  // alignment. The other three are review-only directories and paginate
  // at 10 per page.
  const approvedPagination = usePagination(approved, 10);
  const incompletePagination = usePagination(incomplete, 10);
  const suspendedPagination = usePagination(suspended, 10);

  // Keyboard nav iterates pending + the visible page of each historical
  // slab. Enter on a row with a handle navigates to its detail page;
  // rows without a handle ignore Enter the same way the legacy click
  // handler does.
  const visibleRows = useMemo(
    () => [
      ...pending,
      ...approvedPagination.pageItems,
      ...incompletePagination.pageItems,
      ...suspendedPagination.pageItems,
    ],
    [
      pending,
      approvedPagination.pageItems,
      incompletePagination.pageItems,
      suspendedPagination.pageItems,
    ],
  );
  const rowIds = useMemo(
    () => visibleRows.map((u) => u.userId),
    [visibleRows],
  );
  const handleById = useMemo(() => {
    const map = new Map<string, string | null>();
    for (const u of sorted) map.set(u.userId, u.handle ?? null);
    return map;
  }, [sorted]);
  const { focusedId, setFocusedId } = useQueueKeyboardNav({
    rowIds,
    disabled: false,
    onActivate: (id) => {
      const handle = handleById.get(id);
      if (!handle) return;
      setFocusedId(id);
      router.push(`${ROUTES.ADMIN_PHOTOGRAPHERS}/${handle}`);
    },
  });
  useSearchFocusShortcut({ active: true, inputRef: searchInputRef });

  return (
    <>
      <Header total={totalAll} pending={pendingAllCount} />

      <div className="mt-8 md:mt-10">
        <label htmlFor="photographer-search" className="sr-only">
          Search photographers
        </label>
        <input
          id="photographer-search"
          ref={searchInputRef}
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search by name, brand, handle, or email  ·  /"
          className="w-full rounded-2xl border border-line bg-surface px-5 py-3 font-sans text-sm text-ink placeholder:text-slate-soft focus:outline-none focus:ring-2 focus:ring-fresh focus:border-fresh"
        />
        {query.trim().length > 0 && (
          <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft mt-3 tnum">
            {filtered.length} match{filtered.length === 1 ? "" : "es"}
          </p>
        )}
      </div>

      <DirectorySlab
        id="pending"
        number="01"
        title="Pending"
        caption="Awaiting decision"
        totalCount={pending.length}
        rows={pending}
        empty="Queue is clear."
        focusedId={focusedId}
      />
      <DirectorySlab
        id="approved"
        number="02"
        title="Approved"
        caption="Active on platform"
        totalCount={approved.length}
        rows={approvedPagination.pageItems}
        empty="No approved photographers yet."
        focusedId={focusedId}
        pagination={{
          ariaLabel: "Approved photographers pagination",
          currentPage: approvedPagination.currentPage,
          totalPages: approvedPagination.totalPages,
          onPageChange: approvedPagination.setPage,
        }}
      />
      <DirectorySlab
        id="incomplete"
        number="03"
        title="Incomplete"
        caption="Settings still missing fields"
        totalCount={incomplete.length}
        rows={incompletePagination.pageItems}
        empty="No incomplete profiles."
        focusedId={focusedId}
        pagination={{
          ariaLabel: "Incomplete photographers pagination",
          currentPage: incompletePagination.currentPage,
          totalPages: incompletePagination.totalPages,
          onPageChange: incompletePagination.setPage,
        }}
      />
      <DirectorySlab
        id="suspended"
        number="04"
        title="Suspended"
        caption="Frozen accounts"
        totalCount={suspended.length}
        rows={suspendedPagination.pageItems}
        empty="No active suspensions."
        focusedId={focusedId}
        pagination={{
          ariaLabel: "Suspended photographers pagination",
          currentPage: suspendedPagination.currentPage,
          totalPages: suspendedPagination.totalPages,
          onPageChange: suspendedPagination.setPage,
        }}
      />
    </>
  );
}

function Header({ total, pending }: { total: number; pending: number }) {
  return (
    <header className="pb-8 md:pb-12 border-b border-line">
      <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate">
        Photographers · <span className="tnum">{total}</span> on platform ·{" "}
        <span className="tnum">{pending}</span> pending
      </p>
      <h1 className="font-display text-3xl md:text-4xl font-medium tracking-tight leading-[1.05] text-ink mt-3">
        Photographers.
      </h1>
      <p className="font-sans text-sm md:text-base text-ink-soft mt-3 max-w-xl">
        Every photographer, every status. Click a row for the full profile
        and admin actions.
      </p>
    </header>
  );
}

function DirectorySlab({
  id,
  number,
  title,
  caption,
  rows,
  totalCount,
  empty,
  focusedId,
  pagination,
}: {
  id: string;
  number: string;
  title: string;
  caption: string;
  rows: AdminUserRow[];
  totalCount: number;
  empty: string;
  focusedId: string | null;
  pagination?: {
    ariaLabel: string;
    currentPage: number;
    totalPages: number;
    onPageChange: (page: number) => void;
  };
}) {
  const noun = totalCount === 1 ? "photographer" : "photographers";
  return (
    <Slab
      id={id}
      number={number}
      title={title}
      caption={caption}
      trailing={`${totalCount} ${noun}`}
    >
      {totalCount === 0 ? (
        <p className="font-sans text-sm text-slate-soft">{empty}</p>
      ) : (
        <>
          <ul className="space-y-4">
            {rows.map((row) => (
              <li key={row.userId}>
                <AdminPhotographerCard
                  row={row}
                  focused={focusedId === row.userId}
                />
              </li>
            ))}
          </ul>
          {pagination && (
            <Pagination
              ariaLabel={pagination.ariaLabel}
              currentPage={pagination.currentPage}
              totalPages={pagination.totalPages}
              onPageChange={pagination.onPageChange}
            />
          )}
        </>
      )}
    </Slab>
  );
}
