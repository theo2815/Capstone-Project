"use client";

import { useEffect, useMemo } from "react";
import { create } from "zustand";
import { fetchAdminUsers } from "@/lib/api-admin";
import { type AdminUserRow } from "@/lib/admin-user-registry";
import { useAdminUserStore } from "@/store/admin-user-store";

// Phase 2 — admin user reads now come from GET /api/v1/admin/users instead
// of the mock seed + submissions. This module owns the cache + dedup + the
// merge with optimistic mutations (Phase 3 will drop the merge once
// approve/reject consume responses directly).
//
// Stale time: 30s. Admin UI doesn't need real-time data, but we want
// reasonably fresh data when an admin navigates between surfaces. Phase 3
// mutations call `invalidate()` to force a refetch on the next read.

const STALE_MS = 30_000;
const PAGE_LIMIT = 200;

interface AdminUsersServerState {
  rows: AdminUserRow[];
  loading: boolean;
  error: string | null;
  fetchedAt: number;
  inFlight: Promise<void> | null;
  refetch: () => Promise<void>;
  invalidate: () => void;
  /** Replace or insert a row by userId. Called from admin mutation actions
   *  in `useAdminUserStore` after the BE confirms the change — the BE
   *  response carries the canonical row, so we don't need a full refetch. */
  upsertRow: (row: AdminUserRow) => void;
  /** Wipe the cache. Called from `resetUserScopedStores()` on auth
   *  transitions so the next admin login refetches a fresh user list
   *  instead of inheriting the previous admin's cached rows. */
  clear: () => void;
}

export const useAdminUsersServerStore = create<AdminUsersServerState>(
  (set, get) => ({
    rows: [],
    loading: false,
    error: null,
    fetchedAt: 0,
    inFlight: null,
    refetch: async () => {
      // Dedupe concurrent calls — if a fetch is already in flight, every
      // caller awaits the same promise instead of firing a second request.
      const existing = get().inFlight;
      if (existing) return existing;
      const p = (async () => {
        set({ loading: true, error: null });
        try {
          const rows = await fetchAdminUsers({ limit: PAGE_LIMIT });
          set({
            rows,
            loading: false,
            fetchedAt: Date.now(),
            inFlight: null,
          });
        } catch (err) {
          set({
            loading: false,
            error:
              err instanceof Error
                ? err.message
                : "Failed to load admin users",
            inFlight: null,
          });
        }
      })();
      set({ inFlight: p });
      return p;
    },
    invalidate: () => set({ fetchedAt: 0 }),
    upsertRow: (row) =>
      set((s) => {
        const idx = s.rows.findIndex((r) => r.userId === row.userId);
        if (idx === -1) return { rows: [...s.rows, row] };
        const next = s.rows.slice();
        next[idx] = row;
        return { rows: next };
      }),
    clear: () =>
      set({
        rows: [],
        loading: false,
        error: null,
        fetchedAt: 0,
        inFlight: null,
      }),
  }),
);

export interface UseAdminUsersDataResult {
  rows: AdminUserRow[];
  loading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

// Single source of truth for admin user reads. Merges:
//   1. Server rows from GET /admin/users — canonical state.
//   2. Client-side submissions (Phase 1 mirror — still needed in the demo
//      so a photographer's just-submitted row shows up instantly in the
//      same browser before the 30s revalidation hits).
//
// Phase 3 removed the `overrides` merge: admin mutations now await the BE
// response and write the canonical row straight into the cache via
// `upsertRow`, so there's no optimistic-overlay layer to maintain.
// Submissions are filtered against the server set to avoid double-rendering
// once the BE catches up.
export function useAdminUsersData(): UseAdminUsersDataResult {
  const serverRows = useAdminUsersServerStore((s) => s.rows);
  const loading = useAdminUsersServerStore((s) => s.loading);
  const error = useAdminUsersServerStore((s) => s.error);
  const fetchedAt = useAdminUsersServerStore((s) => s.fetchedAt);
  const refetch = useAdminUsersServerStore((s) => s.refetch);
  const submissions = useAdminUserStore((s) => s.submissions);

  useEffect(() => {
    if (fetchedAt === 0 || Date.now() - fetchedAt > STALE_MS) {
      void refetch();
    }
  }, [fetchedAt, refetch]);

  // Phase 5 — focus-refetch (admin parity with photographer's Phase 4 sync).
  // When the admin tab regains focus after being in the background (e.g.
  // they switched to /dashboard/settings as their own photographer account
  // and submitted something), pull the queue again so the new row appears
  // without waiting out the 30s stale window.
  useEffect(() => {
    function onVisible() {
      if (!document.hidden) void refetch();
    }
    document.addEventListener("visibilitychange", onVisible);
    window.addEventListener("focus", onVisible);
    return () => {
      document.removeEventListener("visibilitychange", onVisible);
      window.removeEventListener("focus", onVisible);
    };
  }, [refetch]);

  const rows = useMemo(() => {
    const seen = new Set(serverRows.map((r) => r.userId));
    const submissionRows = Object.values(submissions).filter(
      (r) => !seen.has(r.userId),
    );
    return submissionRows.length === 0
      ? serverRows
      : [...serverRows, ...submissionRows];
  }, [serverRows, submissions]);

  return { rows, loading, error, refetch };
}

// Non-reactive read — for one-shot lookups outside React contexts (event
// handlers in stores, server-side helpers, etc). Same merge semantics as
// the hook but reads current snapshot synchronously.
export function getAdminUsersDataSnapshot(): AdminUserRow[] {
  const { rows: serverRows } = useAdminUsersServerStore.getState();
  const { submissions } = useAdminUserStore.getState();
  const seen = new Set(serverRows.map((r) => r.userId));
  const submissionRows = Object.values(submissions).filter(
    (r) => !seen.has(r.userId),
  );
  return submissionRows.length === 0
    ? serverRows
    : [...serverRows, ...submissionRows];
}
