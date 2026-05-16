"use client";

import { useEffect } from "react";
import { create } from "zustand";
import {
  fetchAdminPhotographerSettings,
  type AdminPhotographerSettingsResponse,
} from "@/lib/api-admin";

// F-NEW-1 — lazy per-userId cache of admin-side photographer settings.
// Each entry is fetched once per session (60s stale time) and shared
// across every component that calls the hook for the same userId — so
// opening the verifications drawer and the photographer detail page on
// the same row only hits the BE once.
//
// Why a global cache instead of useState in the hook: the drawer and the
// /admin/photographers/[handle] page can mount simultaneously; without
// dedup we'd fire two parallel fetches with identical responses.

const STALE_MS = 60_000;

interface Entry {
  data: AdminPhotographerSettingsResponse | null;
  loading: boolean;
  error: string | null;
  fetchedAt: number;
  inFlight: Promise<void> | null;
}

interface AdminPhotographerSettingsState {
  byId: Record<string, Entry>;
  fetchById: (userId: string) => Promise<void>;
  invalidate: (userId: string) => void;
}

const useAdminPhotographerSettingsStore =
  create<AdminPhotographerSettingsState>((set, get) => ({
    byId: {},
    fetchById: async (userId) => {
      const existing = get().byId[userId];
      if (existing?.inFlight) return existing.inFlight;
      const p = (async () => {
        set((s) => ({
          byId: {
            ...s.byId,
            [userId]: {
              data: existing?.data ?? null,
              loading: true,
              error: null,
              fetchedAt: existing?.fetchedAt ?? 0,
              inFlight: null,
            },
          },
        }));
        try {
          const data = await fetchAdminPhotographerSettings(userId);
          set((s) => ({
            byId: {
              ...s.byId,
              [userId]: {
                data,
                loading: false,
                error: null,
                fetchedAt: Date.now(),
                inFlight: null,
              },
            },
          }));
        } catch (err) {
          set((s) => ({
            byId: {
              ...s.byId,
              [userId]: {
                data: existing?.data ?? null,
                loading: false,
                error:
                  err instanceof Error
                    ? err.message
                    : "Failed to load photographer settings",
                fetchedAt: Date.now(),
                inFlight: null,
              },
            },
          }));
        }
      })();
      set((s) => ({
        byId: {
          ...s.byId,
          [userId]: {
            ...(s.byId[userId] ?? {
              data: null,
              loading: true,
              error: null,
              fetchedAt: 0,
            }),
            inFlight: p,
          },
        },
      }));
      return p;
    },
    invalidate: (userId) =>
      set((s) => {
        if (!s.byId[userId]) return s;
        return {
          byId: {
            ...s.byId,
            [userId]: { ...s.byId[userId], fetchedAt: 0 },
          },
        };
      }),
  }));

export interface UseAdminPhotographerSettingsResult {
  data: AdminPhotographerSettingsResponse | null;
  loading: boolean;
  error: string | null;
}

// F-NEW-1a — force-refetch a single photographer's settings. Called from
// useAdminUserStore admin actions (approve/reject/suspend/forceEdit/etc.)
// so any open drawer or detail page sees fresh BE state without waiting
// out the 60s TTL. Skips when the userId isn't in the cache — no point
// pre-populating a row no one is viewing.
export function refetchAdminPhotographerSettings(userId: string): void {
  const state = useAdminPhotographerSettingsStore.getState();
  if (!state.byId[userId]) return;
  useAdminPhotographerSettingsStore.setState((s) => {
    const existing = s.byId[userId];
    if (!existing) return s;
    return {
      byId: {
        ...s.byId,
        [userId]: {
          ...existing,
          fetchedAt: 0,
          inFlight: null,
        },
      },
    };
  });
  void state.fetchById(userId);
}

// Pass `null` to disable the fetch (when the caller already has data from
// another source — typically the live photographer-settings store for the
// session photographer's own row).
export function useAdminPhotographerSettings(
  userId: string | null,
): UseAdminPhotographerSettingsResult {
  const entry = useAdminPhotographerSettingsStore((s) =>
    userId ? s.byId[userId] : null,
  );
  const fetchById = useAdminPhotographerSettingsStore((s) => s.fetchById);

  useEffect(() => {
    if (!userId) return;
    const current = useAdminPhotographerSettingsStore.getState().byId[userId];
    if (!current || Date.now() - current.fetchedAt > STALE_MS) {
      void fetchById(userId);
    }
  }, [userId, fetchById]);

  return {
    data: entry?.data ?? null,
    loading: entry?.loading ?? !!userId,
    error: entry?.error ?? null,
  };
}
