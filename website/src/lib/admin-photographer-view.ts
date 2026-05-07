"use client";

import { useMemo } from "react";
import {
  ADMIN_USER_SEED,
  type AdminUserRow,
} from "@/lib/admin-user-registry";
import {
  PHOTOGRAPHER_REGISTRY,
  getPhotographerByHandle,
  type PhotographerProfile,
} from "@/lib/photographer-registry";
import { useAdminUserStore, type DecisionLogEntry } from "@/store/admin-user-store";
import { useAuthStore } from "@/store/auth-store";
import {
  usePhotographerSettingsStore,
  type PhotographerSettings,
} from "@/store/photographer-settings-store";

// Merged read of everything needed to render `/admin/photographers/[handle]`.
// Reconciles three sources:
//   1. AdminUserRow — full directory shape (status, suspension, snapshot).
//   2. PhotographerProfile — rich public-profile data (only 2 entries today).
//   3. PhotographerSettings — live state (only when handle matches session).
// Plus the filtered admin decision log for the activity slab.

export interface AdminPhotographerView {
  row: AdminUserRow;
  profile: PhotographerProfile | null;
  liveSettings: PhotographerSettings | null;
  decisions: DecisionLogEntry[];
}

function findRowByHandle(
  handle: string,
  effective: ReadonlyArray<AdminUserRow>,
): AdminUserRow | null {
  const normalized = handle.trim().toLowerCase();
  return effective.find((r) => r.handle === normalized) ?? null;
}

// Non-reactive: useful for one-shot reads (server components, useEffect, etc).
// React components should prefer the hook below so override mutations re-render.
export function getAdminPhotographerView(
  handle: string,
): AdminPhotographerView | null {
  const overrides = useAdminUserStore.getState().overrides;
  const log = useAdminUserStore.getState().log;
  const effective = ADMIN_USER_SEED.map((row) => {
    const patch = overrides[row.userId];
    return patch ? { ...row, ...patch } : row;
  });
  const row = findRowByHandle(handle, effective);
  if (!row) return null;
  const profile = getPhotographerByHandle(handle);
  const sessionUser = useAuthStore.getState().user;
  const isSelf = !!sessionUser && sessionUser.id === row.userId;
  const liveSettings = isSelf
    ? usePhotographerSettingsStore.getState()
    : null;
  const decisions = log.filter((e) => e.userId === row.userId);
  return {
    row,
    profile,
    liveSettings: liveSettings,
    decisions,
  };
}

// Reactive subscriber. Subscribes to the stable underlying state (never
// `getXxx()` selectors) and derives the merged view via useMemo so React 19's
// useSyncExternalStore Object.is comparison stays stable.
export function useAdminPhotographerView(
  handle: string,
): AdminPhotographerView | null {
  const overrides = useAdminUserStore((s) => s.overrides);
  const log = useAdminUserStore((s) => s.log);
  const sessionUser = useAuthStore((s) => s.user);
  // Subscribe to the WHOLE settings slice — when own status flips via admin
  // approve/reject, this hook re-renders to reflect the new completeness.
  const liveSettings = usePhotographerSettingsStore();

  return useMemo<AdminPhotographerView | null>(() => {
    const effective = ADMIN_USER_SEED.map((row) => {
      const patch = overrides[row.userId];
      return patch ? { ...row, ...patch } : row;
    });
    const row = findRowByHandle(handle, effective);
    if (!row) return null;
    const profile = getPhotographerByHandle(handle);
    const isSelf = !!sessionUser && sessionUser.id === row.userId;
    const decisions = log.filter((e) => e.userId === row.userId);
    return {
      row,
      profile,
      liveSettings: isSelf ? liveSettings : null,
      decisions,
    };
  }, [handle, overrides, log, sessionUser, liveSettings]);
}

// Synthetic gradient for photographers who don't have a PhotographerProfile
// (i.e. no entry in PHOTOGRAPHER_REGISTRY). Hash-derived from userId so each
// photographer gets a stable, unique-ish cover without us hand-crafting one.
export function syntheticCoverGradient(userId: string): { from: string; to: string } {
  let h = 2166136261;
  for (let i = 0; i < userId.length; i++) {
    h ^= userId.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  const hue = (h >>> 0) % 360;
  const from = `hsl(${hue}, 38%, 38%)`;
  const to = `hsl(${(hue + 30) % 360}, 32%, 14%)`;
  return { from, to };
}

// Re-export for convenience so detail page imports stay tight.
export { PHOTOGRAPHER_REGISTRY };
