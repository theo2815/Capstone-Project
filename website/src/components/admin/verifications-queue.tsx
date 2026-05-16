"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { Slab } from "@/components/profile-shell";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { PAGE_SIZE } from "@/lib/pagination-config";
import { AdminRejectModal } from "@/components/admin/admin-reject-modal";
import { AdminDetailDrawer } from "@/components/admin/admin-detail-drawer";
import { useAdminUserStore } from "@/store/admin-user-store";
import { useAdminUsersData } from "@/lib/admin-users-data";
import {
  notifyVerificationApproved,
  notifyVerificationRejected,
} from "@/lib/admin-photographer-notifications";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/hooks/use-auth";
import { useUrlState } from "@/hooks/use-url-state";
import {
  useQueueKeyboardNav,
  useDrawerVerbs,
} from "@/hooks/use-admin-keyboard";
import {
  PAYOUT_METHOD_LABEL,
  SOCIAL_PLATFORM_LABEL,
  SOCIAL_PLATFORM_TILE,
  usePhotographerSettingsStore,
} from "@/store/photographer-settings-store";
import {
  type AdminUserRow,
  type PhotographerSettingsSnapshot,
} from "@/lib/admin-user-registry";
import {
  useEffectivePhotographerSettings,
  type EffectivePhotographerSettings,
} from "@/lib/admin-photographer-view";
import { formatPayoutNumber } from "@/lib/payout-format";
import { formatRegionLabel } from "@/lib/ph-regions";

// Verifications queue body — extracted from /admin/verifications/page.tsx so
// /admin/inbox can render the same three slabs (Open queue · Incomplete ·
// Recent decisions) under its own KPI-stripped header.
//
// Phase 2 added bulk-select on the Open queue: tick checkboxes to stage
// approve/reject across multiple pending photographers. Selection is
// local Set<string>, never URL or store. ESC clears the selection when
// no modal is open. The floating bar mirrors the payouts bar pattern.
//
// Phase 3 adds the focus-mode drawer. Click a pending row → drawer with
// the full review surface (photographer header, completeness panel,
// Approve/Reject in the footer). With at least one row selected, the row
// click toggles the checkbox instead of opening the drawer; the chevron
// on the right always opens. URL state via ?row=<userId> so refresh and
// browser-back behave; the drawer mirrors the chip route pattern.
//
// Demo loop: when the admin Approves the row matching the currently
// logged-in user, useAdminUserStore.approve() also flips
// usePhotographerSettingsStore.verificationStatus to "approved" — so
// toggling back to PHOTOGRAPHER role shows the updated dashboard banner
// without a refresh.

const SNAPSHOT_FIELDS: ReadonlyArray<{
  key: keyof PhotographerSettingsSnapshot;
  label: string;
  short: string;
}> = [
  { key: "hasCover", label: "Cover banner", short: "Cover" },
  { key: "hasBrandName", label: "Brand name", short: "Brand" },
  { key: "hasWatermark", label: "Watermark", short: "WM" },
  { key: "hasHandle", label: "Public handle", short: "Handle" },
  { key: "hasRegion", label: "Region", short: "Region" },
];

export function VerificationsQueue() {
  // Server data + optimistic overrides/submissions merged in a single hook.
  // The decision log + mutation actions still live in useAdminUserStore.
  const { rows: effective, loading, error, refetch } = useAdminUsersData();
  const log = useAdminUserStore((s) => s.log);
  const approve = useAdminUserStore((s) => s.approve);
  const reject = useAdminUserStore((s) => s.reject);
  const { showToast } = useToast();

  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [bulkRejectOpen, setBulkRejectOpen] = useState(false);
  const [drawerRejectOpen, setDrawerRejectOpen] = useState(false);
  const [rowId, setRowId] = useUrlState<string>("row", "");

  const pending = useMemo(
    () =>
      effective.filter(
        (u) =>
          u.role === "PHOTOGRAPHER" && u.verificationStatus === "pending",
      ),
    [effective],
  );

  const incomplete = useMemo(
    () =>
      effective.filter(
        (u) =>
          u.role === "PHOTOGRAPHER" && u.verificationStatus === "incomplete",
      ),
    [effective],
  );

  // Lookup map for resolving log entries by userId without re-iterating.
  const byId = useMemo(() => {
    const map = new Map<string, AdminUserRow>();
    for (const row of effective) map.set(row.userId, row);
    return map;
  }, [effective]);

  // The row backing the open drawer. If the URL points at an id that has
  // left the pending pool (just approved/rejected, or stale link), we
  // close the drawer by clearing the URL state.
  const openRow = useMemo<AdminUserRow | null>(() => {
    if (!rowId) return null;
    const candidate = byId.get(rowId);
    if (!candidate) return null;
    if (
      candidate.role !== "PHOTOGRAPHER" ||
      candidate.verificationStatus !== "pending"
    )
      return null;
    return candidate;
  }, [rowId, byId]);

  useEffect(() => {
    if (rowId && !openRow) setRowId("");
  }, [rowId, openRow, setRowId]);

  // Drop ids that have left the pending pool (e.g. just approved /
  // rejected — the row no longer renders a checkbox).
  const pendingIds = useMemo(
    () => new Set(pending.map((p) => p.userId)),
    [pending],
  );
  useEffect(() => {
    setSelected((prev) => {
      let mutated = false;
      const next = new Set<string>();
      for (const id of prev) {
        if (pendingIds.has(id)) next.add(id);
        else mutated = true;
      }
      return mutated ? next : prev;
    });
  }, [pendingIds]);

  // ESC clears selection when no modal/drawer is open.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key !== "Escape") return;
      if (bulkRejectOpen || drawerRejectOpen || openRow) return;
      if (selected.size === 0) return;
      setSelected(new Set());
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [selected, bulkRejectOpen, drawerRejectOpen, openRow]);

  const selectionActive = selected.size > 0;

  // Keyboard nav covers the Open queue only — the Incomplete and Recent
  // decisions slabs are read-only listings, not openable rows.
  const rowIds = useMemo(() => pending.map((p) => p.userId), [pending]);

  // Pagination — load-more model. Open queue stays full visibility so
  // J/K, the bulk-bar, and the drawer flow all see every pending row at
  // once. Read-only slabs (Incomplete + Recent decisions) start at 10
  // and grow by 10 per click.
  const [incompleteLoaded, setIncompleteLoaded] = useState(
    PAGE_SIZE.ADMIN_INITIAL,
  );
  const [decisionsLoaded, setDecisionsLoaded] = useState(
    PAGE_SIZE.ADMIN_INITIAL,
  );
  const incompleteVisible = useMemo(
    () => incomplete.slice(0, incompleteLoaded),
    [incomplete, incompleteLoaded],
  );
  const decisionsVisible = useMemo(
    () => log.slice(0, decisionsLoaded),
    [log, decisionsLoaded],
  );
  const queueNavDisabled =
    openRow !== null || bulkRejectOpen || drawerRejectOpen;
  const { focusedId, setFocusedId } = useQueueKeyboardNav({
    rowIds,
    disabled: queueNavDisabled,
    onActivate: (id) => handleRowActivate(id),
  });

  // Mirror URL-driven and click-driven row openings into focusedId so
  // J/K resumes from the row the user just looked at.
  useEffect(() => {
    if (rowId) setFocusedId(rowId);
  }, [rowId, setFocusedId]);

  function toggleSelect(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function clearSelection() {
    setSelected(new Set());
  }

  function handleRowActivate(id: string) {
    setFocusedId(id);
    if (selectionActive) toggleSelect(id);
    else setRowId(id);
  }

  function handleOpenDrawer(id: string) {
    setFocusedId(id);
    setRowId(id);
  }

  function handleBulkApprove() {
    const ids = Array.from(selected);
    for (const id of ids) {
      const target = byId.get(id);
      approve(id);
      if (target) {
        notifyVerificationApproved({
          photographerId: id,
          brandName: target.brandName ?? target.name,
        });
      }
    }
    clearSelection();
    showToast({
      kind: "success",
      message: `Approved · ${ids.length} ${
        ids.length === 1 ? "photographer" : "photographers"
      }`,
    });
  }

  function handleBulkRejectSubmit(reason: string) {
    const ids = Array.from(selected);
    for (const id of ids) {
      const target = byId.get(id);
      reject(id, reason);
      if (target) {
        notifyVerificationRejected({
          photographerId: id,
          brandName: target.brandName ?? target.name,
          reason,
        });
      }
    }
    setBulkRejectOpen(false);
    clearSelection();
    showToast({
      kind: "info",
      message: `Sent back · ${ids.length} ${
        ids.length === 1 ? "photographer" : "photographers"
      }`,
    });
  }

  function handleDrawerApprove() {
    if (!openRow) return;
    const brand = openRow.brandName ?? openRow.name;
    approve(openRow.userId);
    notifyVerificationApproved({
      photographerId: openRow.userId,
      brandName: brand,
    });
    showToast({ kind: "success", message: `Approved · ${brand}` });
    setRowId("");
  }

  function handleDrawerRejectSubmit(reason: string) {
    if (!openRow) return;
    const brand = openRow.brandName ?? openRow.name;
    reject(openRow.userId, reason);
    notifyVerificationRejected({
      photographerId: openRow.userId,
      brandName: brand,
      reason,
    });
    setDrawerRejectOpen(false);
    showToast({ kind: "info", message: `Sent back · ${brand}` });
    setRowId("");
  }

  // Drawer verbs: E approves the open photographer, R opens the reject
  // reason modal. Disabled while the modal is on top so a stray E/R
  // doesn't punch through.
  useDrawerVerbs({
    active: openRow !== null && !drawerRejectOpen,
    verbs: openRow
      ? {
          E: handleDrawerApprove,
          R: () => setDrawerRejectOpen(true),
        }
      : {},
  });

  // First-load indicator: only surface the spinner/error when we have no
  // cached rows to show. After the initial fetch lands, the SWR-style cache
  // keeps the queue visible while background revalidations run.
  const showFirstLoad = loading && effective.length === 0;
  const showError = !!error && effective.length === 0;

  return (
    <>
      {(showFirstLoad || showError) && (
        <div className="rounded-2xl border border-line bg-bone-deep/40 px-5 py-4 font-mono uppercase tracking-[0.25em] text-[10px] text-slate-soft">
          {showFirstLoad
            ? "Loading admin queue…"
            : (
              <span className="inline-flex items-center gap-3">
                <span>Couldn&apos;t load queue — {error}</span>
                <button
                  type="button"
                  onClick={() => void refetch()}
                  className="text-ink underline decoration-line underline-offset-4 hover:decoration-ink"
                >
                  Retry
                </button>
              </span>
            )}
        </div>
      )}
      <Slab
        id="open-queue"
        number="01"
        title="Open queue"
        caption="Awaiting review"
        trailing={`${pending.length} pending`}
      >
        {pending.length === 0 ? (
          <QueueEmpty />
        ) : (
          <ul className="space-y-4">
            {pending.map((row) => (
              <li key={row.userId}>
                <PendingRow
                  row={row}
                  selected={selected.has(row.userId)}
                  selectionActive={selectionActive}
                  focused={focusedId === row.userId}
                  onToggleSelect={() => toggleSelect(row.userId)}
                  onActivate={() => handleRowActivate(row.userId)}
                  onOpenDrawer={() => handleOpenDrawer(row.userId)}
                />
              </li>
            ))}
          </ul>
        )}
      </Slab>

      <Slab
        id="incomplete"
        number="02"
        title="Incomplete"
        caption="Settings still missing fields"
        trailing={`${incomplete.length} stuck`}
      >
        {incomplete.length === 0 ? (
          <p className="font-sans text-sm text-slate-soft">
            No photographers stuck on incomplete settings right now.
          </p>
        ) : (
          <>
            <ul className="space-y-3">
              {incompleteVisible.map((row) => (
                <li
                  key={row.userId}
                  className="flex items-center justify-between gap-4 border-b border-line pb-3"
                >
                  <div className="min-w-0">
                    <p className="font-display text-base text-ink truncate">
                      {row.brandName ?? row.name}
                    </p>
                    <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft mt-1">
                      {row.handle ? `@${row.handle}` : "No handle yet"}
                      <span className="text-slate-soft"> · </span>
                      <CompletenessFraction snapshot={row.settingsSnapshot} />
                    </p>
                  </div>
                  <span className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft shrink-0">
                    Self-serve
                  </span>
                </li>
              ))}
            </ul>
            <LoadMoreButton
              shown={incompleteVisible.length}
              total={incomplete.length}
              increment={PAGE_SIZE.ADMIN_INCREMENT}
              onLoadMore={() =>
                setIncompleteLoaded((c) => c + PAGE_SIZE.ADMIN_INCREMENT)
              }
            />
          </>
        )}
      </Slab>

      <Slab
        id="recent-decisions"
        number="03"
        title="Recent decisions"
        caption="Last 30 days"
        trailing={`${log.length} logged`}
      >
        {log.length === 0 ? (
          <p className="font-sans text-sm text-slate-soft">
            No decisions in the last 30 days.
          </p>
        ) : (
          <>
            <ul className="space-y-3">
              {decisionsVisible.map((entry, i) => {
                const target = byId.get(entry.userId);
                return (
                  <li
                    key={`${entry.userId}-${entry.decidedAt}-${i}`}
                    className="flex items-center justify-between gap-4 border-b border-line pb-3"
                  >
                    <div className="min-w-0">
                      <p className="font-display text-base text-ink truncate">
                        {target?.brandName ?? target?.name ?? entry.userId}
                      </p>
                      <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft mt-1 tnum">
                        {formatDecidedAt(entry.decidedAt)}
                        {entry.reason && (
                          <>
                            <span className="text-slate-soft"> · </span>
                            <span className="text-slate">{entry.reason}</span>
                          </>
                        )}
                      </p>
                    </div>
                    <span
                      className={`font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] shrink-0 ${
                        entry.decision === "approved"
                          ? "text-fresh"
                          : "text-slate"
                      }`}
                    >
                      {entry.decision === "approved" ? "Approved" : "Rejected"}
                    </span>
                  </li>
                );
              })}
            </ul>
            <LoadMoreButton
              shown={decisionsVisible.length}
              total={log.length}
              increment={PAGE_SIZE.ADMIN_INCREMENT}
              onLoadMore={() =>
                setDecisionsLoaded((c) => c + PAGE_SIZE.ADMIN_INCREMENT)
              }
            />
          </>
        )}
      </Slab>

      <VerificationsBulkBar
        count={selected.size}
        onApprove={handleBulkApprove}
        onReject={() => setBulkRejectOpen(true)}
        onClear={clearSelection}
      />

      <AdminRejectModal
        open={bulkRejectOpen}
        onClose={() => setBulkRejectOpen(false)}
        onSubmit={handleBulkRejectSubmit}
        photographerName={`${selected.size} ${
          selected.size === 1 ? "photographer" : "photographers"
        }`}
      />

      {openRow && (
        <AdminDetailDrawer
          open={true}
          onClose={() => setRowId("")}
          escDisabled={drawerRejectOpen}
          kicker="Verification"
          title={openRow.brandName ?? openRow.name}
          rightHeader={
            <span className="inline-flex items-center font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] rounded-full border border-line text-slate px-3 py-0.5">
              Pending
            </span>
          }
          subtitle={
            <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate">
              {openRow.handle ? `@${openRow.handle}` : "No handle yet"}
              <span className="text-slate-soft"> · </span>
              {openRow.region ?? openRow.city}
              <span className="text-slate-soft"> · </span>
              <span className="tnum">
                Joined {formatJoinDate(openRow.createdAt)}
              </span>
            </p>
          }
          actions={
            <>
              <button
                type="button"
                onClick={() => setDrawerRejectOpen(true)}
                className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate hover:text-ink transition-colors px-4 py-2"
              >
                Reject…
              </button>
              <button
                type="button"
                onClick={handleDrawerApprove}
                className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-bone bg-ink hover:bg-fresh hover:text-bone transition-colors rounded-full px-5 py-2"
              >
                Approve
              </button>
            </>
          }
        >
          <VerificationDetailBody row={openRow} />
        </AdminDetailDrawer>
      )}

      {openRow && (
        <AdminRejectModal
          open={drawerRejectOpen}
          onClose={() => setDrawerRejectOpen(false)}
          onSubmit={handleDrawerRejectSubmit}
          photographerName={openRow.brandName ?? openRow.name}
        />
      )}
    </>
  );
}

// Pending count is also exported so headers (inbox / verifications) can
// show "N waiting" without re-deriving from the store.
export function usePendingVerificationsCount(): number {
  const { rows } = useAdminUsersData();
  return useMemo(() => {
    let count = 0;
    for (const row of rows) {
      if (
        row.role === "PHOTOGRAPHER" &&
        row.verificationStatus === "pending"
      ) {
        count += 1;
      }
    }
    return count;
  }, [rows]);
}

function PendingRow({
  row,
  selected,
  selectionActive,
  focused,
  onToggleSelect,
  onActivate,
  onOpenDrawer,
}: {
  row: AdminUserRow;
  selected: boolean;
  selectionActive: boolean;
  focused: boolean;
  onToggleSelect: () => void;
  onActivate: () => void;
  onOpenDrawer: () => void;
}) {
  const { user: sessionUser } = useAuth();
  const liveSettings = usePhotographerSettingsStore();

  const isSelf = sessionUser?.id === row.userId;
  // For the seeded user matching the logged-in admin's previous session,
  // pull live settings instead of the snapshot so the demo loop reflects
  // photographer's most recent /dashboard/settings edits.
  const liveSnapshot: PhotographerSettingsSnapshot | null = isSelf
    ? {
        hasCover: !!liveSettings.cover,
        hasBrandName: liveSettings.brandName.trim().length > 0,
        hasWatermark: !!liveSettings.watermark,
        hasHandle: liveSettings.handle.trim().length >= 3,
        hasRegion: !!liveSettings.region,
        socialCount: liveSettings.socials.length,
        payoutCount: liveSettings.payouts.length,
      }
    : row.settingsSnapshot;

  const completeness = computeCompleteness(liveSnapshot);

  function handleArticleKeyDown(e: React.KeyboardEvent<HTMLElement>) {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onActivate();
    }
  }

  return (
    <article
      role="button"
      tabIndex={0}
      data-row-id={row.userId}
      aria-label={`${row.brandName ?? row.name}${
        selectionActive ? " — toggle selection" : " — open review"
      }`}
      onClick={onActivate}
      onKeyDown={handleArticleKeyDown}
      className={`group rounded-2xl border bg-bone p-5 md:p-6 transition-colors cursor-pointer ${
        selected ? "border-ink" : "border-line hover:border-ink"
      } ${focused ? "ring-2 ring-ink ring-offset-2 ring-offset-bone" : ""}`}
    >
      <div className="flex items-start gap-4 flex-wrap md:flex-nowrap">
        <input
          type="checkbox"
          checked={selected}
          onChange={onToggleSelect}
          onClick={(e) => e.stopPropagation()}
          aria-label={`Select ${row.brandName ?? row.name}`}
          className="mt-2 size-4 accent-ink shrink-0 cursor-pointer"
        />
        <div className="min-w-0 flex-1">
          <div className="flex items-start justify-between gap-4 flex-wrap">
            <div className="min-w-0 flex-1">
              <div className="flex items-baseline gap-3 flex-wrap">
                <h3 className="font-display text-xl md:text-2xl font-medium text-ink truncate">
                  {row.brandName ?? row.name}
                </h3>
                <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate">
                  @{row.handle ?? "—"}
                </p>
              </div>
              <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft mt-2">
                {row.region ?? row.city}
                <span className="text-slate-soft"> · </span>
                <span className="tnum">
                  Joined {formatJoinDate(row.createdAt)}
                </span>
              </p>

              <div className="mt-5 flex items-center gap-3 flex-wrap">
                <CompletenessStrip snapshot={liveSnapshot} />
                <span className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft tnum">
                  {completeness.filled}/{completeness.total} fields
                </span>
                {liveSnapshot && (
                  <span className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft">
                    ·{" "}
                    <span className="tnum">{liveSnapshot.socialCount}</span>{" "}
                    social ·{" "}
                    <span className="tnum">{liveSnapshot.payoutCount}</span>{" "}
                    payout
                  </span>
                )}
              </div>
            </div>

            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onOpenDrawer();
              }}
              aria-label={`Open review for ${row.brandName ?? row.name}`}
              className="shrink-0 size-8 rounded-full text-slate hover:text-ink hover:bg-bone-deep transition-colors flex items-center justify-center"
            >
              <span aria-hidden="true" className="text-lg leading-none">
                ›
              </span>
            </button>
          </div>
        </div>
      </div>
    </article>
  );
}

function VerificationDetailBody({ row }: { row: AdminUserRow }) {
  const { user: sessionUser } = useAuth();
  const liveSettings = usePhotographerSettingsStore();
  const effective = useEffectivePhotographerSettings(row.userId);

  const isSelf = sessionUser?.id === row.userId;
  const liveSnapshot: PhotographerSettingsSnapshot | null = effective
    ? {
        hasCover: !!effective.cover,
        hasBrandName: effective.brandName.trim().length > 0,
        hasWatermark: !!effective.watermark,
        hasHandle: effective.handle.trim().length >= 3,
        hasRegion: !!effective.region,
        socialCount: effective.socials.length,
        payoutCount: effective.payouts.length,
      }
    : isSelf
      ? {
          hasCover: !!liveSettings.cover,
          hasBrandName: liveSettings.brandName.trim().length > 0,
          hasWatermark: !!liveSettings.watermark,
          hasHandle: liveSettings.handle.trim().length >= 3,
          hasRegion: !!liveSettings.region,
          socialCount: liveSettings.socials.length,
          payoutCount: liveSettings.payouts.length,
        }
      : row.settingsSnapshot;

  const completeness = computeCompleteness(liveSnapshot);
  const resolvedRegion = effective?.region
    ? formatRegionLabel(effective.region.regionCode, effective.region.provinceCode)
    : null;
  const regionLine = resolvedRegion ?? row.region ?? "Not set";
  const publicHandle = effective?.handle?.trim() || row.handle || "";

  return (
    <div className="space-y-10">
      <section>
        <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft mb-3">
          Identity
        </p>
        <dl className="grid grid-cols-1 sm:grid-cols-2 gap-y-4 gap-x-8">
          <FieldRow label="Email" value={row.email} mono={false} />
          <FieldRow label="Legal name" value={row.name} mono={false} />
          <FieldRow
            label="Brand name"
            value={row.brandName ?? "—"}
            mono={false}
          />
          <FieldRow
            label="Handle"
            value={publicHandle ? `@${publicHandle}` : "—"}
            mono
          />
          <FieldRow label="City" value={row.city} mono={false} />
          <FieldRow label="Region" value={regionLine} mono={false} />
        </dl>
      </section>

      <CoverSection effective={effective} />

      <PublicUrlSection handle={publicHandle} />

      <WatermarkSection effective={effective} brand={row.brandName ?? row.name} />

      <SocialsSection effective={effective} />

      <PayoutsSection effective={effective} />

      <section>
        <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft mb-3">
          Settings completeness
        </p>
        <div className="rounded-2xl border border-line bg-bone-deep p-5 md:p-6 space-y-4">
          <div className="flex items-center justify-between gap-4 flex-wrap">
            <CompletenessStrip snapshot={liveSnapshot} />
            <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate tnum">
              {completeness.filled}/{completeness.total} fields
            </p>
          </div>
          <ul className="space-y-2">
            {SNAPSHOT_FIELDS.map((field) => {
              const filled = !!liveSnapshot?.[field.key];
              return (
                <li
                  key={field.key}
                  className="flex items-center justify-between gap-4 font-sans text-sm"
                >
                  <span className="text-ink">{field.label}</span>
                  <span
                    className={`font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] ${
                      filled ? "text-fresh" : "text-slate-soft"
                    }`}
                  >
                    {filled ? "Filled" : "Missing"}
                  </span>
                </li>
              );
            })}
            {liveSnapshot && (
              <>
                <li className="flex items-center justify-between gap-4 font-sans text-sm pt-2 border-t border-line">
                  <span className="text-ink">Social profiles</span>
                  <span className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate tnum">
                    {liveSnapshot.socialCount} linked
                  </span>
                </li>
                <li className="flex items-center justify-between gap-4 font-sans text-sm">
                  <span className="text-ink">Payout methods</span>
                  <span className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate tnum">
                    {liveSnapshot.payoutCount} on file
                  </span>
                </li>
              </>
            )}
          </ul>
        </div>
      </section>

      <section>
        <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft mb-3">
          Account
        </p>
        <dl className="grid grid-cols-1 sm:grid-cols-2 gap-y-4 gap-x-8">
          <FieldRow
            label="Joined"
            value={formatJoinDate(row.createdAt)}
            mono
          />
          <FieldRow label="Status" value="Pending review" mono />
        </dl>
      </section>
    </div>
  );
}

function CoverSection({
  effective,
}: {
  effective: EffectivePhotographerSettings | null;
}) {
  const cover = effective?.cover ?? null;
  return (
    <section>
      <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft mb-3">
        Public profile cover
      </p>
      {!cover ? (
        <p className="font-sans text-sm text-slate-soft">
          No cover banner uploaded.
        </p>
      ) : (
        <div className="relative bg-bone-deep border border-line rounded-2xl aspect-[16/7] md:aspect-[16/5] overflow-hidden">
          {cover.kind === "image" ? (
            // eslint-disable-next-line @next/next/no-img-element -- data URL preview; backend serves signed S3 URL.
            <img
              src={cover.url}
              alt=""
              className="size-full object-cover"
              draggable={false}
            />
          ) : (
            <div
              aria-hidden
              className="size-full"
              style={{
                background: `linear-gradient(135deg, ${cover.from}, ${cover.to})`,
              }}
            />
          )}
        </div>
      )}
    </section>
  );
}

function PublicUrlSection({ handle }: { handle: string }) {
  return (
    <section>
      <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft mb-3">
        Public URL
      </p>
      {!handle ? (
        <p className="font-sans text-sm text-slate-soft">
          No handle set yet.
        </p>
      ) : (
        <div className="flex items-center justify-between gap-4 flex-wrap rounded-2xl border border-line bg-bone-deep px-4 py-3">
          <p className="font-mono text-[13px] min-[400px]:text-[14px] md:text-[13px] text-ink tnum truncate">
            quickpitik.com/{handle}
          </p>
          <Link
            href={`/${handle}`}
            target="_blank"
            rel="noopener"
            className="shrink-0 font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate hover:text-ink transition-colors"
          >
            Open ↗
          </Link>
        </div>
      )}
    </section>
  );
}

function WatermarkSection({
  effective,
  brand,
}: {
  effective: EffectivePhotographerSettings | null;
  brand: string;
}) {
  const watermark = effective?.watermark ?? null;
  return (
    <section>
      <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft mb-3">
        Watermark
      </p>
      {!watermark ? (
        <p className="font-sans text-sm text-slate-soft">
          No watermark uploaded.
        </p>
      ) : (
        <div className="rounded-2xl border border-line bg-bone-deep p-6 md:p-8 flex items-center justify-center min-h-[120px]">
          {watermark.kind === "image" ? (
            // eslint-disable-next-line @next/next/no-img-element -- data URL preview from photographer settings store; backend serves signed S3 URL.
            <img
              src={watermark.dataUrl}
              alt={`${brand} watermark`}
              className="max-h-24 max-w-full object-contain"
              draggable={false}
            />
          ) : (
            <span
              className="font-mono uppercase tracking-[0.4em] text-2xl md:text-3xl text-ink/70"
              aria-label={`Watermark label: ${watermark.label}`}
            >
              © {watermark.label}
            </span>
          )}
        </div>
      )}
    </section>
  );
}

function SocialsSection({
  effective,
}: {
  effective: EffectivePhotographerSettings | null;
}) {
  const socials = effective?.socials ?? [];
  return (
    <section>
      <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft mb-3">
        Social & verification links · {socials.length}{" "}
        {socials.length === 1 ? "link" : "links"}
      </p>
      {socials.length === 0 ? (
        <p className="font-sans text-sm text-slate-soft">
          No social profiles linked yet.
        </p>
      ) : (
        <ul className="space-y-3">
          {socials.map((social) => (
            <li
              key={social.id}
              className="flex items-center gap-4 border-b border-line pb-3"
            >
              <span
                aria-hidden
                className="shrink-0 size-10 rounded-xl border border-line bg-bone-deep flex items-center justify-center font-mono uppercase tracking-[0.15em] text-[13px] text-ink"
              >
                {SOCIAL_PLATFORM_TILE[social.platform]}
              </span>
              <div className="min-w-0 flex-1">
                <p className="font-display text-base text-ink">
                  {SOCIAL_PLATFORM_LABEL[social.platform]}
                </p>
                <p className="font-mono text-[13px] min-[400px]:text-[14px] md:text-[13px] text-slate-soft mt-0.5 truncate">
                  {social.url}
                </p>
              </div>
              <a
                href={social.url}
                target="_blank"
                rel="noopener noreferrer"
                className="shrink-0 font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate hover:text-ink transition-colors"
              >
                Open ↗
              </a>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function PayoutsSection({
  effective,
}: {
  effective: EffectivePhotographerSettings | null;
}) {
  const payouts = effective?.payouts ?? [];
  return (
    <section>
      <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft mb-3">
        Payout accounts · {payouts.length}{" "}
        {payouts.length === 1 ? "account" : "accounts"}
      </p>
      {payouts.length === 0 ? (
        <p className="font-sans text-sm text-slate-soft">
          No payout accounts on file.
        </p>
      ) : (
        <ul className="space-y-3">
          {payouts.map((p) => (
            <li
              key={p.id}
              className="flex items-center justify-between gap-4 border-b border-line pb-3"
            >
              <div className="min-w-0">
                <p className="font-display text-base text-ink truncate">
                  {PAYOUT_METHOD_LABEL[p.method]}
                  {p.isPrimary && (
                    <span className="ml-2 font-mono uppercase tracking-[0.25em] text-[10px] text-fresh tnum">
                      Primary
                    </span>
                  )}
                </p>
                <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft mt-1 tnum">
                  {formatPayoutNumber(p.method, p.accountNumber)}
                </p>
              </div>
              <p className="font-sans text-sm text-slate-soft truncate">
                {p.accountName}
              </p>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function FieldRow({
  label,
  value,
  mono,
}: {
  label: string;
  value: string;
  mono: boolean;
}) {
  return (
    <div>
      <dt className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft">
        {label}
      </dt>
      <dd
        className={`mt-1 ${
          mono
            ? "font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-ink tnum"
            : "font-sans text-sm text-ink"
        }`}
      >
        {value}
      </dd>
    </div>
  );
}

function VerificationsBulkBar({
  count,
  onApprove,
  onReject,
  onClear,
}: {
  count: number;
  onApprove: () => void;
  onReject: () => void;
  onClear: () => void;
}) {
  if (count === 0) return null;

  return (
    <div
      role="region"
      aria-label="Verifications bulk action bar"
      className="fixed inset-x-4 bottom-4 lg:left-auto lg:right-8 lg:bottom-6 lg:max-w-xl z-40 rounded-2xl border border-ink bg-bone shadow-2xl px-4 md:px-6 py-4 animate-[fade-up_0.2s_ease-out_both]"
    >
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate">
          <span className="tnum">{count}</span> selected
        </p>
        <div className="flex items-center gap-2 flex-wrap">
          <button
            type="button"
            onClick={onApprove}
            className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] bg-fresh text-bone hover:bg-fresh-deep transition-colors rounded-full px-4 py-2"
          >
            Approve {count}
          </button>
          <button
            type="button"
            onClick={onReject}
            className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-ink border border-line hover:bg-ink hover:text-bone hover:border-ink transition-colors rounded-full px-4 py-2"
          >
            Reject {count}…
          </button>
          <button
            type="button"
            onClick={onClear}
            className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate hover:text-ink transition-colors px-3 py-2"
          >
            Clear
          </button>
        </div>
      </div>
    </div>
  );
}

function CompletenessStrip({
  snapshot,
}: {
  snapshot: PhotographerSettingsSnapshot | null;
}) {
  if (!snapshot) {
    return null;
  }
  return (
    <div
      className="flex items-center gap-1.5"
      aria-label="Settings completeness"
    >
      {SNAPSHOT_FIELDS.map((field) => {
        const filled = !!snapshot[field.key];
        return (
          <span
            key={field.key}
            title={`${field.label}${filled ? " ✓" : " — missing"}`}
            className={`size-2 rounded-full ${
              filled ? "bg-fresh" : "bg-line"
            }`}
            aria-hidden="true"
          />
        );
      })}
    </div>
  );
}

function CompletenessFraction({
  snapshot,
}: {
  snapshot: PhotographerSettingsSnapshot | null;
}) {
  const c = computeCompleteness(snapshot);
  return (
    <span className="tnum">
      {c.filled}/{c.total} fields
    </span>
  );
}

function computeCompleteness(
  snapshot: PhotographerSettingsSnapshot | null,
): { filled: number; total: number } {
  if (!snapshot) return { filled: 0, total: SNAPSHOT_FIELDS.length };
  const filled = SNAPSHOT_FIELDS.reduce(
    (acc, field) => acc + (snapshot[field.key] ? 1 : 0),
    0,
  );
  return { filled, total: SNAPSHOT_FIELDS.length };
}

function QueueEmpty() {
  return (
    <div className="rounded-2xl border border-dashed border-line p-10 text-center">
      <p className="font-display text-xl text-ink">Queue is clear.</p>
      <p className="font-sans text-sm text-slate-soft mt-2">
        Nice. Check back when a photographer submits for review.
      </p>
    </div>
  );
}

function formatJoinDate(iso: string): string {
  const d = new Date(iso + "T00:00:00");
  const month = d.toLocaleString("en-US", { month: "short" }).toUpperCase();
  return `${month} ${d.getFullYear()}`;
}

function formatDecidedAt(iso: string): string {
  const d = new Date(iso);
  const month = d.toLocaleString("en-US", { month: "short" }).toUpperCase();
  const day = d.getDate().toString().padStart(2, "0");
  const time = d.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
  });
  return `${month} ${day} · ${time}`;
}
