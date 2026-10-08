"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { Slab } from "@/components/profile-shell";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import {
  BTN_DANGER,
  BTN_PRIMARY,
  BTN_SECONDARY,
  BTN_SIZE,
} from "@/components/ui/button-styles";
import { PAGE_SIZE } from "@/lib/pagination-config";
import { safeHttpUrl } from "@/lib/utils";
import { AdminRejectModal } from "@/components/admin/admin-reject-modal";
import { AdminDetailDrawer } from "@/components/admin/admin-detail-drawer";
import { CollapsibleReviewSection } from "@/components/admin/admin-collapse";
import { AvatarDisc } from "@/components/account/avatar-disc";
import { useAdminUserActions, useAdminUserDetail } from "@/hooks/use-admin-users";
import { AdminDecisionsTimeline } from "./admin-decisions-timeline";
import { useAdminUsers, EMPTY_ADMIN_USERS } from "@/hooks/use-admin-users";
import { useAuthStore } from "@/store/auth-store";
import { useToast } from "@/hooks/use-toast";
import { useUrlState } from "@/hooks/use-url-state";
import {
  useQueueKeyboardNav,
  useDrawerVerbs,
} from "@/hooks/use-admin-keyboard";
import {
  PAYOUT_METHOD_LABEL,
  SOCIAL_PLATFORM_LABEL,
  SOCIAL_PLATFORM_TILE,
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
import { useRegions } from "@/hooks/use-regions";

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
  const users = useAdminUsers();
  const effective = users.data ?? EMPTY_ADMIN_USERS;
  const { isPending: loading, refetch } = users;
  const error = users.error?.message;
  const { approve, reject, verifyBatch, isPending: actionPending, pendingAction } = useAdminUserActions();
  const { showToast } = useToast();

  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [bulkRejectOpen, setBulkRejectOpen] = useState(false);
  const [drawerRejectOpen, setDrawerRejectOpen] = useState(false);
  const [rejectRevisions, setRejectRevisions] = useState<Record<string, number>>({});
  const [rowId, setRowId] = useUrlState<string>("row", "");
  const detail = useAdminUserDetail(rowId || null);
  const submitting = actionPending([...selected, ...(rowId ? [rowId] : [])]);
  const activeLabel = pendingAction(rowId || Array.from(selected)[0] || "");

  const pending = useMemo(
    () =>
      effective.filter(
        (u) =>
          u.role === "PHOTOGRAPHER" && u.verificationStatus === "pending" && u.suspendedAt === null,
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
    const candidate = detail.isSuccess ? detail.data : byId.get(rowId);
    if (!candidate) return null;
    if (
      candidate.role !== "PHOTOGRAPHER" ||
      candidate.verificationStatus !== "pending" || candidate.suspendedAt !== null
    )
      return null;
    return candidate;
  }, [rowId, byId, detail.data, detail.isSuccess]);

  useEffect(() => {
    if (rowId && detail.isSuccess && !detail.isFetching && !openRow && !submitting) setRowId("");
  }, [rowId, openRow, setRowId, detail.isSuccess, detail.isFetching, submitting]);

  // Drop ids that have left the pending pool (e.g. just approved /
  // rejected — the row no longer renders a checkbox).
  const pendingIds = useMemo(
    () => new Set(pending.map((p) => p.userId)),
    [pending],
  );
  useEffect(() => {
    if (submitting) return;
    setSelected((prev) => {
      let mutated = false;
      const next = new Set<string>();
      for (const id of prev) {
        if (pendingIds.has(id)) next.add(id);
        else mutated = true;
      }
      return mutated ? next : prev;
    });
  }, [pendingIds, submitting]);

  // ESC clears selection when no modal/drawer is open.
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key !== "Escape") return;
      if (submitting || bulkRejectOpen || drawerRejectOpen || openRow) return;
      if (selected.size === 0) return;
      setSelected(new Set());
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [selected, bulkRejectOpen, drawerRejectOpen, openRow, submitting]);

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
  const incompleteVisible = useMemo(
    () => incomplete.slice(0, incompleteLoaded),
    [incomplete, incompleteLoaded],
  );
  const queueNavDisabled =
    submitting || openRow !== null || bulkRejectOpen || drawerRejectOpen;
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
    if (submitting) return;
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function clearSelection() {
    if (submitting) return;
    setSelected(new Set());
  }

  function handleRowActivate(id: string) {
    if (submitting) return;
    setFocusedId(id);
    if (selectionActive) toggleSelect(id);
    else setRowId(id);
  }

  function handleOpenDrawer(id: string) {
    if (submitting) return;
    setFocusedId(id);
    setRowId(id);
  }

  // Bulk handlers await explicit outcomes in parallel so
  // one BE failure doesn't block the rest. The summary toast tallies
  // successes vs failures so admin sees an honest count — individual
  // failures already surfaced their own error toast from the action hook.
  async function handleBulkApprove() {
    await handleBulkDecision();
  }

  async function handleBulkRejectSubmit(reason: string) {
    await handleBulkDecision(reason);
  }

  function openRejection(bulk: boolean) {
    const rows = bulk ? Array.from(selected).map((id) => byId.get(id)!) : openRow ? [openRow] : [];
    setRejectRevisions(Object.fromEntries(rows.map((row) => [row.userId, row.verificationRevision])));
    if (bulk) setBulkRejectOpen(true);
    else setDrawerRejectOpen(true);
  }

  async function handleBulkDecision(reason?: string) {
    const sessionVersion = useAuthStore.getState().sessionVersion;
    const ids = Array.from(selected).filter((id) => pendingIds.has(id));
    if (!ids.length) return;
    const results = await verifyBatch(ids.map((id) => ({ userId: id, verificationRevision: reason === undefined ? byId.get(id)!.verificationRevision : rejectRevisions[id] })), reason === undefined ? "approve" : "reject", reason);
    if (!results) return;
    if (useAuthStore.getState().sessionVersion !== sessionVersion) return;
    const failed = ids.filter((_, i) => !results[i]);
    setSelected(new Set(failed));
    const count = ids.length - failed.length;
    if (!failed.length) setBulkRejectOpen(false);
    if (count) showToast({
      kind: failed.length ? "error" : reason === undefined ? "success" : "info",
      message: failed.length ? `${count}/${ids.length} updated — ${failed.length} failed` : `${reason === undefined ? "Approved" : "Sent back"} · ${count} ${count === 1 ? "photographer" : "photographers"}`,
    });
  }

  async function handleDrawerApprove() {
    if (!openRow) return;
    const row = openRow;
      if (await approve(row.userId, row.verificationRevision)) {
        showToast({ kind: "success", message: `Approved · ${row.brandName ?? row.name}` });
        setRowId("");
      }
  }

  async function handleDrawerRejectSubmit(reason: string) {
    if (!openRow) return;
    const row = openRow;
      if (await reject(row.userId, reason, rejectRevisions[row.userId])) {
        showToast({ kind: "info", message: `Sent back · ${row.brandName ?? row.name}` });
        setDrawerRejectOpen(false); setRowId("");
      }
  }

  // Drawer verbs: E approves the open photographer, R opens the reject
  // reason modal. Disabled while the modal is on top so a stray E/R
  // doesn't punch through.
  useDrawerVerbs({
    active: openRow !== null && !drawerRejectOpen && !submitting,
    verbs: openRow
      ? {
          E: handleDrawerApprove,
          R: () => openRejection(false),
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
      {rowId && detail.isPending && !openRow && <p role="status">Loading verification…</p>}
      {rowId && detail.isError && <p role="alert">Couldn&apos;t load verification. <button onClick={() => void detail.refetch()}>Try again</button></p>}
      {(showFirstLoad || showError) && (
        <div className="rounded-2xl border border-line bg-bone-deep/40 px-5 py-4 font-mono uppercase tracking-[0.18em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft">
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
                  disabled={submitting}
                  aria-busy={submitting}
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
                    <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft mt-1">
                      {row.handle ? `@${row.handle}` : "No handle yet"}
                      <span className="text-slate-soft"> · </span>
                      <CompletenessFraction snapshot={row.settingsSnapshot} />
                    </p>
                  </div>
                  <span className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft shrink-0">
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

      <Slab id="recent-decisions" number="03" title="Decision history">
        <AdminDecisionsTimeline />
      </Slab>

      <VerificationsBulkBar
        submitting={submitting}
        count={selected.size}
        onApprove={handleBulkApprove}
        onReject={() => openRejection(true)}
        onClear={clearSelection}
      />

      <AdminRejectModal

        submitting={submitting}
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
          onClose={() => { if (!submitting) setRowId(""); }}
          escDisabled={drawerRejectOpen || submitting}
          kicker="Verification"
          title={openRow.brandName ?? openRow.name}
          rightHeader={
            <span className="inline-flex items-center font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] rounded-full border border-line text-slate px-3 py-0.5">
              Pending
            </span>
          }
          subtitle={
            <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate">
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
                disabled={submitting}
                  aria-busy={submitting}
                onClick={() => openRejection(false)}
                className={`${BTN_DANGER} ${BTN_SIZE.sm}`}
              >
                Reject…
              </button>
              <button
                type="button"
                disabled={submitting}
                  aria-busy={submitting}
                onClick={handleDrawerApprove}
                className={`${BTN_PRIMARY} ${BTN_SIZE.sm}`}
              >
                {activeLabel === "Approving…" ? activeLabel : "Approve"}
              </button>
            </>
          }
        >
          <VerificationDetailBody row={openRow} />
        </AdminDetailDrawer>
      )}

      {openRow && (
        <AdminRejectModal
          submitting={submitting}
          open={drawerRejectOpen}
          onClose={() => setDrawerRejectOpen(false)}
          onSubmit={handleDrawerRejectSubmit}
          photographerName={openRow.brandName ?? openRow.name}
        />
      )}
    </>
  );
}

function PendingRow({
  row,
  disabled,
  selected,
  selectionActive,
  focused,
  onToggleSelect,
  onActivate,
  onOpenDrawer,
}: {
  row: AdminUserRow;
  disabled: boolean;
  selected: boolean;
  selectionActive: boolean;
  focused: boolean;
  onToggleSelect: () => void;
  onActivate: () => void;
  onOpenDrawer: () => void;
}) {
  const completeness = computeCompleteness(row.settingsSnapshot);

  function handleArticleKeyDown(e: React.KeyboardEvent<HTMLElement>) {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onActivate();
    }
  }

  return (
    <article
      role="button"
      aria-disabled={disabled}
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
          disabled={disabled}
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
                <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate">
                  @{row.handle ?? "—"}
                </p>
              </div>
              <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft mt-2">
                {row.region ?? row.city}
                <span className="text-slate-soft"> · </span>
                <span className="tnum">
                  Joined {formatJoinDate(row.createdAt)}
                </span>
              </p>

              <div className="mt-5 flex items-center gap-3 flex-wrap">
                <CompletenessStrip snapshot={row.settingsSnapshot} />
                <span className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft tnum">
                  {completeness.filled}/{completeness.total} fields
                </span>
                {row.settingsSnapshot && (
                  <span className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft">
                    ·{" "}
                    <span className="tnum">{row.settingsSnapshot.socialCount}</span>{" "}
                    social ·{" "}
                    <span className="tnum">{row.settingsSnapshot.payoutCount}</span>{" "}
                    payout
                  </span>
                )}
              </div>
            </div>

            <button
              type="button"
              disabled={disabled}
              onClick={(e) => {
                e.stopPropagation();
                onOpenDrawer();
              }}
              aria-label={`Open review for ${row.brandName ?? row.name}`}
              className="disabled:opacity-50 disabled:cursor-not-allowed shrink-0 size-8 rounded-full text-slate hover:text-ink hover:bg-bone-deep transition-colors flex items-center justify-center"
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
  const settings = useEffectivePhotographerSettings(row.userId);
  const effective = settings.data;
  const { regions } = useRegions();

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
    : row.settingsSnapshot;

  const completeness = computeCompleteness(liveSnapshot);
  const resolvedRegion = effective?.region
    ? formatRegionLabel(
        regions,
        effective.region.regionCode,
        effective.region.provinceCode,
      )
    : null;
  const regionLine = resolvedRegion ?? row.region ?? "Not set";
  const publicHandle = effective?.handle?.trim() || row.handle || "";

  const socialsCount = effective?.socials.length ?? 0;
  const payoutsCount = effective?.payouts.length ?? 0;
  const avatarUrl = effective?.avatarUrl ?? row.avatarUrl ?? null;

  if (settings.isPending) return <p role="status">Loading verification details…</p>;
  if (settings.isError) return <p role="alert">Couldn&apos;t load verification details. <button onClick={() => void settings.refetch()}>Try again</button></p>;

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4 pb-2">
        <AvatarDisc
          name={row.brandName ?? row.name}
          size="md"
          tone="ink"
          avatarOverride={avatarUrl ? { dataUrl: avatarUrl } : null}
        />
        <div className="min-w-0">
          <p className="font-display text-lg md:text-xl text-ink truncate">
            {row.brandName ?? row.name}
          </p>
          <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft truncate">
            {row.email}
          </p>
        </div>
      </div>

      <CollapsibleReviewSection kicker="Identity">
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
      </CollapsibleReviewSection>

      <CollapsibleReviewSection kicker="Public profile cover">
        <CoverSection effective={effective} />
      </CollapsibleReviewSection>

      <CollapsibleReviewSection kicker="Public URL">
        <PublicUrlSection handle={publicHandle} />
      </CollapsibleReviewSection>

      <CollapsibleReviewSection kicker="Watermark">
        <WatermarkSection
          effective={effective}
          brand={row.brandName ?? row.name}
        />
      </CollapsibleReviewSection>

      <CollapsibleReviewSection
        kicker={
          <>
            Social &amp; verification links ·{" "}
            <span className="tnum">{socialsCount}</span>{" "}
            {socialsCount === 1 ? "link" : "links"}
          </>
        }
      >
        <SocialsSection effective={effective} />
      </CollapsibleReviewSection>

      <CollapsibleReviewSection
        kicker={
          <>
            Payout accounts ·{" "}
            <span className="tnum">{payoutsCount}</span>{" "}
            {payoutsCount === 1 ? "account" : "accounts"}
          </>
        }
      >
        <PayoutsSection effective={effective} />
      </CollapsibleReviewSection>

      <CollapsibleReviewSection kicker="Settings completeness">
        <div className="rounded-2xl border border-line bg-bone-deep p-5 md:p-6 space-y-4">
          <div className="flex items-center justify-between gap-4 flex-wrap">
            <CompletenessStrip snapshot={liveSnapshot} />
            <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate tnum">
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
                    className={`font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] ${
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
                  <span className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate tnum">
                    {liveSnapshot.socialCount} linked
                  </span>
                </li>
                <li className="flex items-center justify-between gap-4 font-sans text-sm">
                  <span className="text-ink">Payout methods</span>
                  <span className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate tnum">
                    {liveSnapshot.payoutCount} on file
                  </span>
                </li>
              </>
            )}
          </ul>
        </div>
      </CollapsibleReviewSection>

      <CollapsibleReviewSection kicker="Account">
        <dl className="grid grid-cols-1 sm:grid-cols-2 gap-y-4 gap-x-8">
          <FieldRow
            label="Joined"
            value={formatJoinDate(row.createdAt)}
            mono
          />
          <FieldRow label="Status" value="Pending review" mono />
        </dl>
      </CollapsibleReviewSection>
    </div>
  );
}

function CoverSection({
  effective,
}: {
  effective: EffectivePhotographerSettings | null;
}) {
  const cover = effective?.cover ?? null;
  if (!cover) {
    return (
      <p className="font-sans text-sm text-slate-soft">
        No cover banner uploaded.
      </p>
    );
  }
  return (
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
  );
}

function PublicUrlSection({ handle }: { handle: string }) {
  if (!handle) {
    return (
      <p className="font-sans text-sm text-slate-soft">No handle set yet.</p>
    );
  }
  return (
    <div className="flex items-center justify-between gap-4 flex-wrap rounded-2xl border border-line bg-bone-deep px-4 py-3">
      <p className="font-mono text-[13px] min-[400px]:text-[14px] md:text-[13px] text-ink tnum truncate">
        quickpitik.com/{handle}
      </p>
      <Link
        href={`/${handle}`}
        target="_blank"
        rel="noopener"
        className="shrink-0 font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate hover:text-ink transition-colors"
      >
        Open ↗
      </Link>
    </div>
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
  if (!watermark) {
    return (
      <p className="font-sans text-sm text-slate-soft">No watermark uploaded.</p>
    );
  }
  return (
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
  );
}

function SocialsSection({
  effective,
}: {
  effective: EffectivePhotographerSettings | null;
}) {
  const socials = effective?.socials ?? [];
  if (socials.length === 0) {
    return (
      <p className="font-sans text-sm text-slate-soft">
        No social profiles linked yet.
      </p>
    );
  }
  return (
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
          {safeHttpUrl(social.url) && (
            <a
              href={safeHttpUrl(social.url)!}
              target="_blank"
              rel="noopener noreferrer"
              className="shrink-0 font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate hover:text-ink transition-colors"
            >
              Open ↗
            </a>
          )}
        </li>
      ))}
    </ul>
  );
}

function PayoutsSection({
  effective,
}: {
  effective: EffectivePhotographerSettings | null;
}) {
  const payouts = effective?.payouts ?? [];
  if (payouts.length === 0) {
    return (
      <p className="font-sans text-sm text-slate-soft">
        No payout accounts on file.
      </p>
    );
  }
  return (
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
                <span className="ml-2 font-mono uppercase tracking-[0.18em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-fresh tnum">
                  Primary
                </span>
              )}
            </p>
            <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft mt-1 tnum">
              {formatPayoutNumber(p.method, p.accountNumber)}
            </p>
          </div>
          <p className="font-sans text-sm text-slate-soft truncate">
            {p.accountName}
          </p>
        </li>
      ))}
    </ul>
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
      <dt className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft">
        {label}
      </dt>
      <dd
        className={`mt-1 ${
          mono
            ? "font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-ink tnum"
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
  submitting,
  onApprove,
  onReject,
  onClear,
}: {
  count: number;
  submitting: boolean;
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
        <p className="kicker">
          <span className="tnum">{count}</span> selected
        </p>
        <div className="flex items-center gap-2 flex-wrap">
          <button
            type="button"
            disabled={submitting}
                  aria-busy={submitting}
            onClick={onApprove}
            className={`${BTN_PRIMARY} ${BTN_SIZE.sm}`}
          >
            Approve {count}
          </button>
          <button
            type="button"
            disabled={submitting}
                  aria-busy={submitting}
            onClick={onReject}
            className={`${BTN_SECONDARY} ${BTN_SIZE.sm}`}
          >
            Reject {count}…
          </button>
          <button
            type="button"
            disabled={submitting}
                  aria-busy={submitting}
            onClick={onClear}
            className="disabled:opacity-50 disabled:cursor-not-allowed font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate hover:text-ink transition-colors px-3 py-2"
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
