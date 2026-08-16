"use client";

import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";
import { useRouter } from "next/navigation";
import { Modal } from "@/components/ui/modal";
import { useAdminPaletteStore } from "@/store/admin-palette-store";
import { useAdminDisputeStore } from "@/store/admin-dispute-store";
import { useAdminFlagStore } from "@/store/admin-flag-store";
import { useAdminPayoutStore } from "@/store/admin-payout-store";
import { useAdminUsersData } from "@/lib/admin-users-data";
import {
  buildPaletteIndex,
  fuzzyScore,
  PALETTE_KIND_LABEL,
  type PaletteEntry,
} from "@/lib/admin-palette-index";
import { ADMIN_FLAGS_ENABLED } from "@/lib/constants";
import { cn } from "@/lib/utils";

// Phase 5 admin ⌘K command palette. Reads live snapshots from the four
// admin stores so in-flight overrides flow into the index immediately
// (e.g. an approved photographer disappears from the verifications group
// without a refresh). Pure UI; navigation is router.push to whatever
// route the entry advertises — for queue rows that's
// /admin/inbox?type=<queue>&row=<id> so the queue's drawer auto-opens.
//
// Keyboard:
//   - Cmd/Ctrl+K toggles open (handled in useCommandPaletteShortcut).
//   - Esc closes (handled by the Modal primitive).
//   - Up/Down cycles the visible result list (no wrap).
//   - Enter activates the focused row.
//   - All key handling is bound to the input so focus stays on the
//     search field — list rows are visually highlighted via aria-current
//     instead of receiving real focus.

const MAX_RESULTS = 30;

export function CommandPalette() {
  const open = useAdminPaletteStore((s) => s.open);
  const setOpen = useAdminPaletteStore((s) => s.setOpen);
  const recents = useAdminPaletteStore((s) => s.recents);
  const pushRecent = useAdminPaletteStore((s) => s.pushRecent);

  const { rows: userPool } = useAdminUsersData();
  const disputeOverrides = useAdminDisputeStore((s) => s.overrides);
  const flagOverrides = useAdminFlagStore((s) => s.overrides);
  const payoutOverrides = useAdminPayoutStore((s) => s.overrides);

  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLUListElement>(null);

  const [query, setQuery] = useState("");
  const [activeIdx, setActiveIdx] = useState(0);

  const index = useMemo(
    () =>
      buildPaletteIndex({
        userPool,
        disputeOverrides,
        flagOverrides,
        payoutOverrides,
      }),
    [userPool, disputeOverrides, flagOverrides, payoutOverrides],
  );

  const indexById = useMemo(() => {
    const map = new Map<string, PaletteEntry>();
    for (const e of index) map.set(e.id, e);
    return map;
  }, [index]);

  const trimmed = query.trim();
  const results = useMemo(() => {
    if (trimmed.length === 0) {
      // Empty query → recents (resolved against current index so stale
      // ids that no longer match a live entry are dropped).
      const seen = new Set<string>();
      const out: PaletteEntry[] = [];
      for (const id of recents) {
        if (seen.has(id)) continue;
        seen.add(id);
        const entry = indexById.get(id);
        if (entry) out.push(entry);
      }
      return out;
    }
    type Scored = { entry: PaletteEntry; score: number };
    const scored: Scored[] = [];
    for (const entry of index) {
      const score = fuzzyScore(trimmed, entry);
      if (score === null) continue;
      scored.push({ entry, score });
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, MAX_RESULTS).map((x) => x.entry);
  }, [trimmed, index, indexById, recents]);

  // Reset on every open. New open == fresh palette.
  useEffect(() => {
    if (open) {
      setQuery("");
      setActiveIdx(0);
      // Defer focus to next tick so the Modal portal has mounted.
      const id = window.setTimeout(() => inputRef.current?.focus(), 0);
      return () => window.clearTimeout(id);
    }
  }, [open]);

  // Keep activeIdx within bounds when the result list shrinks.
  useEffect(() => {
    if (activeIdx > 0 && activeIdx >= results.length) {
      setActiveIdx(Math.max(0, results.length - 1));
    }
  }, [results.length, activeIdx]);

  // Scroll active row into view on every change.
  useEffect(() => {
    if (!open) return;
    const list = listRef.current;
    if (!list) return;
    const child = list.children.item(activeIdx) as HTMLElement | null;
    child?.scrollIntoView({ block: "nearest" });
  }, [activeIdx, open]);

  function activate(entry: PaletteEntry) {
    pushRecent(entry.id);
    setOpen(false);
    router.push(entry.route);
  }

  function handleInputKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === "ArrowDown") {
      if (results.length === 0) return;
      e.preventDefault();
      setActiveIdx((i) => Math.min(i + 1, results.length - 1));
      return;
    }
    if (e.key === "ArrowUp") {
      if (results.length === 0) return;
      e.preventDefault();
      setActiveIdx((i) => Math.max(i - 1, 0));
      return;
    }
    if (e.key === "Enter") {
      if (results.length === 0) return;
      e.preventDefault();
      const entry = results[activeIdx];
      if (entry) activate(entry);
      return;
    }
    if (e.key === "Home") {
      e.preventDefault();
      setActiveIdx(0);
      return;
    }
    if (e.key === "End") {
      e.preventDefault();
      setActiveIdx(Math.max(0, results.length - 1));
    }
  }

  return (
    <Modal
      isOpen={open}
      onClose={() => setOpen(false)}
      className="max-w-xl p-0"
    >
      <div className="border-b border-line p-4 md:p-5">
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setActiveIdx(0);
          }}
          onKeyDown={handleInputKeyDown}
          // The fuzzy scorer runs over every page, queue and photographer on
          // each keystroke; a pasted wall of text makes that scale badly for
          // a query no result can ever match.
          maxLength={200}
          placeholder="Search admin · pages, queues, photographers"
          aria-label="Search admin"
          aria-controls="palette-results"
          aria-activedescendant={
            results[activeIdx] ? `palette-row-${results[activeIdx].id}` : undefined
          }
          className="w-full bg-transparent font-sans text-base text-ink placeholder:text-slate-soft focus:outline-none"
        />
      </div>
      <div className="max-h-[60vh] overflow-y-auto">
        {trimmed.length === 0 && results.length === 0 ? (
          <EmptyState />
        ) : results.length === 0 ? (
          <NoMatches query={trimmed} />
        ) : (
          <>
            {trimmed.length === 0 && (
              <p className="font-mono uppercase tracking-[0.25em] text-[10px] text-slate-soft px-5 pt-4 pb-2">
                Recent
              </p>
            )}
            <ul
              ref={listRef}
              id="palette-results"
              role="listbox"
              aria-label="Palette results"
              className="px-2 py-2 space-y-1"
            >
              {results.map((entry, i) => (
                <PaletteRow
                  key={entry.id}
                  entry={entry}
                  active={i === activeIdx}
                  onHover={() => setActiveIdx(i)}
                  onClick={() => activate(entry)}
                />
              ))}
            </ul>
          </>
        )}
      </div>
      <div className="border-t border-line px-5 py-3 flex items-center justify-between gap-3 flex-wrap">
        <p className="font-mono uppercase tracking-[0.25em] text-[10px] text-slate-soft">
          {results.length === 0
            ? trimmed.length === 0
              ? "Type to search"
              : "0 matches"
            : `${results.length} ${results.length === 1 ? "match" : "matches"}`}
        </p>
        <div className="flex items-center gap-3 flex-wrap">
          <Hint label="Move" keys={["↑", "↓"]} />
          <Hint label="Open" keys={["Enter"]} />
          <Hint label="Close" keys={["Esc"]} />
        </div>
      </div>
    </Modal>
  );
}

function PaletteRow({
  entry,
  active,
  onHover,
  onClick,
}: {
  entry: PaletteEntry;
  active: boolean;
  onHover: () => void;
  onClick: () => void;
}) {
  return (
    <li
      id={`palette-row-${entry.id}`}
      role="option"
      aria-selected={active}
      onMouseEnter={onHover}
      onClick={onClick}
      className={cn(
        "rounded-xl px-3 py-2.5 cursor-pointer transition-colors flex items-center gap-3",
        active ? "bg-bone-deep" : "hover:bg-bone-deep/60",
      )}
    >
      <span
        className={cn(
          "shrink-0 inline-flex items-center font-mono uppercase tracking-[0.25em] text-[10px] rounded-full border px-2 py-0.5",
          active ? "border-ink text-ink" : "border-line text-slate",
        )}
      >
        {PALETTE_KIND_LABEL[entry.kind]}
      </span>
      <div className="min-w-0 flex-1">
        <p className="font-sans text-sm text-ink truncate">{entry.title}</p>
        {entry.subtitle && (
          <p className="font-mono uppercase tracking-[0.25em] text-[10px] text-slate-soft mt-1 truncate">
            {entry.subtitle}
          </p>
        )}
      </div>
      <span
        aria-hidden="true"
        className={cn(
          "shrink-0 text-base leading-none",
          active ? "text-ink" : "text-slate-soft",
        )}
      >
        ↵
      </span>
    </li>
  );
}

function EmptyState() {
  return (
    <div className="px-5 py-8">
      <p className="font-sans text-sm text-ink-soft">
        {ADMIN_FLAGS_ENABLED
          ? "Search anything in admin — pages, pending verifications, disputes, flags, payouts, photographers."
          : "Search anything in admin — pages, pending verifications, disputes, payouts, photographers."}
      </p>
      <p className="font-mono uppercase tracking-[0.25em] text-[10px] text-slate-soft mt-3">
        Picks you make appear here as recents on next open.
      </p>
    </div>
  );
}

function NoMatches({ query }: { query: string }) {
  return (
    <div className="px-5 py-8">
      <p className="font-sans text-sm text-ink-soft">
        No matches for{" "}
        <span className="font-mono text-ink">&ldquo;{query}&rdquo;</span>.
      </p>
      <p className="font-mono uppercase tracking-[0.25em] text-[10px] text-slate-soft mt-3">
        Try a brand name, queue id, runner handle, or page name.
      </p>
    </div>
  );
}

function Hint({ label, keys }: { label: string; keys: string[] }) {
  return (
    <div className="flex items-center gap-1.5">
      {keys.map((k, i) => (
        <kbd
          key={i}
          className="font-mono text-[11px] text-ink rounded-md border border-line bg-bone-deep px-1.5 py-0.5"
        >
          {k}
        </kbd>
      ))}
      <span className="font-mono uppercase tracking-[0.25em] text-[10px] text-slate-soft">
        {label}
      </span>
    </div>
  );
}
