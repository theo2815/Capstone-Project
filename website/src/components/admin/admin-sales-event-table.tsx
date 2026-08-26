"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { Kicker } from "@/components/ui/kicker";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { ROUTES } from "@/lib/constants";
import { formatLongDate } from "@/lib/format";
import { formatPrice, cn } from "@/lib/utils";
import {
  useSalesByEvent,
  type SalesEventRow,
} from "@/lib/admin-sales";

type SortKey = "gmv" | "name" | "date" | "refunds";
type SortDir = "asc" | "desc";

const PAGE_SIZE = 10;

const SORT_OPTIONS: Array<{ key: SortKey; label: string; defaultDir: SortDir }> = [
  { key: "gmv", label: "GMV", defaultDir: "desc" },
  { key: "name", label: "Name", defaultDir: "asc" },
  { key: "date", label: "Date", defaultDir: "desc" },
  { key: "refunds", label: "Refunds", defaultDir: "desc" },
];

// Sales-by-event audit table. All events with photoCount > 0, sortable by
// implied GMV / name / date / refund $. Hybrid Load-More via the shared
// <LoadMoreButton> primitive (matches the 11 other paginated surfaces).
export function AdminSalesEventTable() {
  const all = useSalesByEvent();
  const [sortKey, setSortKey] = useState<SortKey>("gmv");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [shown, setShown] = useState(PAGE_SIZE);

  const sorted = useMemo(() => {
    const dir = sortDir === "asc" ? 1 : -1;
    return [...all].sort((a, b) => {
      switch (sortKey) {
        case "gmv":
          return dir * (a.impliedGmv - b.impliedGmv);
        case "refunds":
          return dir * (a.refundsIssued - b.refundsIssued);
        case "name":
          return dir * a.name.localeCompare(b.name);
        case "date":
          return dir * a.date.localeCompare(b.date);
        default:
          return 0;
      }
    });
  }, [all, sortKey, sortDir]);

  const total = sorted.length;
  const visible = sorted.slice(0, shown);

  function handleSort(key: SortKey) {
    if (key === sortKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      const next = SORT_OPTIONS.find((o) => o.key === key);
      setSortKey(key);
      setSortDir(next?.defaultDir ?? "desc");
    }
    setShown(PAGE_SIZE);
  }

  if (total === 0) {
    return (
      <p className="font-sans text-sm text-ink-soft">
        No events with photo sales yet — once a race accumulates photos, its
        implied GMV row appears here.
      </p>
    );
  }

  return (
    <div>
      <div className="flex flex-wrap items-center gap-2 mb-5">
        <Kicker tone="soft">Sort by</Kicker>
        {SORT_OPTIONS.map((opt) => {
          const isActive = sortKey === opt.key;
          const arrow = isActive ? (sortDir === "asc" ? "↑" : "↓") : null;
          return (
            <button
              key={opt.key}
              type="button"
              onClick={() => handleSort(opt.key)}
              aria-pressed={isActive}
              className={cn(
                "font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] rounded-full px-3 py-1 border transition-colors",
                isActive
                  ? "border-ink text-ink"
                  : "border-line text-slate hover:text-ink hover:border-ink",
              )}
            >
              {opt.label}
              {arrow && <span className="ml-1.5 tnum">{arrow}</span>}
            </button>
          );
        })}
      </div>

      <ul className="border-y border-line divide-y divide-line">
        {visible.map((row) => (
          <EventTableRow key={row.id} row={row} />
        ))}
      </ul>

      <LoadMoreButton
        shown={Math.min(shown, total)}
        total={total}
        increment={PAGE_SIZE}
        onLoadMore={() => setShown((s) => s + PAGE_SIZE)}
        countLabel={`Showing ${Math.min(shown, total).toLocaleString()} of ${total.toLocaleString()} events`}
        terminalLabel={`All ${total.toLocaleString()} events shown`}
      />
    </div>
  );
}

function EventTableRow({ row }: { row: SalesEventRow }) {
  const href = `${ROUTES.ADMIN_EVENTS}/${row.id}`;
  return (
    <li>
      <Link
        href={href}
        className="group block py-4 md:py-5 transition-colors"
      >
        <div className="md:grid md:grid-cols-[2fr_1fr_1fr_1fr_1fr] md:items-baseline md:gap-6">
          <div className="min-w-0">
            <div className="flex items-baseline gap-3 flex-wrap">
              <p className="font-display text-base text-ink truncate group-hover:text-fresh transition-colors">
                {row.name}
              </p>
              <Kicker tone="soft" className="border border-line rounded-full px-2 py-0.5">
                Implied
              </Kicker>
            </div>
            <Kicker as="p" tone="soft" tnum className="mt-1">
              {row.city} · {capitalize(row.state)} · {row.photoCount.toLocaleString()} photos
            </Kicker>
          </div>

          <DataCell label="Date" value={formatLongDate(row.date, true)} mono />
          <DataCell label="GMV" value={formatPrice(row.impliedGmv)} emphasis />
          <DataCell label="Cut" value={formatPrice(row.impliedCut)} />
          <DataCell
            label="Refunds"
            value={row.refundsIssued > 0 ? `−${formatPrice(row.refundsIssued)}` : "—"}
            tone={row.refundsIssued > 0 ? "warning" : "muted"}
          />
        </div>
      </Link>
    </li>
  );
}

interface DataCellProps {
  label: string;
  value: string;
  mono?: boolean;
  emphasis?: boolean;
  tone?: "default" | "warning" | "muted";
}

function DataCell({
  label,
  value,
  mono = false,
  emphasis = false,
  tone = "default",
}: DataCellProps) {
  const valueClass =
    tone === "warning"
      ? "text-ink-soft"
      : tone === "muted"
        ? "text-slate-soft"
        : "text-ink";
  return (
    <div className="mt-3 md:mt-0 flex items-baseline justify-between md:block md:text-right gap-3">
      <Kicker tone="soft" className="md:hidden">
        {label}
      </Kicker>
      <p
        className={cn(
          "font-mono tnum",
          mono
            ? "uppercase tracking-[0.15em] text-[14px] min-[400px]:text-[15px] md:text-[13px]"
            : emphasis
              ? "font-medium text-base md:text-base"
              : "text-sm md:text-sm",
          valueClass,
        )}
      >
        {value}
      </p>
    </div>
  );
}

function capitalize(s: string): string {
  if (s.length === 0) return s;
  return s[0].toUpperCase() + s.slice(1);
}
