"use client";

import { ChevronLeft, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

interface PaginationProps {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
  className?: string;
  /** sr-only label prefix used to disambiguate when multiple paginators
   *  share the same page (e.g. several admin slabs on one route). */
  ariaLabel?: string;
}

const ICON_BTN_CLS =
  "size-8 rounded-full border border-line text-slate hover:text-ink hover:border-ink transition-colors flex items-center justify-center disabled:opacity-30 disabled:hover:text-slate disabled:hover:border-line disabled:cursor-not-allowed";

const PAGE_BTN_BASE =
  "min-w-[36px] rounded-full px-3 py-1 font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] tnum transition-colors";
const PAGE_BTN_INACTIVE = "text-slate hover:text-ink";
const PAGE_BTN_ACTIVE = "text-ink border border-ink";

const ELLIPSIS_CLS =
  "px-1 font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft";

export function Pagination({
  currentPage,
  totalPages,
  onPageChange,
  className,
  ariaLabel = "Pagination",
}: PaginationProps) {
  if (totalPages <= 1) return null;

  const pages = getPageNumbers(currentPage, totalPages);

  return (
    <nav
      aria-label={ariaLabel}
      className={cn(
        "flex items-center justify-end gap-1.5 pt-4 mt-4 border-t border-line",
        className,
      )}
    >
      <button
        type="button"
        onClick={() => onPageChange(currentPage - 1)}
        disabled={currentPage <= 1}
        aria-label="Previous page"
        className={ICON_BTN_CLS}
      >
        <ChevronLeft className="size-4" aria-hidden="true" />
      </button>

      {pages.map((page, i) =>
        page === "ellipsis" ? (
          <span
            key={`ellipsis-${i}`}
            aria-hidden="true"
            className={ELLIPSIS_CLS}
          >
            …
          </span>
        ) : (
          <button
            key={page}
            type="button"
            onClick={() => onPageChange(page)}
            aria-label={`Page ${page}`}
            aria-current={currentPage === page ? "page" : undefined}
            className={cn(
              PAGE_BTN_BASE,
              currentPage === page ? PAGE_BTN_ACTIVE : PAGE_BTN_INACTIVE,
            )}
          >
            {page}
          </button>
        ),
      )}

      <button
        type="button"
        onClick={() => onPageChange(currentPage + 1)}
        disabled={currentPage >= totalPages}
        aria-label="Next page"
        className={ICON_BTN_CLS}
      >
        <ChevronRight className="size-4" aria-hidden="true" />
      </button>
    </nav>
  );
}

function getPageNumbers(
  current: number,
  total: number,
): (number | "ellipsis")[] {
  if (total <= 7) return Array.from({ length: total }, (_, i) => i + 1);

  if (current <= 3) return [1, 2, 3, 4, "ellipsis", total];
  if (current >= total - 2)
    return [1, "ellipsis", total - 3, total - 2, total - 1, total];
  return [1, "ellipsis", current - 1, current, current + 1, "ellipsis", total];
}
