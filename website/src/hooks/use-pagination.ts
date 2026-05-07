"use client";

import { useEffect, useMemo, useState } from "react";

export interface UsePaginationResult<T> {
  pageItems: T[];
  currentPage: number;
  totalPages: number;
  pageSize: number;
  setPage: (page: number) => void;
  resetPage: () => void;
}

// Slice an array into a 1-indexed paginated view. When `items` shrinks
// below the current page (e.g. a filter narrows the list), the page
// auto-resets to 1 — otherwise the user can be stranded on "page 5 of 3"
// looking at an empty slice. `pageItems` is memoized so consumers feeding
// it back into useMemo / useSyncExternalStore stay stable.
export function usePagination<T>(
  items: ReadonlyArray<T>,
  pageSize: number = 10,
): UsePaginationResult<T> {
  const [currentPage, setCurrentPage] = useState(1);

  const totalPages = Math.max(1, Math.ceil(items.length / pageSize));

  useEffect(() => {
    if (currentPage > totalPages) setCurrentPage(1);
  }, [currentPage, totalPages]);

  const safePage = Math.min(currentPage, totalPages);

  const pageItems = useMemo(() => {
    const start = (safePage - 1) * pageSize;
    return items.slice(start, start + pageSize);
  }, [items, safePage, pageSize]);

  return {
    pageItems,
    currentPage: safePage,
    totalPages,
    pageSize,
    setPage: setCurrentPage,
    resetPage: () => setCurrentPage(1),
  };
}
