"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";

type UseUrlStateOptions<T extends string> = {
  debounceMs?: number;
  parse?: (raw: string) => T;
  serialize?: (value: T) => string | null;
};

function defaultParse<T extends string>(raw: string): T {
  return raw as T;
}

function defaultSerialize<T extends string>(value: T): string | null {
  const str = String(value ?? "");
  return str === "" ? null : str;
}

// `router.replace` is a transition — it does not commit synchronously, so
// `useSearchParams()` keeps handing back the pre-navigation snapshot for a
// while afterwards. Two writers inside that window each compose the next URL
// from the same stale snapshot, and whoever writes last silently drops the
// other's param. On /events that is picking a city inside the search box's
// 300ms debounce: the debounced write flushes against params that predate the
// city and the city disappears.
//
// So every writer shares the query string it last *asked* for. While a write
// is in flight that string, not the snapshot, is the base for the next one.
// It is dropped the moment the snapshot catches up — or the pathname changes —
// so an external navigation (a <Link>, the back button) is never composed on
// top of a stale base.
let pendingUrl: { pathname: string; search: string } | null = null;

function composeUrl(
  pathname: string,
  snapshot: URLSearchParams,
  mutate: (params: URLSearchParams) => void,
): string {
  const params =
    pendingUrl && pendingUrl.pathname === pathname
      ? new URLSearchParams(pendingUrl.search)
      : new URLSearchParams(Array.from(snapshot.entries()));

  mutate(params);

  const search = params.toString();
  pendingUrl = { pathname, search };
  return search ? `${pathname}?${search}` : pathname;
}

/**
 * Retire the in-flight base as soon as the URL moves at all.
 *
 * Unconditional on purpose. This effect only re-runs when the pathname or the
 * query string actually changed, and any change means the snapshot is now at
 * least as fresh as what we were holding — whether that change is our own
 * write committing, a <Link>, or the back button. Clearing only on an exact
 * match would strand the base after a back button (snapshot moves to a value
 * we never wrote, so it would never match) and the next write would compose
 * on top of the state the user just backed out of.
 */
function useReconcilePendingUrl(pathname: string, search: string) {
  useEffect(() => {
    pendingUrl = null;
  }, [pathname, search]);
}

export function useUrlState<T extends string>(
  name: string,
  defaultValue: T,
  options: UseUrlStateOptions<T> = {},
): [T, (next: T) => void] {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const parse = options.parse ?? (defaultParse as (raw: string) => T);
  const serialize = options.serialize ?? (defaultSerialize as (v: T) => string | null);
  const debounceMs = options.debounceMs ?? 0;

  const urlValue = (() => {
    const raw = searchParams.get(name);
    if (raw === null) return defaultValue;
    return parse(raw);
  })();

  const [localValue, setLocalValue] = useState<T>(urlValue);
  const lastUrlValueRef = useRef<T>(urlValue);

  useEffect(() => {
    if (urlValue !== lastUrlValueRef.current) {
      lastUrlValueRef.current = urlValue;
      setLocalValue(urlValue);
    }
  }, [urlValue]);

  // Read through a ref rather than the render closure. `searchParams` changes
  // identity on every navigation, so listing it in writeToUrl's deps made
  // writeToUrl unstable — which turned the unmount-flush effect below into
  // something that re-ran on every URL change, flushing the pending value
  // against a params snapshot that predated whatever a sibling hook had just
  // written. On /events that meant typing in the search box and picking a city
  // inside the 300ms debounce silently reverted the city.
  const searchParamsRef = useRef(searchParams);
  useEffect(() => {
    searchParamsRef.current = searchParams;
  }, [searchParams]);

  useReconcilePendingUrl(pathname, searchParams.toString());

  const writeToUrl = useCallback(
    (value: T) => {
      const url = composeUrl(pathname, searchParamsRef.current, (params) => {
        const serialized = serialize(value);
        if (serialized === null || serialized === defaultValue) {
          params.delete(name);
        } else {
          params.set(name, serialized);
        }
      });
      router.replace(url, { scroll: false });
    },
    [name, defaultValue, pathname, router, serialize],
  );

  const debounceTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingValueRef = useRef<T | null>(null);

  // Flush a pending debounced write on unmount so a value typed and then
  // navigated away from still lands. Both refs are cleared after the flush —
  // leaving them set meant a later cleanup saw a stale timer id plus a stale
  // pending value and wrote it to the URL again, resurrecting a search term
  // the user had already cleared.
  useEffect(() => {
    return () => {
      if (!debounceTimer.current) return;
      clearTimeout(debounceTimer.current);
      debounceTimer.current = null;
      if (pendingValueRef.current !== null) {
        writeToUrl(pendingValueRef.current);
        pendingValueRef.current = null;
      }
    };
  }, [writeToUrl]);

  const setValue = useCallback(
    (next: T) => {
      setLocalValue(next);
      lastUrlValueRef.current = next;

      if (debounceMs > 0) {
        pendingValueRef.current = next;
        if (debounceTimer.current) clearTimeout(debounceTimer.current);
        debounceTimer.current = setTimeout(() => {
          if (pendingValueRef.current !== null) {
            writeToUrl(pendingValueRef.current);
            pendingValueRef.current = null;
          }
          debounceTimer.current = null;
        }, debounceMs);
        return;
      }

      writeToUrl(next);
    },
    [debounceMs, writeToUrl],
  );

  return [localValue, setValue];
}

type BatchPatch = Record<string, string | null>;

export function useUrlStateBatch(): (patch: BatchPatch) => void {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  useReconcilePendingUrl(pathname, searchParams.toString());

  return useCallback(
    (patch: BatchPatch) => {
      const url = composeUrl(pathname, searchParams, (params) => {
        for (const [key, value] of Object.entries(patch)) {
          if (value === null || value === "") {
            params.delete(key);
          } else {
            params.set(key, value);
          }
        }
      });
      router.replace(url, { scroll: false });
    },
    [pathname, router, searchParams],
  );
}
