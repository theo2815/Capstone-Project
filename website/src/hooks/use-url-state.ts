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

  const writeToUrl = useCallback(
    (value: T) => {
      const params = new URLSearchParams(
        Array.from(searchParamsRef.current.entries()),
      );
      const serialized = serialize(value);
      if (serialized === null || serialized === defaultValue) {
        params.delete(name);
      } else {
        params.set(name, serialized);
      }
      const qs = params.toString();
      router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
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

  return useCallback(
    (patch: BatchPatch) => {
      const params = new URLSearchParams(Array.from(searchParams.entries()));
      for (const [key, value] of Object.entries(patch)) {
        if (value === null || value === "") {
          params.delete(key);
        } else {
          params.set(key, value);
        }
      }
      const qs = params.toString();
      router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
    },
    [pathname, router, searchParams],
  );
}
