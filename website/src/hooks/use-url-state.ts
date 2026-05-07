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

  const writeToUrl = useCallback(
    (value: T) => {
      const params = new URLSearchParams(Array.from(searchParams.entries()));
      const serialized = serialize(value);
      if (serialized === null || serialized === defaultValue) {
        params.delete(name);
      } else {
        params.set(name, serialized);
      }
      const qs = params.toString();
      router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
    },
    [name, defaultValue, pathname, router, searchParams, serialize],
  );

  const debounceTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingValueRef = useRef<T | null>(null);

  useEffect(() => {
    return () => {
      if (debounceTimer.current) {
        clearTimeout(debounceTimer.current);
        if (pendingValueRef.current !== null) {
          writeToUrl(pendingValueRef.current);
        }
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
