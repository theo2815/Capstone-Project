"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type RefObject,
} from "react";
import { useAdminLegendStore } from "@/store/admin-legend-store";
import { useAdminPaletteStore } from "@/store/admin-palette-store";

// Phase 4 admin keyboard layer.
//
// Four hooks split by surface:
//   - useAdminLegendShortcut() — global `?` toggles the shortcut legend.
//     Mount once inside <AdminShell>.
//   - useQueueKeyboardNav({ rowIds, disabled, onActivate }) — J/K cycle
//     through queue rows, Enter activates the focused id. The queue
//     threads the returned focusedId to its row components and slaps a
//     data-row-id="<id>" attribute on each row root so the hook can scroll
//     the focused row into view via querySelector.
//   - useDrawerVerbs({ active, verbs }) — when a detail drawer is the
//     topmost surface, E/R/H/S map to whichever opener the queue wires in.
//   - useSearchFocusShortcut({ active, inputRef }) — `/` focuses the
//     registered search input (verifications/photographers/flags etc.).
//
// Every listener short-circuits when the user is typing in an input,
// textarea, or contentEditable element (handled by isTypingTarget). The
// queue/verb/search hooks ALSO short-circuit while the legend or the
// command palette is open so keystrokes don't leak through to the surface
// sitting underneath.

export function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
}

export function useAdminLegendShortcut() {
  const toggle = useAdminLegendStore((s) => s.toggle);
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key !== "?") return;
      if (isTypingTarget(e.target)) return;
      e.preventDefault();
      toggle();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle]);
}

interface QueueNavOpts {
  rowIds: string[];
  disabled: boolean;
  onActivate: (id: string) => void;
}

export function useQueueKeyboardNav({
  rowIds,
  disabled,
  onActivate,
}: QueueNavOpts) {
  const legendOpen = useAdminLegendStore((s) => s.open);
  const paletteOpen = useAdminPaletteStore((s) => s.open);
  const effectiveDisabled = disabled || legendOpen || paletteOpen;

  const [focusedId, setFocusedIdState] = useState<string | null>(null);

  // Drop focusedId if it disappears from rowIds (filter changed, row
  // moved off-list because of an action).
  useEffect(() => {
    if (focusedId && !rowIds.includes(focusedId)) {
      setFocusedIdState(null);
    }
  }, [rowIds, focusedId]);

  // Scroll focused row into view on every change. {block:"nearest"} stays
  // calm on small movements and snaps just enough on big jumps.
  useEffect(() => {
    if (!focusedId) return;
    const el = document.querySelector(
      `[data-row-id="${cssEscape(focusedId)}"]`,
    ) as HTMLElement | null;
    if (el) el.scrollIntoView({ block: "nearest" });
  }, [focusedId]);

  // Capture latest onActivate without re-binding the listener on every
  // render — queues commonly hand a freshly-bound handler.
  const onActivateRef = useRef(onActivate);
  useEffect(() => {
    onActivateRef.current = onActivate;
  }, [onActivate]);

  useEffect(() => {
    if (effectiveDisabled) return;
    function onKey(e: KeyboardEvent) {
      if (isTypingTarget(e.target)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const key = e.key;
      const isJ = key === "j" || key === "J";
      const isK = key === "k" || key === "K";
      const isEnter = key === "Enter";
      if (!isJ && !isK && !isEnter) return;

      const idx = focusedId ? rowIds.indexOf(focusedId) : -1;

      if (isJ) {
        if (rowIds.length === 0) return;
        e.preventDefault();
        const next = idx < 0 ? 0 : Math.min(idx + 1, rowIds.length - 1);
        setFocusedIdState(rowIds[next] ?? null);
        return;
      }
      if (isK) {
        if (rowIds.length === 0) return;
        e.preventDefault();
        const next = idx < 0 ? rowIds.length - 1 : Math.max(idx - 1, 0);
        setFocusedIdState(rowIds[next] ?? null);
        return;
      }
      if (isEnter) {
        if (!focusedId) return;
        e.preventDefault();
        onActivateRef.current(focusedId);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [effectiveDisabled, rowIds, focusedId]);

  const setFocusedId = useCallback((id: string | null) => {
    setFocusedIdState(id);
  }, []);

  return { focusedId, setFocusedId };
}

interface DrawerVerbsOpts {
  active: boolean;
  verbs: Partial<Record<"E" | "R" | "H" | "S", () => void>>;
}

export function useDrawerVerbs({ active, verbs }: DrawerVerbsOpts) {
  const legendOpen = useAdminLegendStore((s) => s.open);
  const paletteOpen = useAdminPaletteStore((s) => s.open);
  const effectiveActive = active && !legendOpen && !paletteOpen;

  const verbsRef = useRef(verbs);
  useEffect(() => {
    verbsRef.current = verbs;
  }, [verbs]);

  useEffect(() => {
    if (!effectiveActive) return;
    function onKey(e: KeyboardEvent) {
      if (isTypingTarget(e.target)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const k = e.key.toUpperCase();
      if (k !== "E" && k !== "R" && k !== "H" && k !== "S") return;
      const handler = verbsRef.current[k as "E" | "R" | "H" | "S"];
      if (!handler) return;
      e.preventDefault();
      handler();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [effectiveActive]);
}

interface SearchFocusOpts {
  active: boolean;
  inputRef: RefObject<HTMLInputElement | null>;
}

export function useSearchFocusShortcut({ active, inputRef }: SearchFocusOpts) {
  const legendOpen = useAdminLegendStore((s) => s.open);
  const paletteOpen = useAdminPaletteStore((s) => s.open);
  const effectiveActive = active && !legendOpen && !paletteOpen;

  useEffect(() => {
    if (!effectiveActive) return;
    function onKey(e: KeyboardEvent) {
      if (e.key !== "/") return;
      if (isTypingTarget(e.target)) return;
      const input = inputRef.current;
      if (!input) return;
      e.preventDefault();
      input.focus();
      input.select();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [effectiveActive, inputRef]);
}

function cssEscape(id: string): string {
  if (typeof CSS !== "undefined" && typeof CSS.escape === "function") {
    return CSS.escape(id);
  }
  return id.replace(/[^\w-]/g, "\\$&");
}
