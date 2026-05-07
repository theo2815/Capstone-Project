"use client";

import { useEffect } from "react";
import { useAdminPaletteStore } from "@/store/admin-palette-store";
import { useAdminLegendStore } from "@/store/admin-legend-store";
import { isTypingTarget } from "@/hooks/use-admin-keyboard";

// Phase 5 admin command palette — global Cmd+K / Ctrl+K listener. Mounts
// once inside <AdminShell> via <AdminShellKeyboard>. Suppresses while the
// keyboard legend is on top so the legend's own ESC handler stays in
// control. The browser default for Cmd+K is "open search bar" in some
// browsers — preventDefault unconditionally to claim the chord.
//
// `isTypingTarget` is intentionally NOT checked here: power users expect
// Cmd+K to open the palette even from inside an input. The palette's own
// search field becomes focused, so the keystroke that opened it doesn't
// leak as a character into the original input.

export function useCommandPaletteShortcut() {
  const toggle = useAdminPaletteStore((s) => s.toggle);
  const legendOpen = useAdminLegendStore((s) => s.open);
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const isK = e.key === "k" || e.key === "K";
      if (!isK) return;
      if (!(e.metaKey || e.ctrlKey)) return;
      if (e.altKey || e.shiftKey) return;
      if (legendOpen) return;
      e.preventDefault();
      toggle();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle, legendOpen]);
}

// Re-export so palette internals can keep typing-target checks consistent.
export { isTypingTarget };
