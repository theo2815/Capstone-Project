"use client";

import { KeyboardLegend } from "@/components/admin/keyboard-legend";
import { CommandPalette } from "@/components/admin/command-palette";
import { useAdminLegendShortcut } from "@/hooks/use-admin-keyboard";
import { useCommandPaletteShortcut } from "@/hooks/use-command-palette-shortcut";

// Client-side wrapper for the shell-level keyboard pieces — mounts the
// global `?` and Cmd/Ctrl+K listeners, the legend portal (its hint pill
// sits in the rail footer), and
// the command palette portal. Lives in its own file so <AdminShell> can
// stay a thin server component (composing this alongside SiteHeader /
// AdminRail).
export function AdminShellKeyboard() {
  useAdminLegendShortcut();
  useCommandPaletteShortcut();
  return (
    <>
      <KeyboardLegend />
      <CommandPalette />
    </>
  );
}
