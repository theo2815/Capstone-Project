"use client";

import { Modal } from "@/components/ui/modal";
import { Kicker } from "@/components/ui/kicker";
import { useAdminLegendStore } from "@/store/admin-legend-store";

interface ShortcutRow {
  label: string;
  keys: string[];
  scope: "Queue" | "Drawer" | "Anywhere";
}

const ROWS: ShortcutRow[] = [
  { label: "Open command palette", keys: ["⌘", "K"], scope: "Anywhere" },
  { label: "Navigate rows", keys: ["J", "K"], scope: "Queue" },
  { label: "Open focused row", keys: ["Enter"], scope: "Queue" },
  { label: "Focus search", keys: ["/"], scope: "Queue" },
  { label: "Approve", keys: ["E"], scope: "Drawer" },
  { label: "Reject", keys: ["R"], scope: "Drawer" },
  { label: "Hide", keys: ["H"], scope: "Drawer" },
  { label: "Suspend", keys: ["S"], scope: "Drawer" },
  { label: "Close drawer / modal", keys: ["Esc"], scope: "Anywhere" },
  { label: "Toggle this legend", keys: ["?"], scope: "Anywhere" },
];

// Phase 4 admin keyboard shortcut sheet. Mounts once in <AdminShell>; the
// Modal primitive portals to document.body so it renders at z-60 above
// any drawer or queue surface. Triggered by the global `?` shortcut (see
// useAdminLegendShortcut) and by the bottom-right <KeyboardHintPill>.
//
// Verbs are scope-aware — pressing E/R/H/S inside a drawer only fires
// when the queue has wired the verb (e.g. Approve runs in verifications
// and pending payouts; Hide runs only inside an open flag drawer). The
// modal copy makes that clear so users don't expect blanket coverage.
export function KeyboardLegend() {
  const open = useAdminLegendStore((s) => s.open);
  const setOpen = useAdminLegendStore((s) => s.setOpen);
  return (
    <Modal
      isOpen={open}
      onClose={() => setOpen(false)}
      title="Keyboard shortcuts"
    >
      <ul className="space-y-3">
        {ROWS.map((row) => (
          <li
            key={row.label}
            className="flex items-center justify-between gap-4 border-b border-line pb-3 last:border-b-0 last:pb-0"
          >
            <div className="min-w-0">
              <p className="font-sans text-sm text-ink">{row.label}</p>
              <Kicker as="p" tone="soft" className="mt-1">
                {row.scope}
              </Kicker>
            </div>
            <div className="flex items-center gap-1.5 shrink-0">
              {row.keys.map((k, i) => (
                <kbd
                  key={i}
                  className="font-mono text-[12px] text-ink rounded-md border border-line bg-bone-deep px-2 py-1 min-w-[2rem] text-center"
                >
                  {k}
                </kbd>
              ))}
            </div>
          </li>
        ))}
      </ul>
      <p className="font-sans text-[13px] text-slate-soft mt-6">
        ⌘K (Ctrl+K on Windows / Linux) opens the command palette — search
        every page, queue row, and photographer in one place. Verbs only
        fire when the open drawer has that action available; outside a
        drawer, verb keys are ignored.
      </p>
    </Modal>
  );
}
