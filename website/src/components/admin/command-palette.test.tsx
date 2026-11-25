import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useAdminPaletteStore } from "@/store/admin-palette-store";
import { CommandPalette } from "./command-palette";
import { adminUser } from "@/test/fixtures/admin";

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn(), replace: vi.fn() }) }));
const users = vi.hoisted(() => ({ rows: [] as unknown[], failed: false }));
vi.mock("@/hooks/use-admin-users", () => ({ useAdminUsers: () => ({ data: users.rows, isError: users.failed }), EMPTY_ADMIN_USERS: [] }));
vi.mock("@/hooks/use-admin-disputes", () => ({ useAdminDisputes: () => ({ data: [] }), EMPTY_DISPUTES: [] }));
vi.mock("@/hooks/use-admin-payouts", () => ({ useAdminPayouts: () => ({ data: [] }), EMPTY_PAYOUTS: [] }));
vi.mock("@/hooks/use-admin-data", () => ({ useAdminFlags: () => ({ items: [] }), EMPTY_FLAGS: [] }));

beforeEach(() => { act(() => useAdminPaletteStore.setState({ open: true, recents: [] })); });
afterEach(() => { cleanup(); users.rows = []; users.failed = false; act(() => useAdminPaletteStore.setState({ open: false })); });

// FU-24 (U24.2): a runner from the loaded user list shows under its own Runners label.
it("a typed runner name finds the runner under Runners", () => {
  users.rows = [{ ...adminUser, userId: "runner-1", role: "RUNNER", name: "Joana Reyes", email: "joana@example.test", brandName: null, handle: null }];
  render(<CommandPalette />);
  fireEvent.change(screen.getByRole("combobox", { name: "Search admin" }), { target: { value: "joana" } });
  const option = screen.getAllByRole("option").find((o) => o.textContent?.includes("Joana Reyes"));
  expect(option).toBeDefined();
  expect(option).toHaveTextContent(/^Runners/);
});

// FU-24 (U24.3): a failed user list says so once two characters are typed, and pages still match.
it("a failed people list shows the failure line and keeps the loaded rows", () => {
  users.failed = true;
  render(<CommandPalette />);
  const box = screen.getByRole("combobox", { name: "Search admin" });
  fireEvent.change(box, { target: { value: "s" } });
  expect(screen.queryByText("Couldn't search people. The list shows what is loaded.")).not.toBeInTheDocument();
  fireEvent.change(box, { target: { value: "security" } });
  expect(screen.getByText("Couldn't search people. The list shows what is loaded.")).toBeInTheDocument();
  expect(screen.getAllByRole("option").length).toBeGreaterThan(0);
});

it("the dialog has a name", () => {
  render(<CommandPalette />);
  expect(screen.getByRole("dialog", { name: "Search admin" })).toBeTruthy();
});

it("the search box is a combobox, collapsed while no list is shown", () => {
  render(<CommandPalette />);
  const box = screen.getByRole("combobox", { name: "Search admin" });
  expect(box).toHaveAttribute("aria-expanded", "false");
  expect(box).not.toHaveAttribute("aria-controls");
});

it("expands over the result list and announces the count politely", () => {
  render(<CommandPalette />);
  const box = screen.getByRole("combobox", { name: "Search admin" });
  fireEvent.change(box, { target: { value: "security" } });
  expect(box).toHaveAttribute("aria-expanded", "true");
  expect(box).toHaveAttribute("aria-controls", screen.getByRole("listbox").id);
  const n = screen.getAllByRole("option").length;
  expect(screen.getByRole("status")).toHaveTextContent(new RegExp(`^${n} results?$`));
});

// FT-43 (B-06): the hints name only what the palette really finds (no runners, no events).
it("the empty and no-match hints name only what the palette finds", () => {
  render(<CommandPalette />);
  expect(screen.getByText("Search the admin: pages, verifications, disputes, flags, payouts and photographers.")).toBeInTheDocument();
  fireEvent.change(screen.getByRole("combobox", { name: "Search admin" }), { target: { value: "zzqx" } });
  expect(screen.getByText("Try a photographer's name, an id, or a page name.")).toBeInTheDocument();
  expect(screen.queryByText(/runner handle/)).not.toBeInTheDocument();
});
