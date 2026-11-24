import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useAdminPaletteStore } from "@/store/admin-palette-store";
import { CommandPalette } from "./command-palette";

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn(), replace: vi.fn() }) }));
vi.mock("@/hooks/use-admin-users", () => ({ useAdminUsers: () => ({ data: [] }), EMPTY_ADMIN_USERS: [] }));
vi.mock("@/hooks/use-admin-disputes", () => ({ useAdminDisputes: () => ({ data: [] }), EMPTY_DISPUTES: [] }));
vi.mock("@/hooks/use-admin-payouts", () => ({ useAdminPayouts: () => ({ data: [] }), EMPTY_PAYOUTS: [] }));
vi.mock("@/hooks/use-admin-data", () => ({ useAdminFlags: () => ({ items: [] }), EMPTY_FLAGS: [] }));

beforeEach(() => { act(() => useAdminPaletteStore.setState({ open: true, recents: [] })); });
afterEach(() => { cleanup(); act(() => useAdminPaletteStore.setState({ open: false })); });

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
