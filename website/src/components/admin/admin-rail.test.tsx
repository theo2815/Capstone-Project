import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import type { AdminKpis } from "@/lib/api-admin";
import { useAdminPaletteStore } from "@/store/admin-palette-store";
import { AdminRail } from "./admin-rail";
import { AdminMobileStrip } from "./admin-mobile-strip";

const path = vi.hoisted(() => ({ value: "/admin/relevance" }));
vi.mock("next/navigation", () => ({ usePathname: () => path.value }));

const kpis = vi.hoisted(() => ({ value: null as AdminKpis | null }));
vi.mock("@/hooks/use-admin-data", () => ({ useAdminKpis: () => kpis.value }));

const base: AdminKpis = {
  pendingVerifications: 0, approvedPhotographers: 0, suspended: 0, liveEvents: 0,
  decisionsThisWeek: 0, openDisputes: 0, openFlags: 0, pendingPayouts: 0, pendingEventRequests: 0,
};

afterEach(() => {
  cleanup(); path.value = "/admin/relevance"; kpis.value = null;
  useAdminPaletteStore.setState({ open: false });
});

const rowNames = () => screen.getAllByRole("link").map((l) => l.textContent?.replace(/\d+/g, "").trim());

it("the rail groups the rows: queues first, Security last under Account", () => {
  render(<AdminRail />);
  expect(rowNames()).toEqual(["Inbox", "Disputes", "Flags", "Verifications", "Event requests", "Payouts",
    "Photo Relevance", "Photo removals", "Events", "Photographers", "Runners", "Overview", "Sales", "Security"]);
  const catalog = screen.getByText("Catalog").parentElement!;
  expect(within(catalog).getByRole("link", { name: /Runners/ })).toHaveAttribute("href", "/admin/runners");
  const account = screen.getByText("Account").parentElement!;
  expect(within(account).getByRole("link", { name: /Security/ })).toHaveAttribute("href", "/admin/security");
});

it("the Verifications row is active on /admin/verifications, and Inbox is not", () => {
  path.value = "/admin/verifications";
  render(<AdminRail />);
  expect(screen.getByRole("link", { name: /Verifications/ })).toHaveAttribute("aria-current", "page");
  expect(screen.getByRole("link", { name: /Inbox/ })).not.toHaveAttribute("aria-current");
});

it("Photo removals shows its count, and Payouts adds the open reports", () => {
  kpis.value = { ...base, removalsNeedingAction: 3, payoutsAwaitingReview: 2, pendingPayouts: 9, openPayoutReports: 1 };
  render(<AdminRail />);
  expect(within(screen.getByRole("link", { name: /Photo removals/ })).getByLabelText("3 open")).toHaveTextContent("3");
  expect(within(screen.getByRole("link", { name: /Payouts/ })).getByLabelText("3 open")).toHaveTextContent("3");
  expect(within(screen.getByRole("link", { name: /Inbox/ })).getByLabelText("6 open")).toBeInTheDocument();
  expect(screen.getByText("Queues · 6 open")).toBeInTheDocument();
});

it("Photo Relevance counts the cases that need a person", () => {
  kpis.value = { ...base, relevanceNeedsPerson: 4, relevanceHidden: 40, relevanceAppealsOpen: 1 };
  render(<AdminRail />);
  expect(within(screen.getByRole("link", { name: /Photo Relevance/ })).getByLabelText("4 open")).toBeInTheDocument();
});

it("an old backend without the new fields shows no badges and no crash", () => {
  kpis.value = base;
  render(<AdminRail />);
  expect(screen.queryByLabelText(/open$/)).not.toBeInTheDocument();
  expect(screen.getByText("Queues · all clear")).toBeInTheDocument();
});

it("the rail's Search button opens the palette, with the shortcut key printed on it", () => {
  render(<AdminRail />);
  const search = screen.getByRole("button", { name: "Search the admin" });
  expect(search).toHaveTextContent("Search");
  expect(search).toHaveTextContent(/(Ctrl|⌘) K/);
  fireEvent.click(search);
  expect(useAdminPaletteStore.getState().open).toBe(true);
});

it("the rail footer sits outside the scrolling list, so it stays in view", () => {
  render(<AdminRail />);
  const nav = screen.getByRole("navigation", { name: "Admin sections" });
  const aside = nav.parentElement!;
  expect(aside.tagName).toBe("ASIDE");
  expect(nav.className).toContain("md:overflow-y-auto");
  expect(aside.className).not.toContain("md:overflow-y-auto");
  for (const name of ["Search the admin", "Keyboard shortcuts (press ?)", "Hide tips"]) {
    const control = screen.getByRole("button", { name });
    expect(nav.contains(control)).toBe(false);
    expect(aside.contains(control)).toBe(true);
  }
});

it("the shortcuts pill says Shortcuts and keeps its ? key", () => {
  render(<AdminRail />);
  const pill = screen.getByRole("button", { name: "Keyboard shortcuts (press ?)" });
  expect(pill).toHaveTextContent("Shortcuts");
  expect(pill).toHaveTextContent("?");
  expect(pill).not.toHaveTextContent("⌨");
});

it("the mobile strip starts with a Search button that opens the palette", () => {
  kpis.value = { ...base, openDisputes: 2 };
  const { container } = render(<AdminMobileStrip />);
  const first = container.querySelector("li")!;
  const search = within(first).getByRole("button", { name: "Search" });
  expect(search.className).toContain("min-h-[44px]");
  fireEvent.click(search);
  expect(useAdminPaletteStore.getState().open).toBe(true);
});

it("the mobile strip opens with Search and the open total, then the queues, a divider, then the rest", () => {
  kpis.value = { ...base, openDisputes: 2 };
  const { container } = render(<AdminMobileStrip />);
  const items = [...container.querySelectorAll("li")];
  expect(items[0]).toHaveTextContent("Search");
  expect(items[1]).toHaveTextContent("2 open");
  const divider = items.findIndex((li) => li.getAttribute("data-divider") !== null);
  expect(items.slice(2, divider).map((li) => li.textContent?.replace(/\d+/g, "").trim()))
    .toEqual(["Inbox", "Disputes", "Flags", "Verifications", "Event requests", "Payouts", "Photo Relevance", "Photo removals"]);
  expect(items[divider + 1]).toHaveTextContent("Events");
  for (const link of screen.getAllByRole("link")) expect(link.className).toContain("min-h-[44px]");
});
