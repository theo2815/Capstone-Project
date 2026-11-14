import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { AdminRail } from "./admin-rail";
import { AdminMobileStrip } from "./admin-mobile-strip";

vi.mock("next/navigation", () => ({ usePathname: () => "/admin/relevance" }));
vi.mock("@/lib/admin-queues", () => ({
  useAdminAttentionTarget: () => null,
  useAdminQueueCounts: () => ({ inbox: 0, disputes: 0, flags: 0, payouts: 0, events: 0, total: 0 }),
}));

afterEach(cleanup);

it("the rail shows the Photo Relevance item", () => {
  render(<AdminRail />);
  const link = screen.getByRole("link", { name: /Photo Relevance/ });
  expect(link).toHaveAttribute("href", "/admin/relevance");
  expect(link).toHaveAttribute("aria-current", "page");
  expect(within(link).getByText("12")).toBeInTheDocument();
  // It sits with the queues, after Event requests, and leaves every other row as it was.
  const names = screen.getAllByRole("link").map((l) => l.textContent?.replace(/\d+/g, "").trim());
  expect(names).toEqual(["Inbox", "Disputes", "Flags", "Payouts", "Event requests", "Photo Relevance",
    "Events", "Photographers", "Photo removals", "Overview", "Sales", "Security"]);
});

it("the mobile strip shows the Photo Relevance item", () => {
  render(<AdminMobileStrip />);
  const link = screen.getByRole("link", { name: /Photo Relevance/ });
  expect(link).toHaveAttribute("href", "/admin/relevance");
  expect(link).toHaveAttribute("aria-current", "page");
});
