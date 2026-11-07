import { render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { DashboardRail } from "@/components/dashboard/dashboard-rail";

// The studio has no footer (removed 2026-05-06 to keep the rail in view), so
// the Privacy and Terms links sit at the bottom of the rail instead.
vi.mock("next/navigation", () => ({ usePathname: () => "/dashboard" }));
vi.mock("@/hooks/use-auth", () => ({
  useAuth: () => ({ user: { id: "p1", name: "Shooter", email: "p@example.test", role: "PHOTOGRAPHER", createdAt: "2026-01-01T00:00:00Z" } }),
}));
vi.mock("@/hooks/use-can-upload", () => ({ useCanUpload: () => ({ kind: "ok" }) }));

it("the studio rail links to the Privacy Policy and the Terms", () => {
  render(<DashboardRail />);

  const legal = screen.getByRole("navigation", { name: "Legal" });
  expect(legal).toContainElement(screen.getByRole("link", { name: "Privacy" }));
  expect(screen.getByRole("link", { name: "Privacy" })).toHaveAttribute("href", "/privacy");
  expect(screen.getByRole("link", { name: "Terms" })).toHaveAttribute("href", "/terms");
});
