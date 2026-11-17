import { readFileSync } from "node:fs";
import path from "node:path";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import OrdersPage from "@/app/orders/page";

// Signed-in pages had no footer, so the Privacy and Terms links were missing
// there (privacy browser pass, 2026-11-07). ProfileShellFooter carries them.
vi.mock("next/navigation", () => ({
  usePathname: () => "/orders", useSearchParams: () => new URLSearchParams(""),
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
}));
vi.mock("@/components/layout/site-header", () => ({ SiteHeader: () => null }));
vi.mock("@/components/profile-shell", async (importActual) => ({
  ...(await importActual<typeof import("@/components/profile-shell")>()),
  IdentityRail: () => null,
}));

afterEach(() => {
  vi.restoreAllMocks();
  useAuthStore.getState().logout();
});

it("the orders page shows the Privacy and Terms links", async () => {
  useAuthStore.getState().startSession({ id: "r1", role: "RUNNER", name: "Runner", email: "r@example.test", createdAt: "2026-01-01T00:00:00Z" });
  vi.spyOn(api, "get").mockResolvedValue({ items: [], total: 0, offset: 0, limit: 200 });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><OrdersPage /></QueryClientProvider>);

  expect(await screen.findByRole("link", { name: /privacy/i })).toHaveAttribute("href", "/privacy");
  expect(screen.getByRole("link", { name: /terms/i })).toHaveAttribute("href", "/terms");
});

it.each(["account", "profile"])("the %s page mounts the footer with the legal links", (page) => {
  const source = readFileSync(path.join(__dirname, page, "page.tsx"), "utf8");
  expect(source).toContain("<ProfileShellFooter />");
});
