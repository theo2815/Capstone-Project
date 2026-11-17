import { cleanup, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useAuthStore } from "@/store/auth-store";
import AccountPage from "./page";

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn(), replace: vi.fn() }), usePathname: () => "/account" }));
vi.mock("@/components/auth/protected-route", () => ({ ProtectedRoute: ({ children }: { children: React.ReactNode }) => <>{children}</> }));
vi.mock("@/components/layout/site-header", () => ({ SiteHeader: () => null }));
vi.mock("@/components/account/avatar-slab", () => ({ AvatarSlab: () => null }));

// The jump-section rail watches the slabs; jsdom has no IntersectionObserver.
vi.stubGlobal("IntersectionObserver", class { observe() {} unobserve() {} disconnect() {} });
afterEach(() => { cleanup(); useAuthStore.getState().logout(); });
function mount(role: "RUNNER" | "PHOTOGRAPHER") {
  useAuthStore.getState().setUser({ id: "u-1", role, name: "A", email: "a@example.test", createdAt: "2026-09-10" });
  return render(<QueryClientProvider client={new QueryClient()}><AccountPage /></QueryClientProvider>);
}

describe("account danger slab", () => {
  it("the account slab links to the request page", () => {
    mount("RUNNER");
    expect(screen.getByRole("link", { name: "Request photo deletion →" })).toHaveAttribute("href", "/account/photo-deletion/new");
    expect(screen.getByRole("link", { name: "Your deletion requests →" })).toHaveAttribute("href", "/account/photo-deletion");
  });

  it("photographers do not see the runner deletion links", () => {
    mount("PHOTOGRAPHER");
    expect(screen.getByText("Sign out of QuickPitik.")).toBeVisible();
    expect(screen.queryByRole("link", { name: "Request photo deletion →" })).not.toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Your deletion requests →" })).not.toBeInTheDocument();
  });
});
