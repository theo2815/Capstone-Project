import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import OrdersPage from "@/app/orders/page";

vi.mock("next/navigation", () => ({
  usePathname: () => "/orders", useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
}));
vi.mock("@/components/layout/site-header", () => ({ SiteHeader: () => null }));
vi.mock("@/components/profile-shell", () => ({
  IdentityRail: () => null, ProfileShellFooter: () => null, Slab: ({ children }: { children: React.ReactNode }) => <section>{children}</section>,
}));
vi.mock("@/components/orders/refund-modal", () => ({ RefundModal: () => null }));

const runner = { id: "runner-1", role: "RUNNER" as const, name: "Runner", email: "runner@example.test", createdAt: "2026-01-01T00:00:00Z" };
const order = { id: "order-1", eventId: "event-1", photoIds: ["p1"], total: 80, paymentMethod: "qrph",
  paidAt: "2026-11-01T00:00:00Z", eventName: "Cebu Run", status: "PAID" as const };

afterEach(() => { vi.restoreAllMocks(); useAuthStore.getState().logout(); });

function mount() {
  useAuthStore.getState().startSession(runner);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}><OrdersPage /></QueryClientProvider>);
}

it("never tells a runner they have no purchases while their orders are still loading", () => {
  vi.spyOn(api, "get").mockReturnValue(new Promise(() => {}));
  mount();
  expect(screen.getByText(/loading your purchases/i)).toBeInTheDocument();
  expect(screen.queryByText("No purchases yet.")).not.toBeInTheDocument();
  expect(screen.queryByText(/totals will land here/i)).not.toBeInTheDocument();
});

it("shows an error with a retry instead of an empty history when orders fail to load", async () => {
  const get = vi.spyOn(api, "get").mockRejectedValueOnce(new Error("Server error"));
  mount();
  expect(await screen.findByRole("alert")).toHaveTextContent(/couldn't load your purchases/i);
  expect(screen.queryByText("No purchases yet.")).not.toBeInTheDocument();
  expect(screen.queryByText(/totals will land here/i)).not.toBeInTheDocument();

  get.mockResolvedValue({ items: [order], total: 1, offset: 0, limit: 200 });
  await userEvent.setup().click(screen.getByRole("button", { name: "Try again" }));
  expect(await screen.findByText("Cebu Run")).toBeInTheDocument();
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
});

it("still shows the empty history when the runner really has no purchases", async () => {
  vi.spyOn(api, "get").mockResolvedValue({ items: [], total: 0, offset: 0, limit: 200 });
  mount();
  expect(await screen.findByText("No purchases yet.")).toBeInTheDocument();
});
