import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import { useCartStore } from "@/store/cart-store";
import OrdersPage from "@/app/orders/page";
import type { RunnerDispute } from "@/lib/api-orders";

vi.mock("next/navigation", () => ({
  usePathname: () => "/orders", useSearchParams: () => new URLSearchParams(""),
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
}));
vi.mock("@/components/layout/site-header", () => ({ SiteHeader: () => null }));
vi.mock("@/components/profile-shell", () => ({
  IdentityRail: () => null, Slab: ({ children }: { children: React.ReactNode }) => <section>{children}</section>,
}));
vi.mock("@/components/orders/refund-modal", () => ({ RefundModal: () => null }));

const runner = { id: "runner-1", role: "RUNNER" as const, name: "Runner", email: "runner@example.test", createdAt: "2026-01-01T00:00:00Z" };
const link = "https://pm.test/refund/abc";
const dispute = (over: Partial<RunnerDispute>): RunnerDispute => ({
  id: "d1", photoId: "p1", reason: "other", note: "", status: "escalated", resolution: null, refundAmount: null,
  resolutionNote: null, openedAt: "2026-10-20T00:00:00Z", resolvedAt: null, withdrawnAt: null, refundStatus: "manual_review", ...over,
});

afterEach(() => {
  vi.restoreAllMocks(); useAuthStore.getState().logout();
  useCartStore.getState().setSyncEnabled(false); useCartStore.getState().setItems([]);
});

function renderOrders(d: RunnerDispute) {
  useAuthStore.getState().startSession(runner);
  vi.spyOn(api, "get").mockResolvedValue({
    items: [{ id: "order-1", eventId: "event-1", photoIds: ["p1"], total: 80, paymentMethod: "qrph", paidAt: "2026-10-19T00:00:00Z",
      eventName: "Cebu Run", status: "PAID", disputes: [d] }],
    total: 1, offset: 0, limit: 200,
  });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><OrdersPage /></QueryClientProvider>);
}

it("shows the claim link and claim-by date when the admin sent a link", async () => {
  renderOrders(dispute({ refundLinkUrl: link, refundLinkSentAt: "2026-10-27T12:00:00Z", refundLinkExpiresAt: "2026-10-30T12:00:00Z" }));
  const a = await screen.findByRole("link", { name: /Claim your refund/ });
  expect(a).toHaveAttribute("href", link);
  expect(a).toHaveAttribute("target", "_blank");
  expect(a).toHaveAttribute("rel", "noopener noreferrer");
  expect(screen.getByText("Refund link sent")).toBeVisible();
  expect(screen.getByText("Sent OCT 27 · 2026. Claim by OCT 30 · 2026.")).toBeVisible();
});

it("labels the opened order with the link state, not 'Refund pending review'", async () => {
  renderOrders(dispute({ refundLinkUrl: link, refundLinkSentAt: "2026-10-27T12:00:00Z" }));
  fireEvent.click(await screen.findByRole("button", { name: /View photos/ }));
  expect(await screen.findByRole("button", { name: /Hide photos/ })).toBeVisible();
  expect(screen.queryByText("Refund pending review")).not.toBeInTheDocument();
  // the chip, the refund history row, and the line under the photos
  expect(screen.getAllByText("Refund link sent")).toHaveLength(3);
});

it("shows only the sent date when the server gave no claim-by date", async () => {
  renderOrders(dispute({ refundLinkUrl: link, refundLinkSentAt: "2026-10-27T12:00:00Z" }));
  expect(await screen.findByText("Sent OCT 27 · 2026.")).toBeVisible();
});

it("shows no claim link before a link exists", async () => {
  renderOrders(dispute({}));
  expect(await screen.findByText("Refund approved, waiting for PayMongo refund link")).toBeVisible();
  expect(screen.queryByRole("link", { name: /Claim your refund/ })).not.toBeInTheDocument();
});

it("shows no claim link after the refund succeeded", async () => {
  renderOrders(dispute({ status: "resolved", refundStatus: "succeeded", refundAmount: 80, refundLinkUrl: link, refundLinkSentAt: "2026-10-27T12:00:00Z" }));
  expect(await screen.findByText(/Refund approved/)).toBeVisible();
  expect(screen.queryByRole("link", { name: /Claim your refund/ })).not.toBeInTheDocument();
});
