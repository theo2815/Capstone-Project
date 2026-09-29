import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import { useCartStore } from "@/store/cart-store";
import { CartModal } from "@/components/cart/cart-modal";
import OrdersPage from "@/app/orders/page";
import { DisputesQueue } from "@/components/admin/disputes-queue";

vi.mock("next/navigation", () => ({
  usePathname: () => "/cart", useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
}));
vi.mock("@/hooks/use-url-state", async () => {
  const React = await import("react");
  return { useUrlState: () => React.useState("") };
});
vi.mock("@/hooks/use-orders", () => ({
  useOrdersList: () => ({ orders: [{ id: "order-1", eventId: "event-1", photoIds: ["order-photo"], total: 100,
    paymentMethod: "gcash", paidAt: "2026-09-28T00:00:00Z", eventName: "Cebu Run", status: "PAID" }], isLoading: false, error: null }),
  useOrderDetail: () => ({ detail: null, isLoading: false, error: null }),
}));
vi.mock("@/components/layout/site-header", () => ({ SiteHeader: () => null }));
vi.mock("@/components/profile-shell", () => ({
  IdentityRail: () => null, Slab: ({ children }: { children: React.ReactNode }) => <section>{children}</section>,
}));
vi.mock("@/hooks/use-confirmation", () => ({ useConfirmation: () => ({ confirm: async () => true }) }));
vi.mock("@/components/orders/refund-modal", () => ({ RefundModal: () => null }));
vi.mock("@/hooks/use-admin-disputes", async () => {
  const { dispute } = await import("@/test/fixtures/admin");
  return {
    EMPTY_DISPUTES: [], useAdminDisputes: () => ({ data: [{
      ...dispute, photoSnapshot: { ...dispute.photoSnapshot, thumbnailUrl: "https://photos.test/admin.jpg" },
    }], isPending: false, isError: false, isSuccess: true, isFetching: false }),
    useDisputeActions: () => ({ resolve: vi.fn(), deny: vi.fn(), escalate: vi.fn() }),
  };
});
vi.mock("@/hooks/use-admin-keyboard", () => ({ useQueueKeyboardNav: () => ({ focusedId: null, setFocusedId: vi.fn() }) }));

const runner = { id: "runner-1", role: "RUNNER" as const, name: "Runner", email: "runner@example.test", createdAt: "2026-01-01T00:00:00Z" };
afterEach(() => { vi.restoreAllMocks(); useAuthStore.getState().logout(); useCartStore.getState().setItems([]); });

function mount(ui: React.ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const view = render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
  return () => { view.unmount(); client.clear(); };
}

it("opens the actual cart preview without metadata and keeps checkout working", async () => {
  useCartStore.getState().setItems([{ photoId: "cart-photo", eventId: "event-1", thumbnailUrl: "", price: 100, bib: "101", eventName: "Cebu Run" }]);
  const get = vi.spyOn(api, "get");
  const checkout = vi.fn();
  const unmount = mount(<CartModal isOpen onClose={vi.fn()} onContinueToCheckout={checkout} />);
  const user = userEvent.setup();
  await user.click(screen.getByRole("button", { name: "Preview 101 from Cebu Run" }));
  expect(await screen.findByRole("dialog", { name: "Preview photo 101" })).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Show Metadata" })).toBeNull();
  expect(get).not.toHaveBeenCalled();
  await user.click(screen.getByRole("button", { name: "Checkout" }));
  expect(checkout).toHaveBeenCalledTimes(1);
  unmount();
});

it("opens the actual order preview without metadata", async () => {
  useAuthStore.getState().setUser(runner);
  const get = vi.spyOn(api, "get");
  const unmount = mount(<OrdersPage />);
  const user = userEvent.setup();
  await user.click(screen.getByRole("button", { name: "View photos" }));
  await user.click(screen.getByRole("button", { name: "Preview order-photo" }));
  expect(await screen.findByRole("button", { name: "Download photo" })).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Show Metadata" })).toBeNull();
  expect(get).not.toHaveBeenCalled();
  unmount();
});

it("opens the actual admin dispute review without metadata", async () => {
  const get = vi.spyOn(api, "get");
  const unmount = mount(<DisputesQueue />);
  const user = userEvent.setup();
  await user.click(screen.getByRole("button", { name: "Open dispute dispute-1" }));
  await user.click(await screen.findByRole("button", { name: "Open Runner at full size" }));
  expect(await screen.findByText("Admin · Read-only preview")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Show Metadata" })).toBeNull();
  expect(get).not.toHaveBeenCalled();
  unmount();
});
