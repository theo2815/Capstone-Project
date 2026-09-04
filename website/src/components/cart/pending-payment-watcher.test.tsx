import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { PendingPaymentWatcher } from "@/components/cart/pending-payment-watcher";
import { useCartStore } from "@/store/cart-store";
import { usePendingPaymentStore } from "@/store/pending-payment-store";
import { useToastStore } from "@/store/toast-store";
import { useUiStore } from "@/store/ui-store";

const http = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn() }));
vi.mock("@/lib/api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api")>()),
  api: http,
}));

const pending = {
  orderId: "11111111-2222-4333-8444-555555555555",
  imageUrl: "data:image/png;base64,cXJwaA==",
  expiresAt: new Date(Date.now() + 30 * 60_000).toISOString(),
  returnToken: "return-token",
  email: "runner@example.com",
  total: 125,
  itemCount: 1,
  paidClaimedAt: null,
};

function mount() {
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <PendingPaymentWatcher />
    </QueryClientProvider>,
  );
}

// Mount, then act like the runner left the drawer the load-time reopen put up.
function mountAndLeave() {
  const r = mount();
  useUiStore.getState().closeCheckout();
  return r;
}

describe("PendingPaymentWatcher", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useUiStore.setState({ checkoutOpen: false, cartOpen: false });
    useToastStore.setState({ toasts: [] });
    useCartStore.setState({
      items: [{ photoId: "photo-1", eventId: "event-1", thumbnailUrl: "/p.jpg", price: 125 }],
      syncEnabled: false,
    });
    usePendingPaymentStore.setState({ pending });
  });

  it("toasts the confirmation and clears the cart + record when the order settles", async () => {
    http.get.mockResolvedValue({ id: pending.orderId, status: "FULFILLED", paidAt: "2026-09-05T10:00:00Z" });
    mountAndLeave();

    await waitFor(() => expect(usePendingPaymentStore.getState().pending).toBeNull());
    expect(http.get).toHaveBeenCalledWith(`/orders/${pending.orderId}/status?token=return-token`);
    expect(useCartStore.getState().items).toHaveLength(0);
    const toast = useToastStore.getState().toasts[0];
    expect(toast.kind).toBe("success");
    expect(toast.message).toContain("Ref 11111111");
    expect(toast.link?.href).toBe(`/orders/return?orderId=${pending.orderId}&token=return-token`);
  });

  it("drops the record and says nothing was charged when the QR expired", async () => {
    http.get.mockResolvedValue({ id: pending.orderId, status: "EXPIRED", paidAt: null });
    mountAndLeave();

    await waitFor(() => expect(usePendingPaymentStore.getState().pending).toBeNull());
    expect(useCartStore.getState().items).toHaveLength(1);
    expect(useToastStore.getState().toasts[0].message).toMatch(/nothing was charged/i);
  });

  it("reopens the checkout once on load when a QR is live", async () => {
    http.get.mockResolvedValue({ id: pending.orderId, status: "PENDING", paidAt: null });
    mount();

    await waitFor(() => expect(useUiStore.getState().checkoutOpen).toBe(true));
    // An intentional close later in the same page life must stick.
    useUiStore.getState().closeCheckout();
    await new Promise((r) => setTimeout(r, 30));
    expect(useUiStore.getState().checkoutOpen).toBe(false);
  });

  it("stays idle while the checkout modal owns the poll", async () => {
    useUiStore.setState({ checkoutOpen: true });
    http.get.mockResolvedValue({ id: pending.orderId, status: "FULFILLED", paidAt: null });
    mount();

    await new Promise((r) => setTimeout(r, 50));
    expect(http.get).not.toHaveBeenCalled();
    expect(usePendingPaymentStore.getState().pending).not.toBeNull();
  });
});
