import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { CheckoutModal } from "@/components/cart/checkout-modal";
import { useAuthStore } from "@/store/auth-store";
import { useCartStore } from "@/store/cart-store";

const orderApi = vi.hoisted(() => ({
  postOrder: vi.fn(),
  fetchOrderStatus: vi.fn(),
  fetchOrderStatusForUser: vi.fn(),
}));

vi.mock("@/lib/api-orders", () => orderApi);
vi.mock("next/navigation", () => ({
  usePathname: () => "/events/race",
  useSearchParams: () => new URLSearchParams(),
}));

describe("CheckoutModal QRPH", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useAuthStore.setState({ user: null, isAuthenticated: false, isLoading: false });
    useCartStore.setState({
      items: [
        {
          photoId: "photo-1",
          eventId: "event-1",
          thumbnailUrl: "/photo.jpg",
          price: 125,
        },
      ],
      syncEnabled: false,
    });
    orderApi.fetchOrderStatus.mockResolvedValue({
      id: "order-1",
      status: "PENDING",
      paidAt: null,
    });
    orderApi.postOrder.mockResolvedValue({
      id: "order-1",
      status: "PENDING",
      items: [],
      totalAmount: 125,
      paymentMethod: "qrph",
      createdAt: "2026-09-04T12:00:00Z",
      qrPh: {
        imageUrl: "data:image/png;base64,cXJwaA==",
        expiresAt: "2026-09-04T12:30:00Z",
        returnToken: "return-token",
      },
    });
  });

  it("offers only QR Ph and renders a downloadable provider QR", async () => {
    const user = userEvent.setup();
    render(
      <QueryClientProvider client={new QueryClient()}>
        <CheckoutModal isOpen onClose={vi.fn()} />
      </QueryClientProvider>,
    );

    await user.type(screen.getByLabelText("Email"), "runner@example.com");
    await user.type(screen.getByLabelText("Confirm email"), "runner@example.com");
    await user.click(screen.getByRole("button", { name: "Continue →" }));

    expect(screen.getByText("QR Ph")).toBeInTheDocument();
    expect(screen.queryByText("GCash")).not.toBeInTheDocument();
    expect(screen.queryByText("Maya")).not.toBeInTheDocument();
    expect(screen.queryByText("Credit / Debit card")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Generate QR to pay/ }));

    expect(await screen.findByRole("img", { name: /QR Ph payment code/ })).toHaveAttribute(
      "src",
      "data:image/png;base64,cXJwaA==",
    );
    expect(screen.getByRole("link", { name: "Save QR code" })).toHaveAttribute(
      "download",
      "quickpitik-qrph-order-1.png",
    );
    await waitFor(() =>
      expect(orderApi.postOrder).toHaveBeenCalledWith(
        expect.objectContaining({ paymentMethod: "qrph" }),
      ),
    );

    orderApi.fetchOrderStatus.mockResolvedValue({
      id: "order-1",
      status: "FULFILLED",
      paidAt: "2026-09-04T12:01:00Z",
    });
    const checkButton = screen.getByRole("button", {
      name: /I've paid · Check status/,
    });
    await waitFor(() => expect(checkButton).toBeEnabled());
    await user.click(checkButton);

    expect(await screen.findByText("All yours.")).toBeInTheDocument();
    expect(useCartStore.getState().items).toHaveLength(0);
  });
});
