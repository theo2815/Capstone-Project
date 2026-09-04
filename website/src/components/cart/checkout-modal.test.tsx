import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CheckoutModal } from "@/components/cart/checkout-modal";
import { useAuthStore } from "@/store/auth-store";
import { useCartStore } from "@/store/cart-store";
import { useConfirmationStore } from "@/store/confirmation-store";
import {
  usePendingPaymentStore,
  type PendingPayment,
} from "@/store/pending-payment-store";

// Mock the HTTP client rather than api-orders so the real endpoint choice
// (guest token route vs /me route, `?verify=true`) is what gets asserted.
const http = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn() }));
vi.mock("@/lib/api", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api")>()),
  api: http,
}));
vi.mock("next/navigation", () => ({
  usePathname: () => "/events/race",
  useSearchParams: () => new URLSearchParams(),
}));

const ORDER_ID = "11111111-2222-4333-8444-555555555555";
const STATUS_PATH = `/orders/${ORDER_ID}/status?token=return-token`;
const pendingStatus = { id: ORDER_ID, status: "PENDING", paidAt: null };
const fulfilledStatus = { id: ORDER_ID, status: "FULFILLED", paidAt: "2026-09-04T12:01:00Z" };
const expiresAt = () => new Date(Date.now() + 30 * 60_000).toISOString();

function pendingRecord(overrides: Partial<PendingPayment> = {}): PendingPayment {
  const base: PendingPayment = {
    orderId: ORDER_ID,
    imageUrl: "data:image/png;base64,cXJwaA==",
    expiresAt: expiresAt(),
    returnToken: "return-token" as string | null,
    email: "runner@example.com",
    total: 125,
    itemCount: 1,
    paidClaimedAt: null as string | null,
  };
  return { ...base, ...overrides };
}

function renderModal(onClose = vi.fn()) {
  render(
    <QueryClientProvider client={new QueryClient()}>
      <CheckoutModal isOpen onClose={onClose} />
    </QueryClientProvider>,
  );
  return onClose;
}

// Fake timers + async advance: flushes the awaited status promises between
// ticks so the in-flight guard doesn't swallow every later poll.
const tick = (ms: number) => act(() => vi.advanceTimersByTimeAsync(ms));

async function walkToQr(user: ReturnType<typeof userEvent.setup>) {
  await user.type(screen.getByLabelText("Email"), "runner@example.com");
  await user.type(screen.getByLabelText("Confirm email"), "runner@example.com");
  await user.click(screen.getByRole("button", { name: "Continue →" }));
  await user.click(screen.getByRole("button", { name: /Generate QR to pay/ }));
  await screen.findByRole("img", { name: /QR Ph payment code/ });
}

describe("CheckoutModal QRPH", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers({ shouldAdvanceTime: true });
    useAuthStore.setState({ user: null, isAuthenticated: false, isLoading: false });
    usePendingPaymentStore.setState({ pending: null });
    useConfirmationStore.setState({ active: null, loading: false, error: null });
    useCartStore.setState({
      items: [{ photoId: "photo-1", eventId: "event-1", thumbnailUrl: "/photo.jpg", price: 125 }],
      syncEnabled: false,
    });
    http.get.mockResolvedValue(pendingStatus);
    http.post.mockResolvedValue({
      id: ORDER_ID,
      status: "PENDING",
      items: [],
      totalAmount: 125,
      paymentMethod: "qrph",
      createdAt: "2026-09-04T12:00:00Z",
      qrPh: { imageUrl: "data:image/png;base64,cXJwaA==", expiresAt: expiresAt(), returnToken: "return-token" },
    });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("offers only QR Ph, persists the pending payment, and only succeeds on a confirmed status", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    renderModal();
    await user.type(screen.getByLabelText("Email"), "runner@example.com");
    await user.type(screen.getByLabelText("Confirm email"), "runner@example.com");
    await user.click(screen.getByRole("button", { name: "Continue →" }));

    expect(screen.getByText("QR Ph")).toBeInTheDocument();
    expect(screen.queryByText("GCash")).not.toBeInTheDocument();
    expect(screen.queryByText("Maya")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Generate QR to pay/ }));
    expect(await screen.findByRole("img", { name: /QR Ph payment code/ })).toHaveAttribute(
      "src",
      "data:image/png;base64,cXJwaA==",
    );
    expect(screen.getByRole("link", { name: "Save QR code" })).toHaveAttribute(
      "download",
      `quickpitik-qrph-${ORDER_ID}.png`,
    );
    expect(http.post).toHaveBeenCalledWith(
      "/orders",
      expect.objectContaining({ paymentMethod: "qrph" }),
      expect.anything(),
    );
    expect(usePendingPaymentStore.getState().pending).toMatchObject({
      orderId: ORDER_ID,
      returnToken: "return-token",
      email: "runner@example.com",
      total: 125,
      itemCount: 1,
    });
    expect(screen.getByText("Waiting for your payment")).toBeInTheDocument();
    expect(screen.queryByText("All yours.")).not.toBeInTheDocument();

    // "I've paid" asks the backend to verify with PayMongo right away.
    await user.click(screen.getByRole("button", { name: "I’ve paid" }));
    await waitFor(() => expect(http.get).toHaveBeenCalledWith(`${STATUS_PATH}&verify=true`));
    expect(screen.getByText("Confirming your payment…")).toBeInTheDocument();
    expect(screen.getByText(/Don’t pay again/)).toBeInTheDocument();
    expect(screen.queryByText("All yours.")).not.toBeInTheDocument();

    http.get.mockResolvedValue(fulfilledStatus);
    await tick(3000);
    expect(await screen.findByText("All yours.")).toBeInTheDocument();
    expect(screen.getByText(/Payment confirmed/)).toBeInTheDocument();
    expect(screen.getByText("11111111")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /View receipt & download/ })).toHaveAttribute(
      "href",
      `/orders/return?orderId=${ORDER_ID}&token=return-token`,
    );
    expect(useCartStore.getState().items).toHaveLength(0);
    expect(usePendingPaymentStore.getState().pending).toBeNull();
  });

  it("asks before leaving while a QR is live and ignores Esc under the confirmation", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    const onClose = renderModal();
    await walkToQr(user);

    const [backdrop] = screen.getAllByRole("button", { name: "Close checkout" });
    await user.click(backdrop);
    await waitFor(() => expect(useConfirmationStore.getState().active).not.toBeNull());
    expect(useConfirmationStore.getState().active?.config.title).toBe("Leave checkout?");
    expect(onClose).not.toHaveBeenCalled();

    await user.keyboard("{Escape}");
    expect(onClose).not.toHaveBeenCalled();

    act(() => useConfirmationStore.getState().close(false));
    await waitFor(() => expect(useConfirmationStore.getState().active).toBeNull());
    expect(onClose).not.toHaveBeenCalled();
    expect(screen.getByRole("img", { name: /QR Ph payment code/ })).toBeInTheDocument();

    await user.keyboard("{Escape}");
    await waitFor(() => expect(useConfirmationStore.getState().active).not.toBeNull());
    act(() => useConfirmationStore.getState().close(true));
    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
    // Leaving keeps the payment so the cart pill can bring the runner back.
    expect(usePendingPaymentStore.getState().pending).not.toBeNull();
  });

  it("escalates to the slow tier after a minute and offers to wait for the email", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    const onClose = renderModal();
    await walkToQr(user);
    await user.click(screen.getByRole("button", { name: "I’ve paid" }));
    expect(await screen.findByText("Confirming your payment…")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Checking again in/ })).toBeDisabled();

    await tick(61_000);
    expect(await screen.findByText("Taking longer than usual")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Check again" })).toBeEnabled();
    expect(screen.queryByText("All yours.")).not.toBeInTheDocument();
    // Automatic verify kept asking PayMongo in the meantime (≤ 3/min).
    const verifyCalls = http.get.mock.calls.filter(([p]) => String(p).endsWith("&verify=true"));
    expect(verifyCalls.length).toBeGreaterThanOrEqual(2);
    expect(verifyCalls.length).toBeLessThanOrEqual(5);

    await user.click(screen.getByRole("button", { name: /wait for the email/ }));
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(usePendingPaymentStore.getState().pending).not.toBeNull();
  });

  it("returns to the payment step with a fresh key when the QR expires", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    renderModal();
    await walkToQr(user);
    const firstKey = http.post.mock.calls[0][2].headers["Idempotency-Key"];

    http.get.mockResolvedValue({ id: ORDER_ID, status: "EXPIRED", paidAt: null });
    await tick(3000);
    expect(await screen.findByText(/expired before a payment was detected/)).toBeInTheDocument();
    expect(screen.getByText(/Nothing was charged/)).toBeInTheDocument();
    expect(usePendingPaymentStore.getState().pending).toBeNull();

    http.get.mockResolvedValue(pendingStatus);
    await user.click(screen.getByRole("button", { name: /Generate QR to pay/ }));
    await screen.findByRole("img", { name: /QR Ph payment code/ });
    expect(http.post.mock.calls[1][2].headers["Idempotency-Key"]).not.toBe(firstKey);
  });

  it("resumes a persisted QR without creating a new order", async () => {
    usePendingPaymentStore.setState({ pending: pendingRecord() });
    renderModal();

    expect(await screen.findByRole("img", { name: /QR Ph payment code/ })).toBeInTheDocument();
    expect(http.post).not.toHaveBeenCalled();
    await waitFor(() => expect(http.get).toHaveBeenCalledWith(STATUS_PATH));
    expect(screen.getByText(/Ref 11111111/)).toBeInTheDocument();
  });

  it("picks the endpoint from the record's token, not from auth state", async () => {
    useAuthStore.setState({
      user: { id: "u1", email: "runner@example.com", name: "R", role: "RUNNER" } as never,
      isAuthenticated: true,
      isLoading: false,
    });
    usePendingPaymentStore.setState({ pending: pendingRecord() });
    const first = render(
      <QueryClientProvider client={new QueryClient()}>
        <CheckoutModal isOpen onClose={vi.fn()} />
      </QueryClientProvider>,
    );
    await waitFor(() => expect(http.get).toHaveBeenCalledWith(STATUS_PATH));
    first.unmount();

    vi.clearAllMocks();
    http.get.mockResolvedValue(pendingStatus);
    usePendingPaymentStore.setState({ pending: pendingRecord({ returnToken: null }) });
    renderModal();
    await waitFor(() => expect(http.get).toHaveBeenCalledWith(`/me/orders/${ORDER_ID}/status`));
  });
});
