import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
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
// The pay step quotes the cart (POST /coupons/preview) before any order is
// created, so order-shaped fallbacks are routed by path and order calls are
// picked out by path rather than by call index.
const EMPTY_QUOTE = {
  code: null,
  percentOff: null,
  photographerName: null,
  photographerHandle: null,
  items: [],
  eligibleCount: 0,
  discountTotal: 0,
};
const postRoutes = (fallback: unknown) =>
  http.post.mockImplementation(async (path: string) =>
    path === "/coupons/preview" ? EMPTY_QUOTE : fallback,
  );
const orderCalls = () => http.post.mock.calls.filter(([p]) => p === "/orders");
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
    testUrl: null as string | null,
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
    postRoutes({
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
    expect(screen.getByText(/Step 1 of 4/)).toBeInTheDocument();
    await user.type(screen.getByLabelText("Email"), "runner@example.com");
    await user.type(screen.getByLabelText("Confirm email"), "runner@example.com");
    await user.click(screen.getByRole("button", { name: "Continue →" }));

    expect(screen.getByText(/Step 2 of 4/)).toBeInTheDocument();
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

  // Auto-apply (2026-09-05): the pay step asks the server how the cart is
  // priced — no code typed — and the order goes out without one.
  it("quotes photographer discounts automatically on the pay step", async () => {
    const order = {
      id: ORDER_ID,
      status: "PENDING",
      items: [],
      totalAmount: 106.25,
      paymentMethod: "qrph",
      createdAt: "2026-09-04T12:00:00Z",
      qrPh: { imageUrl: "data:image/png;base64,cXJwaA==", expiresAt: expiresAt(), returnToken: "return-token" },
    };
    http.post.mockImplementation(async (path: string) =>
      path === "/coupons/preview"
        ? {
            code: null,
            percentOff: null,
            photographerName: null,
            photographerHandle: null,
            items: [{ photoId: "photo-1", price: 125, discount: 18.75, couponCode: "AAAA", percentOff: 20 }],
            eligibleCount: 1,
            discountTotal: 18.75,
          }
        : order,
    );
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    renderModal();
    await user.type(screen.getByLabelText("Email"), "runner@example.com");
    await user.type(screen.getByLabelText("Confirm email"), "runner@example.com");
    await user.click(screen.getByRole("button", { name: "Continue →" }));

    await waitFor(() =>
      expect(http.post).toHaveBeenCalledWith("/coupons/preview", { photoIds: ["photo-1"] }),
    );
    expect(await screen.findByText(/Photographer discounts applied/)).toBeInTheDocument();
    expect(screen.getByText("AAAA · −₱18.75")).toBeInTheDocument();
    expect(screen.getAllByText("₱106.25").length).toBeGreaterThan(0);
    // The typed-code path is still there for a private code.
    expect(screen.getByLabelText(/Have a code/)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Generate QR to pay/ }));
    await screen.findByRole("img", { name: /QR Ph payment code/ });
    const orderCall = http.post.mock.calls.find(([path]) => path === "/orders");
    expect(orderCall?.[1]).not.toHaveProperty("couponCode");
    expect(usePendingPaymentStore.getState().pending).toMatchObject({ total: 106.25 });
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

  it("blocks a mismatched Confirm email and accepts a case-insensitive match", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    renderModal();
    await user.type(screen.getByLabelText("Email"), "runner@example.com");
    await user.type(screen.getByLabelText("Confirm email"), "runner@exmaple.com");
    await user.click(screen.getByRole("button", { name: "Continue →" }));
    expect(screen.getByText("Emails don't match.")).toBeInTheDocument();
    expect(screen.queryByText("QR Ph")).not.toBeInTheDocument();

    // The label now also carries the error text, so match its start.
    const confirmField = screen.getByLabelText(/^Confirm email/);
    await user.clear(confirmField);
    await user.type(confirmField, "Runner@Example.com");
    await user.click(screen.getByRole("button", { name: "Continue →" }));
    expect(screen.getByText("QR Ph")).toBeInTheDocument();
    expect(screen.getByText("runner@example.com")).toBeInTheDocument();
  });

  // EXPIRED well before the QR's own deadline can only mean the bank or the
  // simulator failed the payment — the backend records both the same way.
  it("reads an early EXPIRED as a failed payment and retries with a fresh key", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    renderModal();
    await walkToQr(user);
    const firstKey = orderCalls()[0][2].headers["Idempotency-Key"];

    http.get.mockResolvedValue({ id: ORDER_ID, status: "EXPIRED", paidAt: null });
    await tick(3000);
    expect(await screen.findByText("Payment didn't go through")).toBeInTheDocument();
    expect(screen.getByText(/Nothing was charged/)).toBeInTheDocument();
    expect(usePendingPaymentStore.getState().pending).toBeNull();

    http.get.mockResolvedValue(pendingStatus);
    await user.click(screen.getByRole("button", { name: /Generate a new QR/ }));
    await screen.findByRole("img", { name: /QR Ph payment code/ });
    expect(orderCalls()[1][2].headers["Idempotency-Key"]).not.toBe(firstKey);
  });

  it("reads EXPIRED at the QR's deadline as an expired code", async () => {
    usePendingPaymentStore.setState({
      pending: pendingRecord({ expiresAt: new Date(Date.now() - 60_000).toISOString() }),
    });
    http.get.mockResolvedValue({ id: ORDER_ID, status: "EXPIRED", paidAt: null });
    renderModal();

    expect(await screen.findByText("Your QR code expired")).toBeInTheDocument();
    expect(usePendingPaymentStore.getState().pending).toBeNull();
  });

  it("cancels a live QR after confirmation and lands back on the pay step", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    renderModal();
    await walkToQr(user);
    const firstKey = orderCalls()[0][2].headers["Idempotency-Key"];
    postRoutes({ id: ORDER_ID, status: "EXPIRED", paidAt: null });

    await user.click(screen.getByRole("button", { name: "Cancel payment" }));
    await waitFor(() => expect(useConfirmationStore.getState().active).not.toBeNull());
    expect(useConfirmationStore.getState().active?.config.title).toBe("Cancel this payment?");
    // Keep waiting: nothing happens.
    act(() => useConfirmationStore.getState().close(false));
    await waitFor(() => expect(useConfirmationStore.getState().active).toBeNull());
    expect(orderCalls()).toHaveLength(1);
    expect(screen.getByRole("img", { name: /QR Ph payment code/ })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Cancel payment" }));
    await waitFor(() => expect(useConfirmationStore.getState().active).not.toBeNull());
    act(() => useConfirmationStore.getState().close(true));
    await waitFor(() =>
      expect(http.post).toHaveBeenCalledWith(`/orders/${ORDER_ID}/cancel?token=return-token`),
    );
    expect(await screen.findByText("Payment cancelled")).toBeInTheDocument();
    expect(screen.getByText(/Step 2 of 4/)).toBeInTheDocument();
    expect(usePendingPaymentStore.getState().pending).toBeNull();
    // The cart is untouched — only the payment attempt was cancelled.
    expect(useCartStore.getState().items).toHaveLength(1);

    postRoutes({
      id: ORDER_ID,
      status: "PENDING",
      items: [],
      totalAmount: 125,
      paymentMethod: "qrph",
      createdAt: "2026-09-04T12:00:00Z",
      qrPh: { imageUrl: "data:image/png;base64,cXJwaA==", expiresAt: expiresAt(), returnToken: "return-token" },
    });
    await user.click(screen.getByRole("button", { name: /Generate a new QR/ }));
    await screen.findByRole("img", { name: /QR Ph payment code/ });
    const keys = http.post.mock.calls.filter(([p]) => p === "/orders").map((c) => c[2].headers["Idempotency-Key"]);
    expect(keys[1]).not.toBe(firstKey);
  });

  it("shows success when the payment wins the race against cancel", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    renderModal();
    await walkToQr(user);
    postRoutes(fulfilledStatus);

    await user.click(screen.getByRole("button", { name: "Cancel payment" }));
    await waitFor(() => expect(useConfirmationStore.getState().active).not.toBeNull());
    act(() => useConfirmationStore.getState().close(true));

    expect(await screen.findByText("All yours.")).toBeInTheDocument();
    expect(screen.queryByText("Payment cancelled")).not.toBeInTheDocument();
    expect(useCartStore.getState().items).toHaveLength(0);
    expect(usePendingPaymentStore.getState().pending).toBeNull();
  });

  it("cancels a signed-in record through the /me route", async () => {
    usePendingPaymentStore.setState({ pending: pendingRecord({ returnToken: null }) });
    postRoutes({ id: ORDER_ID, status: "EXPIRED", paidAt: null });
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    renderModal();
    await screen.findByRole("img", { name: /QR Ph payment code/ });

    await user.click(screen.getByRole("button", { name: "Cancel payment" }));
    await waitFor(() => expect(useConfirmationStore.getState().active).not.toBeNull());
    act(() => useConfirmationStore.getState().close(true));

    await waitFor(() => expect(http.post).toHaveBeenCalledWith(`/me/orders/${ORDER_ID}/cancel`));
    expect(await screen.findByText("Payment cancelled")).toBeInTheDocument();
  });

  it("shows the PayMongo simulator only when the backend sends a test url", async () => {
    usePendingPaymentStore.setState({
      pending: pendingRecord({ testUrl: "https://test.paymongo.com/qrph/pi_x" }),
    });
    const first = renderModal();
    expect(await screen.findByText(/Test mode/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /Open PayMongo simulator/ })).toHaveAttribute(
      "href",
      "https://test.paymongo.com/qrph/pi_x",
    );
    void first;
    cleanup();

    usePendingPaymentStore.setState({ pending: pendingRecord() });
    renderModal();
    await screen.findByRole("img", { name: /QR Ph payment code/ });
    expect(screen.queryByText(/Test mode/)).not.toBeInTheDocument();
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
    expect(screen.getByText(/Step 2 of 3/)).toBeInTheDocument();
    first.unmount();

    vi.clearAllMocks();
    http.get.mockResolvedValue(pendingStatus);
    usePendingPaymentStore.setState({ pending: pendingRecord({ returnToken: null }) });
    renderModal();
    await waitFor(() => expect(http.get).toHaveBeenCalledWith(`/me/orders/${ORDER_ID}/status`));
  });
});
