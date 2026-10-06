import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import type { PhotographerTransaction } from "@/lib/photographer-mock";
import { useAuthStore } from "@/store/auth-store";
import BillingPage from "./page";

const orderId = "aaaaaaaa-0000-0000-0000-000000000000";
const sale: PhotographerTransaction = {
  id: "sale-1", orderId, photoId: "photo-1", eventId: "event-1", eventName: "Cebu Run",
  buyer: "Aira S.", paidAt: "2026-09-30T16:30:00Z", recordedAt: "2026-09-30T16:30:00Z",
  amountKept: 100.5, thumbnailUrl: "https://cdn.example.test/photo.jpg",
};
const refund: PhotographerTransaction = {
  ...sale, id: "refund-1", isRefund: true, refundOf: sale.id,
  recordedAt: "2026-10-01T00:30:00Z", amountKept: -25.5,
};
const balance = {
  unpaidBalance: 1000, clearing: 0, reserved: 0, minimum: 500,
  hasOpenRequest: false, openRequest: null, accountCooldownUntil: null,
  primaryAccount: { method: "gcash", accountNumber: "09170001111", accountName: "Seller", hasQr: true },
};
const photo = { id: "photo-1", bib: "101", status: "live", salesCount: 1, uploadedAt: sale.paidAt, tone: 0, span: "default", thumbnailUrl: sale.thumbnailUrl };
const purchase = {
  orderId, photoId: sale.photoId, buyer: "Aira S.", purchasedAt: sale.paidAt, orderCreatedAt: sale.paidAt,
  paymentMethod: "gcash", pricePhp: 150, discountPhp: 0, paidPhp: 150, kind: "paid", refundStatus: "partial",
  refundPending: false, refundedPhp: 25.5, refundedAt: refund.recordedAt, netKeptPhp: 75,
  transactions: [
    { id: sale.id, isRefund: false, refundOf: null, amountKept: sale.amountKept, discountPhp: 0, paidAt: sale.paidAt, recordedAt: sale.recordedAt },
    { id: refund.id, isRefund: true, refundOf: sale.id, amountKept: refund.amountKept, discountPhp: 0, paidAt: sale.paidAt, recordedAt: refund.recordedAt },
  ],
};

let client: QueryClient;
function setup(ledger: (offset: number) => Promise<unknown>) {
  useAuthStore.getState().setUser({ id: "seller-1", role: "PHOTOGRAPHER", name: "Seller", email: "seller@example.test", createdAt: "2026-09-01" });
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const get = vi.spyOn(api, "get").mockImplementation((path) => {
    if (path.includes("/billing/transactions")) return ledger(Number(new URL(path, "https://test.invalid").searchParams.get("offset")));
    if (path.includes("/photos/photo-1/purchases")) return Promise.resolve({ items: [purchase], total: 1, offset: 0, limit: 25, photo, event: { id: "event-1", name: "Cebu Run", date: "2026-10-01", deletedAt: null }, selectedPurchase: purchase });
    if (path.includes("/payouts/balance")) return Promise.resolve(balance);
    return Promise.resolve({ items: [], total: 0, offset: 0, limit: 8 });
  });
  render(<QueryClientProvider client={client}><BillingPage /></QueryClientProvider>);
  return get;
}
const page = (items: PhotographerTransaction[], offset = 0, total = items.length) => ({ items, offset, total, limit: 25, monthTotals: { "2026-10": 75 } });

afterEach(() => { client?.clear(); vi.restoreAllMocks(); vi.unstubAllEnvs(); useAuthStore.getState().logout(); });

it("sale and refund rows open the exact purchase and restore row focus", async () => {
  const get = setup(async () => page([sale, refund]));
  const saleAction = await screen.findByRole("button", { name: /sale.*aira s\..*aaaa/i });
  saleAction.focus();
  await userEvent.keyboard("{Enter}");
  expect(await screen.findByRole("dialog")).toBeVisible();
  expect(get.mock.calls.some(([path]) => path.includes(`orderId=${orderId}`))).toBe(true);
  fireEvent.click(screen.getAllByRole("button", { name: /close/i }).at(-1)!);
  await waitFor(() => expect(saleAction).toHaveFocus());
  fireEvent.click(screen.getByRole("button", { name: /refund.*aira s\..*aaaa/i }));
  expect(await screen.findByText(/Earnings adjustment/)).toHaveAttribute("aria-current", "true");
});

it("does not reopen a seller's purchase after the auth session changes", async () => {
  let resolveSellerB!: (value: ReturnType<typeof page>) => void;
  const sellerBLedger = new Promise<ReturnType<typeof page>>((resolve) => { resolveSellerB = resolve; });
  const get = setup(async () => useAuthStore.getState().user?.id === "seller-1" ? page([sale]) : sellerBLedger);
  fireEvent.click(await screen.findByRole("button", { name: /sale.*aira s\..*aaaa/i }));
  expect(await screen.findByRole("dialog")).toBeVisible();
  await waitFor(() => expect(get.mock.calls.filter(([path]) => path.includes("/photos/photo-1/purchases"))).toHaveLength(1));

  act(() => useAuthStore.getState().setUser({ id: "seller-2", role: "PHOTOGRAPHER", name: "Other Seller", email: "other@example.test", createdAt: "2026-09-01" }));
  await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  await act(async () => resolveSellerB(page([])));
  expect(await screen.findByText(/zero-charge purchases.*Purchased photos/i)).toBeVisible();
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(get.mock.calls.filter(([path]) => path.includes("/photos/photo-1/purchases"))).toHaveLength(1);
});

it("uses the Manila month and full server total even in a UTC browser", async () => {
  vi.stubEnv("TZ", "UTC");
  setup(async () => ({ ...page([sale], 0, 50), monthTotals: { "2026-10": 999.75 } }));
  expect(await screen.findByText(/OCT 2026.*999\.75/i)).toBeVisible();
});

it("renders one sign and two decimals on refund money", async () => {
  setup(async () => page([sale, refund]));
  const row = await screen.findByRole("button", { name: /refund.*aira s\..*aaaa/i });
  expect(within(row).getByText(/-.*25\.50/)).toBeVisible();
  expect(within(row).queryByText(/\+.*-/)).not.toBeInTheDocument();
  expect(within(row).getByText("Cebu Run")).toBeVisible();
});

it("keeps loaded rows and retries an append failure", async () => {
  let attempts = 0;
  setup(async (offset) => {
    if (offset === 0) return page([sale], 0, 50);
    if (++attempts === 1) throw new Error("Ledger unavailable");
    return page([refund], 25, 50);
  });
  await screen.findByRole("button", { name: /sale.*aira s\..*aaaa/i });
  fireEvent.click(screen.getByRole("button", { name: /load 25 more/i }));
  expect(await screen.findByRole("alert", { name: /transactions/i })).toBeVisible();
  expect(screen.getByRole("button", { name: /sale.*aira s\..*aaaa/i })).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: /retry.*transactions/i }));
  expect(await screen.findByRole("button", { name: /refund.*aira s\..*aaaa/i })).toBeVisible();
});

it("keeps payout controls usable on a ledger failure", async () => {
  setup(async () => { throw new Error("Ledger unavailable"); });
  expect(await screen.findByRole("alert", { name: /transactions/i })).toBeVisible();
  expect(await screen.findByRole("button", { name: "Request payout" })).toBeEnabled();
});

it("shows an empty ledger with the Purchased photos path", async () => {
  setup(async () => page([]));
  expect(await screen.findByText(/zero-charge purchases.*Purchased photos/i)).toBeVisible();
});

it("keeps a legacy row readable without opening fabricated purchase data", async () => {
  setup(async () => page([{ ...sale, orderId: undefined, thumbnailUrl: null }]));
  expect(await screen.findByText("Cebu Run")).toBeVisible();
  expect(screen.getByRole("img", { name: "Preview unavailable" })).toBeVisible();
  expect(screen.queryByRole("button", { name: /sale.*aira s/i })).not.toBeInTheDocument();
});

it("falls back after a thumbnail error and shows a fresh URL on refetch", async () => {
  const fresh = "https://cdn.example.test/fresh.jpg";
  let reads = 0;
  setup(async () => page([
    { ...sale, thumbnailUrl: ++reads === 1 ? sale.thumbnailUrl : fresh },
    refund,
  ]));
  const saleRow = await screen.findByRole("button", { name: /sale.*aira s\..*aaaa/i });
  const refundRow = screen.getByRole("button", { name: /refund.*aira s\..*aaaa/i });
  expect(saleRow.querySelector("img")).toHaveAttribute("src", sale.thumbnailUrl);
  fireEvent.error(saleRow.querySelector("img")!);
  expect(within(saleRow).getByRole("img", { name: "Preview unavailable" })).toBeVisible();
  expect(refundRow.querySelector("img")).toHaveAttribute("src", refund.thumbnailUrl);
  await act(async () => { await client.invalidateQueries({ queryKey: ["photographer", "transactions"] }); });
  await waitFor(() => expect(saleRow.querySelector("img")).toHaveAttribute("src", fresh));
});

it("stops pagination when a stale total returns an empty page", async () => {
  const get = setup(async (offset) => offset === 0 ? page([sale], 0, 100) : page([], 25, 100));
  await screen.findByRole("button", { name: /sale.*aira s\..*aaaa/i });
  fireEvent.click(screen.getByRole("button", { name: /load 25 more/i }));
  await waitFor(() => expect(get.mock.calls.filter(([path]) => path.includes("/billing/transactions"))).toHaveLength(2));
  expect(screen.getByRole("button", { name: /sale.*aira s\..*aaaa/i })).toBeVisible();
  await waitFor(() => expect(screen.queryByRole("button", { name: /load .*more/i })).not.toBeInTheDocument());
  expect(get.mock.calls.filter(([path]) => path.includes("/billing/transactions"))).toHaveLength(2);
});
