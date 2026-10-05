import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import { formatOrderRef } from "@/lib/utils";
import type { PhotographerPurchase } from "@/lib/api-photographer-earnings";
import { PhotographerPurchasePreview } from "./photographer-purchase-preview";

const photo = { id: "photo-a", bib: "101", status: "live" as const, salesCount: 2, uploadedAt: "2026-10-01T00:00:00Z", tone: 0, span: "default" as const, thumbnailUrl: null };
const orderA = "aaaaaaaa-0000-0000-0000-000000000000";
const orderB = "bbbbbbbb-0000-0000-0000-000000000000";
const receipt = (orderId: string): PhotographerPurchase => ({ orderId, photoId: photo.id, buyer: "Aira S.", purchasedAt: "2026-10-01T16:30:00Z", orderCreatedAt: "2026-10-01T16:00:00Z", paymentMethod: "gcash", pricePhp: 150, discountPhp: 0, paidPhp: 150, kind: "paid", refundStatus: "none", refundPending: false, refundedPhp: null, refundedAt: null, netKeptPhp: 120, transactions: [] });
const response = (items: PhotographerPurchase[], selectedPurchase: PhotographerPurchase | null = null) => ({ photo, event: { id: "event-a", name: "Cebu Run", date: "2026-10-01", deletedAt: null }, items, total: items.length, offset: 0, limit: 25, selectedPurchase });
const login = () => useAuthStore.getState().setUser({ id: "seller-a", role: "PHOTOGRAPHER", name: "Seller", email: "seller@example.test", createdAt: "2026-09-10" });
function view(selection: { photoId: string; orderId?: string; transactionId?: string } | null, client: QueryClient) {
  return <QueryClientProvider client={client}><PhotographerPurchasePreview selection={selection} onClose={vi.fn()} /></QueryClientProvider>;
}
afterEach(() => { vi.restoreAllMocks(); useAuthStore.getState().logout(); });

it("same photo different order replaces context", async () => {
  login();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  vi.spyOn(api, "get").mockResolvedValueOnce(response([receipt(orderA)], receipt(orderA))).mockResolvedValueOnce(response([receipt(orderB)], receipt(orderB)));
  const rendered = render(view({ photoId: photo.id, orderId: orderA }, client));
  expect(await screen.findByText(formatOrderRef(orderA))).toBeVisible();
  rendered.rerender(view({ photoId: photo.id, orderId: orderB }, client));
  expect(await screen.findByText(formatOrderRef(orderB))).toBeVisible();
  expect(screen.queryByText(formatOrderRef(orderA))).not.toBeInTheDocument();
  client.clear();
});

it("missing image retains details", async () => {
  login();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  vi.spyOn(api, "get").mockResolvedValue(response([receipt(orderB)], receipt(orderB)));
  render(view({ photoId: photo.id, orderId: orderB }, client));
  expect(await screen.findByText("Aira S.")).toBeVisible();
  expect(screen.getByText(formatOrderRef(orderB))).toBeVisible();
  expect(screen.getByText("Paid for this photo")).toBeVisible();
  expect(screen.getByText("Preview unavailable")).toBeVisible();
  expect(screen.queryByRole("button", { name: /buy now/i })).not.toBeInTheDocument();
  client.clear();
});

it("highlights a selected refund without duplicating its purchase", async () => {
  login();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const refunded = { ...receipt(orderB), refundStatus: "partial", transactions: [
    { id: "sale", isRefund: false, refundOf: null, amountKept: 120, discountPhp: 0, paidAt: "2026-10-01T16:30:00Z", recordedAt: "2026-10-01T16:30:00Z" },
    { id: "refund", isRefund: true, refundOf: "sale", amountKept: -30, discountPhp: 0, paidAt: "2026-10-01T16:30:00Z", recordedAt: "2026-10-02T00:30:00Z" },
  ] };
  vi.spyOn(api, "get").mockResolvedValue(response([refunded], refunded));
  render(view({ photoId: photo.id, orderId: orderB, transactionId: "refund" }, client));
  expect(await screen.findByText(formatOrderRef(orderB))).toBeVisible();
  expect(screen.getByText(/Refund ·/)).toHaveAttribute("aria-current", "true");
  expect(screen.getAllByText(formatOrderRef(orderB))).toHaveLength(1);
  client.clear();
});

it("does not turn unavailable historical money into zero", async () => {
  login();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const unpriced = { ...receipt(orderB), netKeptPhp: null };
  vi.spyOn(api, "get").mockResolvedValue(response([unpriced], unpriced));
  render(view({ photoId: photo.id, orderId: orderB }, client));
  expect(await screen.findByText("Paid ₱150.00 · Kept —")).toBeVisible();
  client.clear();
});

it("does not fetch after close or account switch", async () => {
  login();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  let resolve!: (value: ReturnType<typeof response>) => void;
  const get = vi.spyOn(api, "get").mockImplementation(() => new Promise((r) => { resolve = r as typeof resolve; }));
  const rendered = render(view({ photoId: photo.id, orderId: orderB }, client));
  await waitFor(() => expect(get).toHaveBeenCalledTimes(1));
  const historyRequest = get.mock.calls[0][1]!;
  rendered.rerender(view(null, client));
  expect(historyRequest.signal?.aborted).toBe(true);
  act(() => useAuthStore.getState().logout());
  await act(async () => resolve(response([receipt(orderB)], receipt(orderB))));
  expect(screen.queryByText("Aira S.")).not.toBeInTheDocument();
  expect(get).toHaveBeenCalledTimes(1);
  client.clear();
});
