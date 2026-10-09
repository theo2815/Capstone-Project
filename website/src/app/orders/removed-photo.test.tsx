import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import * as ordersApi from "@/lib/api-orders";
import OrdersPage from "@/app/orders/page";
import OrdersReturnPage from "@/app/orders/return/page";
import type { OrderPhotoDetail } from "@/types/order";

const receiptReference = "order-removed-1";
const keptPhoto: OrderPhotoDetail = { id: "kept-photo", bib: "101", time: "07:00", tone: 1, thumbnailUrl: "https://photos.test/kept.jpg",
  previewUrl: "https://photos.test/kept-preview.jpg", downloadUrl: "https://photos.test/kept-download.jpg", availability: "AVAILABLE", unavailableReason: null };
// A stale payload may still carry URLs; REMOVED must win over them.
const removedPhoto = (id: string): OrderPhotoDetail => ({ id, bib: "102", time: "07:01", tone: 2, thumbnailUrl: `https://photos.test/${id}.jpg`,
  previewUrl: `https://photos.test/${id}-preview.jpg`, downloadUrl: `https://photos.test/${id}-download.jpg`, availability: "REMOVED", unavailableReason: "PHOTO_REMOVED" });
const summary = { id: receiptReference, eventId: "event-1", photoIds: ["kept-photo", "removed-photo"], total: 160, paymentMethod: "qrph",
  paidAt: "2026-09-28T00:00:00Z", eventName: "Cebu Run", status: "PAID" as const, couponCode: "RUN20", discountTotal: 40 };
const mixedDetail = { ...summary, photos: [keptPhoto, removedPhoto("removed-photo")], shareToken: "share-1" };
const allRemovedDetail = { ...summary, photoIds: ["removed-photo", "removed-two"], photos: [removedPhoto("removed-photo"), removedPhoto("removed-two")], shareToken: "share-1" };

vi.mock("next/navigation", () => ({
  usePathname: () => "/orders", useSearchParams: () => new URLSearchParams(`orderId=${"order-removed-1"}&token=guest-token`),
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
}));
vi.mock("@/components/layout/site-header", () => ({ SiteHeader: () => null }));
vi.mock("@/components/profile-shell", () => ({
  IdentityRail: () => null, Slab: ({ children }: { children: React.ReactNode }) => <section>{children}</section>,
}));
vi.mock("@/components/orders/refund-modal", () => ({ RefundModal: () => null }));

const runner = { id: "runner-1", role: "RUNNER" as const, name: "Runner", email: "runner@example.test", createdAt: "2026-01-01T00:00:00Z" };
afterEach(() => { vi.restoreAllMocks(); useAuthStore.getState().logout(); });

// Network boundary: answers the order-history, order-detail and guest receipt reads.
function serve(detail: typeof mixedDetail) {
  vi.spyOn(api, "get").mockImplementation(async (path: string) => {
    if (path.startsWith("/me/orders?")) return { items: [{ ...summary, photoIds: detail.photoIds }], total: 1, offset: 0, limit: 200 };
    if (path.includes("/status?")) return { id: receiptReference, status: "PAID", paidAt: summary.paidAt };
    return detail;
  });
}
function mount(ui: React.ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}
async function openOrders(detail: typeof mixedDetail) {
  useAuthStore.getState().startSession(runner);
  serve(detail);
  const user = userEvent.setup();
  mount(<OrdersPage />);
  await user.click(await screen.findByRole("button", { name: "View photos" }));
  return user;
}

it("removedOrderStillShowsFinancialHistory", async () => {
  const user = await openOrders(mixedDetail);
  expect(await screen.findByText("Photo removed")).toBeVisible();
  expect(screen.getByText(receiptReference)).toBeVisible();
  expect(screen.getAllByText("₱160")[0]).toBeVisible();
  expect(screen.getByText(/RUN20/)).toBeVisible();
  expect(screen.getByRole("button", { name: "Request a refund" })).toBeVisible();
  expect(screen.queryByRole("button", { name: "Preview removed-photo" })).not.toBeInTheDocument();
  expect(document.querySelector('img[src*="removed"]')).toBeNull();
  expect(screen.queryByRole("link", { name: "Download removed photo" })).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Save photo" })).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Preview kept-photo" }));
  expect(await screen.findByRole("button", { name: "Download photo" })).toBeInTheDocument();
  // The lightbox only steps through photos that still exist.
  expect(screen.queryByRole("button", { name: "Next photo" })).not.toBeInTheDocument();
});

it("offers no save control on the order history when every photo was removed", async () => {
  await openOrders(allRemovedDetail);
  expect(await screen.findAllByText("Photo removed")).toHaveLength(2);
  expect(screen.queryByRole("button", { name: /^Save/ })).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Request a refund" })).toBeVisible();
});

it("shows a removed photo placeholder on the receipt page and keeps the kept photo's own save", async () => {
  serve(mixedDetail);
  mount(<OrdersReturnPage />);
  // The receipt animates in from opacity 0, so check presence rather than visibility.
  expect(await screen.findByText("Photo removed")).toBeInTheDocument();
  expect(screen.getByText(receiptReference.slice(0, 8).toUpperCase())).toBeInTheDocument();
  expect(screen.queryByText(/refund options/i)).not.toBeInTheDocument();
  expect(screen.getAllByRole("link", { name: /Download photo/ })).toHaveLength(1);
  expect(screen.getByRole("link", { name: /Download photo/ })).toHaveClass("bg-fresh");
  expect(screen.queryByRole("button", { name: /^Save/ })).not.toBeInTheDocument();
  expect(document.querySelector('img[src*="removed"]')).toBeNull();
});

it("offers no save control on the receipt page when every photo was removed", async () => {
  serve(allRemovedDetail);
  mount(<OrdersReturnPage />);
  expect(await screen.findAllByText("Photo removed")).toHaveLength(2);
  expect(screen.queryByRole("button", { name: /^Save/ })).not.toBeInTheDocument();
  expect(screen.queryByRole("link", { name: /Download/ })).not.toBeInTheDocument();
});

it("treats order photos from older payloads as available", async () => {
  const legacy: Partial<OrderPhotoDetail> = { ...keptPhoto };
  delete legacy.availability; delete legacy.unavailableReason;
  vi.spyOn(api, "get").mockResolvedValue({ ...mixedDetail, photos: [legacy] });
  const read = await ordersApi.fetchOrderDetail(receiptReference);
  expect(read?.photos[0]).toMatchObject({ availability: "AVAILABLE", unavailableReason: null });
});
