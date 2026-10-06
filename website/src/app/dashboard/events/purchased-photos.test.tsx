import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import EventPage from "./[id]/page";
import { ApiError } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";

const mocks = vi.hoisted(() => ({
  search: "?tab=purchased", detail: null as unknown, error: null as unknown, missing: false,
  refetch: vi.fn(), active: vi.fn(), visibility: vi.fn(), purchased: vi.fn(),
}));
vi.mock("next/navigation", () => ({
  useParams: () => ({ id: "event-1" }),
  useSearchParams: () => new URLSearchParams(mocks.search),
  useRouter: () => ({ replace: vi.fn() }),
  notFound: () => { throw Error("404"); },
}));
vi.mock("@tanstack/react-query", () => ({ useQueryClient: () => ({ invalidateQueries: vi.fn() }) }));
vi.mock("@/hooks/use-toast", () => ({ useToast: () => ({ showToast: vi.fn() }) }));
vi.mock("@/hooks/use-confirmation", () => ({ useConfirmation: () => ({ confirm: async () => true }) }));
vi.mock("@/components/layout/site-header", () => ({ SiteHeader: () => null }));
vi.mock("@/components/dashboard/event-coupon-modal", () => ({ EventCouponModal: () => null }));
vi.mock("@/components/dashboard/my-event-form", () => ({ MyEventForm: () => null }));
vi.mock("@/components/photos/photo-preview-card", () => ({ PhotoPreviewCard: ({ details }: { details?: React.ReactNode }) => <div role="dialog">{details}</div> }));
vi.mock("@/hooks/use-photographer-data", () => ({
  usePhotographerEventDetail: () => ({ detail: mocks.detail, error: mocks.error, isMissing: mocks.missing, isLoading: !mocks.detail && !mocks.error, refetch: mocks.refetch }),
  usePhotographerEventPhotos: mocks.active,
  usePhotographerEventMetadataVisibility: (...args: unknown[]) => {
    mocks.visibility(...args);
    return { isSuccess: false, isPending: true, isError: false, data: undefined, refetch: vi.fn() };
  },
}));
vi.mock("@/hooks/use-photographer-purchases", () => ({
  usePhotographerPurchasedPhotos: mocks.purchased,
  usePhotographerPhotoPurchases: () => ({
    photo: { id: "photo-1", bib: null, tone: 0, span: "default", thumbnailUrl: null },
    event: { id: "event-1", name: "Old Race", date: "2026-09-07", deletedAt: "2026-09-08" },
    items: [{ orderId: "abcdef12-3456", photoId: "photo-1", buyer: "M. Santos", purchasedAt: "2026-09-07T10:00:00Z", orderCreatedAt: "2026-09-07T09:00:00Z", paymentMethod: "gcash", pricePhp: 0, discountPhp: 0, paidPhp: 0, kind: "free", refundStatus: "none", refundPending: false, refundedPhp: 0, refundedAt: null, netKeptPhp: 0, transactions: [] }],
    selectedPurchase: null, total: 1, isLoading: false, error: null, appendError: null, hasNextPage: false, refetch: vi.fn(), fetchNextPage: vi.fn(),
  }),
}));

const detail = { id: "event-1", slug: "old-race", name: "Old Race", date: "2026-09-07", location: "Cebu", state: "past", photoCount: 2, salesCount: 3, revenueKept: 0, pricingMode: "paid", ownedByMe: true, deletedAt: "2026-09-08" };
const photo = { id: "photo-1", bib: null, status: "live", salesCount: 3, uploadedAt: "2026-09-07T00:00:00Z", tone: 0, span: "default", thumbnailUrl: null };

afterEach(() => { cleanup(); vi.clearAllMocks(); useAuthStore.getState().logout(); });

it("removed event renders read-only purchase history", () => {
  useAuthStore.getState().setUser({ id: "seller", role: "PHOTOGRAPHER", name: "Seller", email: "seller@example.test", createdAt: "2026-09-01" });
  mocks.detail = detail;
  mocks.error = null;
  mocks.purchased.mockReturnValue({ items: [{ photo, purchaseCount: 3, lastPurchasedAt: "2026-09-07T10:00:00Z" }], total: 1, isLoading: false, error: null, appendError: null, hasNextPage: false, fetchNextPage: vi.fn(), refetch: vi.fn() });
  render(<EventPage />);
  expect(screen.getByText("Removed event")).toBeInTheDocument();
  expect(screen.getByText("₱0.00")).toBeInTheDocument();
  expect(screen.queryByRole("link", { name: /Upload/ })).toBeNull();
  expect(screen.getByText("3 purchases")).toBeInTheDocument();
  expect(screen.getAllByRole("button", { name: /Open photo/ })).toHaveLength(1);
  expect(screen.getByRole("button", { name: "Open photo 1, 3 purchases" })).toBeInTheDocument();
  expect(mocks.active).not.toHaveBeenCalled();
  expect(mocks.visibility).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: /Open photo/ }));
  expect(screen.getByText("ABCDEF12")).toBeInTheDocument();
});

it("event error is retryable rather than notFound, while a genuine 404 stays missing", () => {
  mocks.detail = null;
  mocks.error = new ApiError([{ code: "TIMEOUT", message: "Timed out" }], 503);
  mocks.missing = false;
  const { rerender } = render(<EventPage />);
  fireEvent.click(screen.getByRole("button", { name: "Retry" }));
  expect(mocks.refetch).toHaveBeenCalledOnce();
  mocks.error = new ApiError([{ code: "EVENT_NOT_FOUND", message: "Missing" }], 404);
  mocks.missing = true;
  expect(() => rerender(<EventPage />)).toThrow("404");
});

it("active and deleted tabs retain management and bare URLs default to active", () => {
  mocks.detail = { ...detail, deletedAt: null, ownedByMe: false, pricingMode: "free" };
  mocks.error = null;
  mocks.missing = false;
  mocks.search = "";
  mocks.active.mockImplementation(() => ({ items: [photo], total: 1, isLoading: false, isFetchingNextPage: false, fetchNextPage: vi.fn() }));
  mocks.visibility.mockReturnValue({ isSuccess: true, data: { metadataPublic: false } });
  render(<EventPage />);
  expect(screen.getByRole("tab", { name: "Active" })).toHaveAttribute("aria-selected", "true");
  expect(screen.getByRole("button", { name: "Select" })).toBeInTheDocument();
  fireEvent.click(screen.getByRole("tab", { name: "Deleted" }));
  expect(screen.getByRole("tab", { name: "Deleted" })).toHaveAttribute("aria-selected", "true");
  expect(screen.getByRole("button", { name: "Select" })).toBeInTheDocument();
});
