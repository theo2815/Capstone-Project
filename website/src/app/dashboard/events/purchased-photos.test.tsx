import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { onlineManager, QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, expect, it, vi } from "vitest";
import EventPage from "./[id]/page";
import { api, ApiError } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";

const mocks = vi.hoisted(() => ({
  search: "?tab=purchased", detail: null as unknown, error: null as unknown, missing: false, realReads: false,
  refetch: vi.fn(), active: vi.fn(), visibility: vi.fn(), purchased: vi.fn(),
  replace: vi.fn(), id: "event-1",
  listeners: new Set<() => void>(),
}));
vi.mock("next/navigation", async () => {
  const { useSyncExternalStore } = await import("react");
  return { useParams: () => ({ id: mocks.id }),
    useSearchParams: () => new URLSearchParams(useSyncExternalStore((onChange) => { mocks.listeners.add(onChange); return () => { mocks.listeners.delete(onChange); }; }, () => mocks.search)),
    useRouter: () => ({ replace: (url: string, options: { scroll: boolean }) => { mocks.replace(url, options); mocks.search = new URL(url, "http://test").search; mocks.listeners.forEach((notify) => notify()); } }),
    notFound: () => { throw Error("404"); } };
});
vi.mock("@/hooks/use-toast", () => ({ useToast: () => ({ showToast: vi.fn() }) }));
vi.mock("@/hooks/use-confirmation", () => ({ useConfirmation: () => ({ confirm: async () => true }) }));
vi.mock("@/components/layout/site-header", () => ({ SiteHeader: () => null }));
vi.mock("@/components/dashboard/event-coupon-modal", () => ({ EventCouponModal: () => null }));
vi.mock("@/components/dashboard/my-event-form", () => ({ MyEventForm: () => null }));
vi.mock("@/components/photos/photo-preview-card", () => ({ PhotoPreviewCard: ({ details }: { details?: React.ReactNode }) => <div role="dialog">{details}</div> }));
vi.mock("@/hooks/use-photographer-data", async (original) => {
  const real = await original<typeof import("@/hooks/use-photographer-data")>();
  return { ...real,
  usePhotographerEventDetail: (...args: Parameters<typeof real.usePhotographerEventDetail>) => mocks.realReads ? real.usePhotographerEventDetail(...args) : ({ detail: mocks.detail, error: mocks.error, isMissing: mocks.missing, isLoading: !mocks.detail && !mocks.error, refetch: mocks.refetch }),
  usePhotographerEventPhotos: mocks.active,
  usePhotographerEventMetadataVisibility: (...args: unknown[]) => {
    mocks.visibility(...args);
    return { isSuccess: false, isPending: true, isError: false, data: undefined, refetch: vi.fn() };
  },
}; });
vi.mock("@/hooks/use-photographer-purchases", async (original) => {
  const real = await original<typeof import("@/hooks/use-photographer-purchases")>();
  return { ...real,
  usePhotographerPurchasedPhotos: (...args: Parameters<typeof real.usePhotographerPurchasedPhotos>) => mocks.realReads ? real.usePhotographerPurchasedPhotos(...args) : mocks.purchased(...args),
  usePhotographerPhotoPurchases: () => ({
    photo: { id: "photo-1", bib: null, tone: 0, span: "default", thumbnailUrl: null },
    event: { id: "event-1", name: "Old Race", date: "2026-09-07", deletedAt: "2026-09-08" },
    items: [{ orderId: "abcdef12-3456", photoId: "photo-1", buyer: "M. Santos", purchasedAt: "2026-09-07T10:00:00Z", orderCreatedAt: "2026-09-07T09:00:00Z", paymentMethod: "gcash", pricePhp: 0, discountPhp: 0, paidPhp: 0, kind: "free", refundStatus: "none", refundPending: false, refundedPhp: 0, refundedAt: null, netKeptPhp: 0, transactions: [] }],
    selectedPurchase: null, total: 1, isLoading: false, error: null, appendError: null, hasNextPage: false, refetch: vi.fn(), fetchNextPage: vi.fn(),
  }),
}; });

const detail = { id: "event-1", slug: "old-race", name: "Old Race", date: "2026-09-07", location: "Cebu", state: "past", photoCount: 2, salesCount: 3, revenueKept: 0, pricingMode: "paid", ownedByMe: true, deletedAt: "2026-09-08" };
const photo = { id: "photo-1", bib: null, status: "live", salesCount: 3, uploadedAt: "2026-09-07T00:00:00Z", tone: 0, span: "default", thumbnailUrl: null };

afterEach(() => { cleanup(); client.clear(); onlineManager.setOnline(true); vi.restoreAllMocks(); vi.clearAllMocks(); useAuthStore.getState().logout(); mocks.search = "?tab=purchased"; mocks.realReads = false; mocks.id = "event-1"; });

let client = new QueryClient();
function renderPage() {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const view = render(<QueryClientProvider client={client}><EventPage /></QueryClientProvider>);
  return { ...view, rerender: () => view.rerender(<QueryClientProvider client={client}><EventPage /></QueryClientProvider>) };
}

it("uncached UUID Purchased route shows recovery before detail skeleton and defers gallery mounting", async () => {
  mocks.realReads = true;
  mocks.id = "b3ad2fe8-4489-4fbd-9c85-65fe8b3a4c15";
  useAuthStore.getState().setUser({ id: "seller", role: "PHOTOGRAPHER", name: "Seller", email: "seller@example.test", createdAt: "2026-09-01" });
  onlineManager.setOnline(false);
  const get = vi.spyOn(api, "get").mockImplementation(async (path) => path.includes("purchased-photos")
    ? { items: [{ photo, purchaseCount: 3 }], total: 1, offset: 0, limit: 24 }
    : { ...detail, id: mocks.id });
  renderPage();
  expect(await screen.findByText(/offline.*reconnect/i)).toBeVisible();
  expect(screen.getByRole("button", { name: "Retry" })).toBeEnabled();
  expect(screen.queryByRole("status")).not.toBeInTheDocument();
  expect(client.getQueryCache().getAll()).toHaveLength(1);
  expect(get).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole("button", { name: "Retry" }));
  expect(get).not.toHaveBeenCalled();
  act(() => onlineManager.setOnline(true));
  expect(await screen.findByRole("button", { name: /Open photo/ })).toBeEnabled();
  expect(get.mock.calls.filter(([path]) => path.includes("purchased-photos"))).toHaveLength(1);
});

it.each([true, false])("Purchased offline initial=%s and append pause preserve gallery selection", async (initialOffline) => {
  mocks.realReads = true;
  useAuthStore.getState().setUser({ id: "seller", role: "PHOTOGRAPHER", name: "Seller", email: "seller@example.test", createdAt: "2026-09-01" });
  const get = vi.spyOn(api, "get").mockImplementation(async (path) => {
    if (!path.includes("purchased-photos")) { if (initialOffline) onlineManager.setOnline(false); return detail; }
    const offset = Number(new URL(path, "https://test.invalid").searchParams.get("offset"));
    // Five on page one: photo 1 is not near the loaded end, so the preview does not load ahead.
    const items = offset === 0 ? [photo, ...[2, 3, 4, 5].map((n) => ({ ...photo, id: `photo-${n}` }))] : [{ ...photo, id: "photo-6" }];
    return { items: items.map((item) => ({ photo: item, purchaseCount: 3 })), total: 30, offset, limit: 24 };
  });
  renderPage();
  if (initialOffline) {
    expect(await screen.findByText(/offline.*reconnect/i)).toBeVisible();
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    expect(screen.queryByText("No completed purchases yet.")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    act(() => onlineManager.setOnline(true));
  }
  fireEvent.click((await screen.findAllByRole("button", { name: /Open photo/ }))[0]);
  expect(screen.getByRole("dialog")).toHaveTextContent("ABCDEF12");
  act(() => onlineManager.setOnline(false));
  fireEvent.click(screen.getByRole("button", { name: "Load 25 more" }));
  expect(await screen.findByText(/offline.*reconnect/i)).toBeVisible();
  expect(screen.getAllByRole("button", { name: /Open photo/ })[0]).toBeEnabled();
  expect(screen.getByRole("dialog")).toHaveTextContent("ABCDEF12");
  expect(screen.getByRole("button", { name: /waiting for connection/i })).toBeDisabled();
  expect(screen.queryByText("Loading…")).not.toBeInTheDocument();
  expect(get.mock.calls.filter(([path]) => path.includes("purchased-photos"))).toHaveLength(1);
  act(() => onlineManager.setOnline(true));
  await waitFor(() => expect(screen.getAllByRole("button", { name: /Open photo/ })).toHaveLength(6));
  expect(screen.getByRole("dialog")).toHaveTextContent("ABCDEF12");
});

it("removed event renders read-only purchase history", () => {
  useAuthStore.getState().setUser({ id: "seller", role: "PHOTOGRAPHER", name: "Seller", email: "seller@example.test", createdAt: "2026-09-01" });
  mocks.detail = detail;
  mocks.error = null;
  mocks.purchased.mockReturnValue({ items: [{ photo, purchaseCount: 3, lastPurchasedAt: "2026-09-07T10:00:00Z" }], total: 1, isLoading: false, error: null, appendError: null, hasNextPage: false, fetchNextPage: vi.fn(), refetch: vi.fn() });
  renderPage();
  expect(screen.getByText("Removed event")).toBeInTheDocument();
  expect(screen.getByText("This event was removed. Your purchase history is still available.")).toBeInTheDocument();
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

const FREE_NOTE = "Direct downloads from free events do not create purchase or buyer records.";
it.each([
  ["empty FREE", "free", null, 0],
  ["populated FREE", "free", null, 1],
  ["removed empty FREE", "free", "2026-09-08", 0],
  ["paid with 100% coupon", "paid", null, 1],
])("%s Purchased shows the completed-purchase wording", (_, pricingMode, deletedAt, count) => {
  mocks.detail = { ...detail, deletedAt, ownedByMe: false, pricingMode };
  mocks.error = null;
  mocks.active.mockReturnValue({ items: [], total: 0, isLoading: false, isFetchingNextPage: false, fetchNextPage: vi.fn() });
  mocks.purchased.mockReturnValue({ items: count ? [{ photo, purchaseCount: 3, lastPurchasedAt: "2026-09-07T10:00:00Z" }] : [], total: count, isLoading: false, error: null, appendError: null, hasNextPage: false, fetchNextPage: vi.fn(), refetch: vi.fn() });
  renderPage();
  expect(screen.queryByText("No completed purchases yet.") !== null).toBe(count === 0);
  expect(screen.queryByText(FREE_NOTE) !== null).toBe(pricingMode === "free");
  expect(screen.queryAllByRole("button", { name: "Open photo 1, 3 purchases" })).toHaveLength(count);
});

it("event error is retryable rather than notFound, while a genuine 404 stays missing", () => {
  mocks.detail = null;
  mocks.error = new ApiError([{ code: "TIMEOUT", message: "Timed out" }], 503);
  mocks.missing = false;
  const { rerender } = renderPage();
  fireEvent.click(screen.getByRole("button", { name: "Retry" }));
  expect(mocks.refetch).toHaveBeenCalledOnce();
  mocks.error = new ApiError([{ code: "EVENT_NOT_FOUND", message: "Missing" }], 404);
  mocks.missing = true;
  expect(() => rerender()).toThrow("404");
});

it("active and deleted tabs retain management and bare URLs default to active", () => {
  mocks.detail = { ...detail, deletedAt: null, ownedByMe: false, pricingMode: "free" };
  mocks.error = null;
  mocks.missing = false;
  mocks.search = "";
  mocks.active.mockImplementation(() => ({ items: [photo], total: 1, isLoading: false, isFetchingNextPage: false, fetchNextPage: vi.fn() }));
  mocks.visibility.mockReturnValue({ isSuccess: true, data: { metadataPublic: false } });
  renderPage();
  expect(screen.getByRole("tab", { name: "Active" })).toHaveAttribute("aria-selected", "true");
  expect(screen.getByRole("button", { name: "Select" })).toBeInTheDocument();
  fireEvent.click(screen.getByRole("tab", { name: "Deleted" }));
  expect(screen.getByRole("tab", { name: "Deleted" })).toHaveAttribute("aria-selected", "true");
  expect(screen.getByRole("button", { name: "Select" })).toBeInTheDocument();
});

it("shows exhausted purchased photos without an ineffective load control", () => {
  mocks.detail = detail;
  mocks.error = null;
  mocks.purchased.mockReturnValue({ items: [{ photo, purchaseCount: 3, lastPurchasedAt: "2026-09-07T10:00:00Z" }], total: 6, isLoading: false, error: null, appendError: null, hasNextPage: false, fetchNextPage: vi.fn(), refetch: vi.fn() });
  renderPage();
  expect(screen.getByRole("button", { name: "Open photo 1, 3 purchases" })).toBeInTheDocument();
  expect(screen.getByText("All available purchased photos loaded")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /Load .* more/ })).toBeNull();
});

it("does not force an active event back to Purchased after external URL navigation", () => {
  mocks.detail = { ...detail, deletedAt: null, ownedByMe: false, pricingMode: "free" };
  mocks.error = null;
  mocks.search = "?tab=purchased";
  mocks.active.mockReturnValue({ items: [], total: 0, isLoading: false, isFetchingNextPage: false, fetchNextPage: vi.fn() });
  mocks.purchased.mockReturnValue({ items: [{ photo, purchaseCount: 3, lastPurchasedAt: "2026-09-07T10:00:00Z" }], total: 1, isLoading: false, error: null, appendError: null, hasNextPage: false, fetchNextPage: vi.fn(), refetch: vi.fn() });
  const { rerender } = renderPage();
  expect(screen.getByRole("tab", { name: "Purchased" })).toHaveAttribute("aria-selected", "true");
  mocks.search = "?tab=deleted";
  rerender();
  expect(screen.getByRole("tab", { name: "Deleted" })).toHaveAttribute("aria-selected", "true");
  expect(mocks.replace).not.toHaveBeenCalled();
});

it.each([
  ["more pages, idle", { hasNextPage: true, isFetchingNextPage: false, appendError: null }, 1],
  ["already fetching", { hasNextPage: true, isFetchingNextPage: true, appendError: null }, 0],
  ["append failed", { hasNextPage: true, isFetchingNextPage: false, appendError: new Error("500") }, 0],
  ["no more pages", { hasNextPage: false, isFetchingNextPage: false, appendError: null }, 0],
])("preview at the last loaded purchased photo loads the next page: %s", (_, state, calls) => {
  useAuthStore.getState().setUser({ id: "seller", role: "PHOTOGRAPHER", name: "Seller", email: "seller@example.test", createdAt: "2026-09-01" });
  mocks.detail = detail;
  mocks.error = null;
  const fetchNextPage = vi.fn();
  mocks.purchased.mockReturnValue({ items: [{ photo, purchaseCount: 3, lastPurchasedAt: "2026-09-07T10:00:00Z" }], total: 210, isLoading: false, error: null, ...state, fetchNextPage, refetch: vi.fn() });
  renderPage();
  fireEvent.click(screen.getByRole("button", { name: "Open photo 1, 3 purchases" }));
  expect(fetchNextPage).toHaveBeenCalledTimes(calls);
});

it.each([
  ["failed", { error: new Error("500"), isPaused: false }],
  ["paused offline", { error: null, isPaused: true }],
])("Purchased header shows no count while the list is %s", (_, state) => {
  mocks.detail = { ...detail, deletedAt: null, ownedByMe: false, pricingMode: "paid" };
  mocks.error = null;
  mocks.active.mockReturnValue({ items: [], total: 0, isLoading: false, isFetchingNextPage: false, fetchNextPage: vi.fn() });
  mocks.purchased.mockReturnValue({ items: [], total: 0, isLoading: false, appendError: null, hasNextPage: false, fetchNextPage: vi.fn(), refetch: vi.fn(), ...state });
  renderPage();
  expect(screen.getByText("Purchased photos")).not.toHaveTextContent(/\d/);
});

it("a purchased thumbnail that failed to load is fetched again with a fresh URL after reconnecting", async () => {
  mocks.realReads = true;
  useAuthStore.getState().setUser({ id: "seller", role: "PHOTOGRAPHER", name: "Seller", email: "seller@example.test", createdAt: "2026-09-01" });
  let signed = 0;
  vi.spyOn(api, "get").mockImplementation(async (path) => path.includes("purchased-photos")
    ? { items: [{ photo: { ...photo, thumbnailUrl: `https://cdn.example.test/sig-${++signed}.jpg` }, purchaseCount: 3 }], total: 1, offset: 0, limit: 24 }
    : detail);
  renderPage();
  await screen.findByRole("button", { name: /Open photo/ });
  const img = () => document.querySelector("img") as HTMLImageElement;
  expect(img()).toHaveAttribute("src", "https://cdn.example.test/sig-1.jpg");
  fireEvent.error(img());
  act(() => { window.dispatchEvent(new Event("online")); });
  await waitFor(() => expect(img().getAttribute("src")).not.toContain("sig-1"));
});
