import { onlineManager, QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { api, ApiError } from "@/lib/api";
import { savePhoto } from "@/lib/save-photo";
import { useAuthStore } from "@/store/auth-store";
import { formatOrderRef } from "@/lib/utils";
import type { PhotographerPurchase } from "@/lib/api-photographer-earnings";
import type { PhotographerLibraryPhoto, PhotographerTransaction } from "@/lib/photographer-mock";
import { PhotographerPurchasePreview } from "./photographer-purchase-preview";

vi.mock("@/lib/save-photo", () => ({ savePhoto: vi.fn() }));

const photo = { id: "photo-a", bib: "101", status: "live" as const, salesCount: 2, uploadedAt: "2026-10-01T00:00:00Z", tone: 0, span: "default" as const, thumbnailUrl: null };
const orderA = "aaaaaaaa-0000-0000-0000-000000000000";
const orderB = "bbbbbbbb-0000-0000-0000-000000000000";
const receipt = (orderId: string): PhotographerPurchase => ({ orderId, photoId: photo.id, buyer: "Aira S.", purchasedAt: "2026-10-01T16:30:00Z", orderCreatedAt: "2026-10-01T16:00:00Z", paymentMethod: "gcash", pricePhp: 150, discountPhp: 0, paidPhp: 150, kind: "paid", refundStatus: "none", refundPending: false, refundedPhp: null, refundedAt: null, netKeptPhp: 120, transactions: [] });
const response = (items: PhotographerPurchase[], selectedPurchase: PhotographerPurchase | null = null) => ({ photo, event: { id: "event-a", name: "Cebu Run", date: "2026-10-01", deletedAt: null }, items, total: items.length, offset: 0, limit: 25, selectedPurchase });
const login = (id = "seller-a") => useAuthStore.getState().setUser({ id, role: "PHOTOGRAPHER", name: "Seller", email: `${id}@example.test`, createdAt: "2026-09-10" });
const clicked = (orderId?: string): PhotographerTransaction => ({ id: "transaction-b", paidAt: "2026-10-01T16:30:00Z", eventId: "event-a", photoId: photo.id, buyer: "Clicked Buyer", amountKept: 120, orderId });
function view(selection: { photoId: string; orderId?: string; transactionId?: string } | null, client: QueryClient, transaction?: PhotographerTransaction) {
  return <QueryClientProvider client={client}><PhotographerPurchasePreview selection={selection} transaction={transaction} onClose={vi.fn()} /></QueryClientProvider>;
}
const fact = (label: string) => screen.getByText(label, { exact: true }).nextElementSibling;
afterEach(() => { onlineManager.setOnline(true); vi.restoreAllMocks(); vi.unstubAllGlobals(); useAuthStore.getState().logout(); });

it.each([true, false])("receipt history offline initial=%s preserves selected receipt on append and stays closed on reconnect", async (initialOffline) => {
  login();
  onlineManager.setOnline(!initialOffline);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const get = vi.spyOn(api, "get").mockImplementation(async (path) => {
    const offset = Number(new URL(path, "https://test.invalid").searchParams.get("offset"));
    return { ...response([receipt(offset === 0 ? orderA : orderB)]), offset, total: 50 };
  });
  const { rerender } = render(view({ photoId: photo.id }, client));
  if (initialOffline) {
    expect(await screen.findByText(/offline.*reconnect/i)).toBeVisible();
    expect(screen.queryByText("Loading purchases…")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry purchases" })).toBeEnabled();
    expect(get).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Retry purchases" }));
    act(() => onlineManager.setOnline(true));
  }
  await waitFor(() => expect(fact("Reference")).toHaveTextContent("AAAAAAAA"));
  act(() => onlineManager.setOnline(false));
  fireEvent.click(screen.getByRole("button", { name: "Load more purchases" }));
  expect(await screen.findByText(/offline.*reconnect/i)).toBeVisible();
  expect(fact("Reference")).toHaveTextContent("AAAAAAAA");
  expect(screen.getByRole("button", { name: /waiting for connection/i })).toBeDisabled();
  expect(get).toHaveBeenCalledTimes(1);
  act(() => onlineManager.setOnline(true));
  await screen.findByText("Purchase history");
  fireEvent.click(screen.getAllByRole("button", { name: /Aira S\./ })[1]);
  expect(fact("Reference")).toHaveTextContent("BBBBBBBB");
  act(() => { onlineManager.setOnline(false); client.invalidateQueries(); });
  expect(await screen.findByText(/offline.*reconnect/i)).toBeVisible();
  rerender(view(null, client));
  act(() => onlineManager.setOnline(true));
  await act(async () => { await Promise.resolve(); });
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(get).toHaveBeenCalledTimes(2);
  client.clear();
});

it.each(["seller-b", "runner-b"])("paused preview cannot populate or reopen after switching to %s", async (id) => {
  login();
  onlineManager.setOnline(false);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const get = vi.spyOn(api, "get").mockResolvedValue(response([receipt(orderA)]));
  const close = vi.fn();
  render(<QueryClientProvider client={client}><PhotographerPurchasePreview selection={{ photoId: photo.id }} onClose={close} /></QueryClientProvider>);
  expect(await screen.findByText(/offline.*reconnect/i)).toBeVisible();
  act(() => useAuthStore.getState().setUser({ id, role: id === "runner-b" ? "RUNNER" : "PHOTOGRAPHER", name: "Other", email: "other@example.test", createdAt: "2026-09-10" }));
  await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  act(() => onlineManager.setOnline(true));
  await act(async () => { await Promise.resolve(); });
  expect(get).not.toHaveBeenCalled();
  expect(close).toHaveBeenCalledOnce();
  client.clear();
});

it.each([undefined, true, false])("uses explicit availability %s without treating a missing thumbnail as denial", async (mediaAvailable) => {
  login();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const get = vi.spyOn(api, "get").mockResolvedValue({ ...response([receipt(orderB)], receipt(orderB)), photo: { ...photo, mediaAvailable } });
  render(view({ photoId: photo.id, orderId: orderB }, client));
  await screen.findByText("Aira S.");
  const button = screen.getByRole("button", { name: "Download photo" });
  if (mediaAvailable === false) {
    expect(button).toBeDisabled();
    expect(screen.queryByRole("button", { name: "Show Metadata" })).not.toBeInTheDocument();
    fireEvent.click(button);
    expect(get).toHaveBeenCalledTimes(1);
    expect(savePhoto).not.toHaveBeenCalled();
  } else {
    expect(button).toBeEnabled();
    expect(screen.getByRole("button", { name: "Show Metadata" })).toBeVisible();
  }
  expect(fact("Reference")).toHaveTextContent(formatOrderRef(orderB));
  client.clear();
});

it("reports delivery rejection after an available response without saving a fallback URL", async () => {
  login();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  vi.spyOn(api, "get").mockResolvedValueOnce({ ...response([receipt(orderB)], receipt(orderB)), photo: { ...photo, mediaAvailable: true } })
    .mockRejectedValueOnce(new ApiError([{ code: "PHOTO_NOT_FOUND", message: "Photo unavailable" }], 410));
  render(view({ photoId: photo.id, orderId: orderB }, client));
  await screen.findByText("Aira S.");
  fireEvent.click(screen.getByRole("button", { name: "Download photo" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Download unavailable. Try again.");
  expect(savePhoto).not.toHaveBeenCalled();
  expect(fact("Reference")).toHaveTextContent(formatOrderRef(orderB));
  client.clear();
});

it("refreshes an expired image URL from the seller response without losing receipts", async () => {
  login();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const expired = "https://cdn.example.test/expired.jpg";
  const fresh = "https://cdn.example.test/fresh.jpg";
  const get = vi.spyOn(api, "get")
    .mockResolvedValueOnce({ ...response([receipt(orderB)], receipt(orderB)), photo: { ...photo, thumbnailUrl: expired } })
    .mockResolvedValueOnce({ ...response([receipt(orderB)], receipt(orderB)), photo: { ...photo, thumbnailUrl: fresh } });
  render(view({ photoId: photo.id, orderId: orderB }, client));
  const image = await screen.findByRole("img", { name: "Race photo of bib 101" });
  expect(image).toHaveAttribute("src", expired);
  fireEvent.error(image);
  expect(screen.getByText("Preview unavailable")).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Refresh preview" }));
  await waitFor(() => expect(get).toHaveBeenCalledTimes(2));
  expect(await screen.findByRole("img", { name: "Race photo of bib 101" })).toHaveAttribute("src", fresh);
  expect(screen.getByText("Aira S.")).toBeVisible();
  client.clear();
});

it("keeps a newly selected receipt while a bounded same-URL refresh finishes", async () => {
  login();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const signed = "https://cdn.example.test/signed.jpg";
  const first = { ...response([
    { ...receipt(orderA), buyer: "Bea A." },
    { ...receipt(orderB), buyer: "Aira B." },
  ], { ...receipt(orderA), buyer: "Bea A." }), photo: { ...photo, thumbnailUrl: signed } };
  let resolveRefresh!: (value: typeof first) => void;
  const get = vi.spyOn(api, "get")
    .mockResolvedValueOnce(first)
    .mockImplementationOnce(() => new Promise((resolve) => { resolveRefresh = resolve; }));
  render(view({ photoId: photo.id }, client));
  const failedImage = await screen.findByRole("img", { name: "Race photo of bib 101" });
  fireEvent.error(failedImage);
  fireEvent.click(screen.getByRole("button", { name: "Refresh preview" }));
  await waitFor(() => expect(get).toHaveBeenCalledTimes(2));
  expect(get.mock.calls[1][0]).toContain("offset=0");
  expect(get.mock.calls[1][0]).toContain("limit=1");
  expect(get.mock.calls[1][0]).toContain(`orderId=${orderA}`);
  const pendingButton = screen.getByRole("button", { name: "Refreshing preview…" });
  fireEvent.click(pendingButton);
  expect(get).toHaveBeenCalledTimes(2);

  fireEvent.click(screen.getByRole("button", { name: /Aira B\./ }));
  expect(fact("Reference")).toHaveTextContent("BBBBBBBB");
  await act(async () => resolveRefresh(first));

  expect(fact("Reference")).toHaveTextContent("BBBBBBBB");
  const retriedImage = await screen.findByRole("img", { name: "Race photo of bib 101" });
  expect(retriedImage).not.toBe(failedImage);
  expect(retriedImage).toHaveAttribute("src", signed);
  client.clear();
});

it("does not replace private history when retrying a gallery fallback", async () => {
  login();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const get = vi.spyOn(api, "get").mockImplementation(() => new Promise(() => undefined));
  render(<QueryClientProvider client={client}><PhotographerPurchasePreview
    selection={{ photoId: photo.id }}
    photos={[{ ...photo, thumbnailUrl: "https://cdn.example.test/gallery-fallback.jpg" }]}
    onClose={vi.fn()}
  /></QueryClientProvider>);
  fireEvent.error(await screen.findByRole("img", { name: "Race photo of bib 101" }));

  expect(screen.getByText("Preview unavailable")).toBeVisible();
  fireEvent.click(screen.getByRole("button", { name: "Refresh preview" }));
  expect(get).toHaveBeenCalledTimes(1);
  expect(get.mock.calls[0][1]?.signal?.aborted).toBe(false);
  client.clear();
});

it("shows free purchase facts without inventing paid or refunded money", async () => {
  login();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const free = { ...receipt(orderB), kind: "free", discountPhp: 150, paidPhp: 0, refundedPhp: 0, netKeptPhp: 0 };
  vi.spyOn(api, "get").mockResolvedValue(response([free], free));
  render(view({ photoId: photo.id, orderId: orderB }, client));
  await screen.findByText("Aira S.");
  expect(fact("Buyer")).toHaveTextContent("Aira S.");
  expect(fact("Reference")).toHaveTextContent(formatOrderRef(orderB));
  expect(fact("Purchased")).toHaveTextContent("Oct 2, 2026, 12:30 AM");
  expect(fact("Event")).toHaveTextContent("Cebu Run");
  expect(screen.getByRole("link", { name: /Cebu Run.*Purchased photos/i })).toHaveAttribute("href", "/dashboard/events/event-a?tab=purchased");
  expect(fact("Photo")).toHaveTextContent("Bib 101");
  expect(fact("Photo price")).toHaveTextContent("₱150.00");
  expect(fact("Discount")).toHaveTextContent("₱150.00");
  expect(fact("Paid for this photo")).toHaveTextContent("Free");
  expect(fact("Refunded")).toHaveTextContent("₱0.00");
  expect(fact("Your net earnings")).toHaveTextContent("₱0.00");
  client.clear();
});

it("does not link an unavailable event from a transaction ID alone", async () => {
  login();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  vi.spyOn(api, "get").mockResolvedValue({ ...response([receipt(orderB)], receipt(orderB)), event: null });
  render(view({ photoId: photo.id, orderId: orderB }, client, clicked(orderB)));
  await screen.findByText("Aira S.");
  expect(fact("Event")).toHaveTextContent("Event unavailable");
  expect(fact("Event")?.querySelector("a")).toBeNull();
  expect(screen.queryByText(/archived/i)).not.toBeInTheDocument();
  client.clear();
});

it("keeps a clicked transaction's event name without linking an unconfirmed event", async () => {
  login();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  vi.spyOn(api, "get").mockResolvedValue({ ...response([receipt(orderB)], receipt(orderB)), event: null });
  render(view({ photoId: photo.id, orderId: orderB }, client, { ...clicked(orderB), eventName: "Cebu Run" }));
  await screen.findByText("Aira S.");
  expect(fact("Event")).toHaveTextContent("Cebu Run");
  expect(fact("Event")?.querySelector("a")).toBeNull();
  expect(screen.queryByText(/archived/i)).not.toBeInTheDocument();
  client.clear();
});

it.each([
  ["404", () => Promise.reject(new ApiError([{ code: "NOT_FOUND", message: "Missing" }], 404))],
  ["500", () => Promise.reject(new ApiError([{ code: "INTERNAL", message: "Down" }], 500))],
  ["loading", () => new Promise(() => undefined)],
  ["offline", null],
] as const)("shows neutral event context while purchases are %s", async (state, reply) => {
  login();
  if (!reply) onlineManager.setOnline(false);
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  vi.spyOn(api, "get").mockImplementation(reply ?? (() => new Promise(() => undefined)));
  vi.useFakeTimers({ shouldAdvanceTime: true });
  render(view({ photoId: photo.id, orderId: orderB }, client));
  // A 500 is retried with backoff before the error settles.
  if (state === "500") await act(() => vi.advanceTimersByTimeAsync(10_000));
  vi.useRealTimers();
  await waitFor(() => expect(fact("Event")).toHaveTextContent(state === "loading" ? "Loading…" : "Event unavailable"));
  expect(screen.queryByText(/archived/i)).not.toBeInTheDocument();
  client.clear();
});

it("distinguishes repeated buyer choices by reference and copies the selected full UUID", async () => {
  login();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const writeText = vi.fn().mockResolvedValue(undefined);
  vi.stubGlobal("navigator", { ...navigator, clipboard: { writeText } });
  vi.spyOn(api, "get").mockResolvedValue(response([receipt(orderA), receipt(orderB)], receipt(orderA)));
  render(view({ photoId: photo.id }, client));
  const choiceA = await screen.findByRole("button", { name: /Aira S\. · AAAAAAAA/ });
  const choiceB = screen.getByRole("button", { name: /Aira S\. · BBBBBBBB/ });
  expect(choiceA).toHaveAttribute("aria-pressed", "true");
  expect(choiceB).toHaveAttribute("aria-pressed", "false");
  fireEvent.click(screen.getByRole("button", { name: "Copy reference" }));
  await waitFor(() => expect(writeText).toHaveBeenLastCalledWith(orderA));
  expect(await screen.findByText("Reference copied.")).toBeVisible();

  fireEvent.click(choiceB);
  expect(fact("Reference")).toHaveTextContent(/^BBBBBBBB$/);
  expect(choiceB).toHaveAttribute("aria-pressed", "true");
  expect(choiceA).toHaveAttribute("aria-pressed", "false");
  expect(screen.queryByText("Reference copied.")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Copy reference" }));
  await waitFor(() => expect(writeText).toHaveBeenLastCalledWith(orderB));
  expect(writeText.mock.calls.flat().join(" ")).not.toMatch(/@|token/i);
  client.clear();
});

it("reports a rejected clipboard write honestly", async () => {
  login();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  vi.stubGlobal("navigator", { ...navigator, clipboard: { writeText: vi.fn().mockRejectedValue(new Error("denied")) } });
  vi.spyOn(api, "get").mockResolvedValue(response([receipt(orderB)], receipt(orderB)));
  render(view({ photoId: photo.id, orderId: orderB }, client));
  await screen.findByText("Aira S.");
  fireEvent.click(screen.getByRole("button", { name: "Copy reference" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Couldn't copy reference.");
  expect(screen.queryByText("Reference copied.")).not.toBeInTheDocument();
  client.clear();
});

it("disables reference copy when no order is known", async () => {
  login();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  vi.spyOn(api, "get").mockResolvedValue(response([{ ...receipt(orderA), buyer: "Other Buyer" }]));
  render(view({ photoId: photo.id }, client, clicked()));
  expect(await screen.findByText("Purchase unavailable.")).toBeVisible();
  expect(fact("Reference")).toHaveTextContent("Unavailable");
  expect(screen.getByRole("button", { name: "Copy reference" })).toBeDisabled();
  client.clear();
});

it("lists a pinned receipt outside the loaded page once and marks it selected", async () => {
  login();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  vi.spyOn(api, "get").mockResolvedValue({ ...response([receipt(orderA)], receipt(orderB)), total: 51 });
  render(view({ photoId: photo.id, orderId: orderB }, client));
  const pinned = await screen.findByRole("button", { name: /BBBBBBBB/ });
  expect(pinned).toHaveAttribute("aria-pressed", "true");
  expect(screen.getAllByRole("button", { name: /Aira S\. · / })).toHaveLength(2);
  fireEvent.click(screen.getByRole("button", { name: /AAAAAAAA/ }));
  expect(fact("Reference")).toHaveTextContent("AAAAAAAA");
  expect(screen.getByRole("button", { name: /BBBBBBBB/ })).toHaveAttribute("aria-pressed", "false");
  client.clear();
});

it("drops copy feedback across an account switch and copies the new seller's order", async () => {
  login();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const writeText = vi.fn().mockResolvedValue(undefined);
  vi.stubGlobal("navigator", { ...navigator, clipboard: { writeText } });
  vi.spyOn(api, "get").mockResolvedValueOnce(response([receipt(orderA)], receipt(orderA))).mockResolvedValue(response([receipt(orderB)], receipt(orderB)));
  const rendered = render(view({ photoId: photo.id }, client));
  await screen.findByText("Aira S.");
  fireEvent.click(screen.getByRole("button", { name: "Copy reference" }));
  expect(await screen.findByText("Reference copied.")).toBeVisible();
  act(() => login("seller-b"));
  await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  rendered.rerender(view(null, client));
  rendered.rerender(view({ photoId: photo.id }, client));
  await waitFor(() => expect(fact("Reference")).toHaveTextContent("BBBBBBBB"));
  expect(screen.queryByText("Reference copied.")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Copy reference" }));
  await waitFor(() => expect(writeText).toHaveBeenLastCalledWith(orderB));
  client.clear();
});

it("separates a completed buyer refund from seller earnings adjustments", async () => {
  login();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const full = { ...receipt(orderB), refundStatus: "full", refundedPhp: 150, refundedAt: "2026-10-03T01:15:00Z", netKeptPhp: 0, transactions: [
    { id: "refund", isRefund: true, refundOf: "sale", amountKept: -120, discountPhp: 0, paidAt: "2026-10-01T16:30:00Z", recordedAt: "2026-10-03T01:30:00Z" },
  ] };
  vi.spyOn(api, "get").mockResolvedValue(response([full], full));
  render(view({ photoId: photo.id, orderId: orderB, transactionId: "refund" }, client));
  await screen.findByText("Aira S.");
  expect(fact("Refunded")).toHaveTextContent("₱150.00");
  expect(fact("Refund status")).toHaveTextContent("Full");
  expect(fact("Refund completed")).toHaveTextContent("Oct 3, 2026, 9:15 AM");
  expect(fact("Your net earnings")).toHaveTextContent("₱0.00");
  expect(await screen.findByText(/Earnings adjustment/)).toHaveAttribute("aria-current", "true");
  client.clear();
});

it("keeps legacy recorded refunds and unknown buyer refund amounts honest", async () => {
  login();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const recorded = { ...receipt(orderB), refundStatus: "recorded", refundedPhp: null, refundedAt: null, netKeptPhp: 90, transactions: [
    { id: "refund", isRefund: true, refundOf: "sale", amountKept: -30, discountPhp: 0, paidAt: "2026-10-01T16:30:00Z", recordedAt: "2026-10-03T01:30:00Z" },
  ] };
  vi.spyOn(api, "get").mockResolvedValue(response([recorded], recorded));
  render(view({ photoId: photo.id, orderId: orderB }, client));
  await screen.findByText("Aira S.");
  expect(fact("Refunded")).toHaveTextContent("Unavailable");
  expect(fact("Refund status")).toHaveTextContent("Recorded");
  expect(fact("Refund recorded")).toHaveTextContent("Oct 3, 2026, 9:30 AM");
  expect(screen.queryByText("Refund completed")).not.toBeInTheDocument();
  expect(screen.getByText(/Earnings adjustment/)).toHaveTextContent("-₱30.00");
  client.clear();
});

it("does not present order creation as a completed purchase time", async () => {
  login();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const noPaidTime = { ...receipt(orderB), purchasedAt: null };
  vi.spyOn(api, "get").mockResolvedValue(response([noPaidTime], noPaidTime));
  render(view({ photoId: photo.id, orderId: orderB }, client));
  await screen.findByText("Aira S.");
  expect(fact("Purchased")).toHaveTextContent("Unavailable");
  expect(fact("Order created")).toHaveTextContent("Oct 2, 2026, 12:00 AM");
  client.clear();
});

it.each(["logout", "account switch", "selection close"])("does not save a delayed download after %s", async (change) => {
  login();
  vi.mocked(savePhoto).mockClear();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  let resolveDownload!: (value: { url: string; expiresAt: string }) => void;
  const get = vi.spyOn(api, "get").mockImplementation((path) => path.endsWith("/download")
    ? new Promise((resolve) => { resolveDownload = resolve; })
    : Promise.resolve(response([receipt(orderB)], receipt(orderB))));
  const rendered = render(view({ photoId: photo.id, orderId: orderB }, client));
  await screen.findByText("Aira S.");
  fireEvent.click(screen.getByRole("button", { name: "Download photo" }));
  await waitFor(() => expect(get.mock.calls.some(([path]) => path.endsWith("/download"))).toBe(true));
  if (change === "logout") act(() => useAuthStore.getState().logout());
  else if (change === "account switch") act(() => login("seller-b"));
  else rendered.rerender(view(null, client));
  await act(async () => resolveDownload({ url: "https://cdn.example.test/seller-a-original.jpg", expiresAt: "2026-10-03T00:00:00Z" }));
  expect(savePhoto).not.toHaveBeenCalled();
  client.clear();
});

it("hides an old clicked fallback across seller sessions until a fresh selection", async () => {
  login();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  let resolveOld!: (value: ReturnType<typeof response>) => void;
  const oldRequest = new Promise<ReturnType<typeof response>>((resolve) => { resolveOld = resolve; });
  const get = vi.spyOn(api, "get").mockReturnValueOnce(oldRequest)
    .mockRejectedValueOnce(new ApiError([{ code: "NOT_FOUND", message: "Not yours" }], 404));
  const selection = { photoId: photo.id };
  const oldTransaction = { ...clicked(), buyer: "Old Buyer" };
  const oldPhoto = { ...photo, thumbnailUrl: "https://cdn.example.test/seller-a.jpg" };
  const onClose = vi.fn();
  const sellerView = (picked: typeof selection | null, transaction: PhotographerTransaction, pictures: PhotographerLibraryPhoto[] = [oldPhoto]) =>
    <QueryClientProvider client={client}><PhotographerPurchasePreview selection={picked} transaction={transaction} photos={pictures} onClose={onClose} /></QueryClientProvider>;
  const rendered = render(sellerView(selection, oldTransaction));
  expect(await screen.findByText("Old Buyer")).toBeVisible();
  expect(screen.getByRole("img", { name: "Race photo of bib 101" })).toHaveAttribute("src", oldPhoto.thumbnailUrl);
  act(() => login("seller-b"));
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(screen.queryByText("Old Buyer")).not.toBeInTheDocument();
  expect(onClose).toHaveBeenCalledTimes(1);
  await act(async () => resolveOld(response([receipt(orderB)], receipt(orderB))));
  expect(screen.queryByText("Old Buyer")).not.toBeInTheDocument();
  rendered.rerender(sellerView(null, oldTransaction));
  rendered.rerender(sellerView(selection, { ...clicked(), buyer: "Fresh Buyer" }, [{ ...photo, thumbnailUrl: null }]));
  expect(await screen.findByText("Fresh Buyer")).toBeVisible();
  await waitFor(() => expect(get).toHaveBeenCalledTimes(2));
  expect(screen.queryByText("Old Buyer")).not.toBeInTheDocument();
  client.clear();
});

it("same photo different order replaces context", async () => {
  login();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  vi.spyOn(api, "get").mockResolvedValueOnce(response([receipt(orderA)], receipt(orderA))).mockResolvedValueOnce(response([receipt(orderB)], receipt(orderB)));
  const rendered = render(view({ photoId: photo.id, orderId: orderA }, client));
  expect(await screen.findByText(formatOrderRef(orderA))).toBeVisible();
  rendered.rerender(view({ photoId: photo.id, orderId: orderB }, client));
  expect(await screen.findByText(formatOrderRef(orderB))).toBeVisible();
  expect(screen.queryByText(formatOrderRef(orderA))).not.toBeInTheDocument();
  const writeText = vi.fn().mockResolvedValue(undefined);
  vi.stubGlobal("navigator", { ...navigator, clipboard: { writeText } });
  fireEvent.click(screen.getByRole("button", { name: "Copy reference" }));
  await waitFor(() => expect(writeText).toHaveBeenCalledWith(orderB));
  client.clear();
});

it("pins the clicked transaction order beyond the first receipt page", async () => {
  login();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const get = vi.spyOn(api, "get").mockResolvedValue(response([receipt(orderA)], receipt(orderB)));
  render(view({ photoId: photo.id }, client, clicked(orderB)));
  expect(await screen.findByText(formatOrderRef(orderB))).toBeVisible();
  expect(get.mock.calls[0][0]).toContain(`orderId=${orderB}`);
  client.clear();
});

it("keeps an older clicked transaction distinct from the first fetched purchase", async () => {
  login();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  vi.spyOn(api, "get").mockResolvedValue(response([{ ...receipt(orderA), buyer: "Other Buyer" }]));
  render(view({ photoId: photo.id }, client, clicked()));
  expect(await screen.findByText("Clicked Buyer")).toBeVisible();
  expect(screen.queryByText("Other Buyer")).not.toBeInTheDocument();
  expect(screen.queryByText(formatOrderRef(orderA))).not.toBeInTheDocument();
  expect(await screen.findByText("Purchase unavailable.")).toBeVisible();
  expect(screen.getByRole("button", { name: "Retry purchases" })).toBeVisible();
  client.clear();
});

it("ignores a pending response when the account changes with the preview open", async () => {
  login();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  let resolveOld!: (value: ReturnType<typeof response>) => void;
  const old = new Promise<ReturnType<typeof response>>((resolve) => { resolveOld = resolve; });
  const get = vi.spyOn(api, "get").mockReturnValueOnce(old).mockResolvedValue(response([{ ...receipt(orderB), buyer: "New Buyer" }], { ...receipt(orderB), buyer: "New Buyer" }));
  const rendered = render(view({ photoId: photo.id, orderId: orderB }, client));
  await waitFor(() => expect(get).toHaveBeenCalledTimes(1));
  const oldSignal = get.mock.calls[0][1]?.signal;
  act(() => login("seller-b"));
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(oldSignal?.aborted).toBe(true);
  await act(async () => resolveOld(response([{ ...receipt(orderB), buyer: "Old Buyer" }], { ...receipt(orderB), buyer: "Old Buyer" })));
  expect(screen.queryByText("Old Buyer")).not.toBeInTheDocument();
  rendered.rerender(view(null, client));
  rendered.rerender(view({ photoId: photo.id, orderId: orderB }, client));
  await waitFor(() => expect(screen.getByText("New Buyer")).toBeVisible());
  expect(screen.getByText("New Buyer")).toBeVisible();
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
  expect(await screen.findByText(/Earnings adjustment/)).toHaveAttribute("aria-current", "true");
  expect(screen.getAllByText(formatOrderRef(orderB))).toHaveLength(1);
  client.clear();
});

it("does not turn unavailable historical money into zero", async () => {
  login();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const unpriced = { ...receipt(orderB), netKeptPhp: null };
  vi.spyOn(api, "get").mockResolvedValue(response([unpriced], unpriced));
  render(view({ photoId: photo.id, orderId: orderB }, client));
  await screen.findByText("Aira S.");
  expect(fact("Paid for this photo")).toHaveTextContent("₱150.00");
  expect(fact("Your net earnings")).toHaveTextContent("Unavailable");
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

const gallery = (n: number) => Array.from({ length: n }, (_, i) => ({ ...photo, id: i === 0 ? photo.id : `photo-${i}` }));
const counter = (text: string) => screen.getAllByText((_, el) => el?.tagName === "SPAN" && el.textContent === text)[0];

it("counter shows the purchased total, not only the loaded photos", async () => {
  login();
  vi.spyOn(api, "get").mockResolvedValue(response([receipt(orderA)]));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><PhotographerPurchasePreview selection={{ photoId: photo.id }} photos={gallery(5)} totalPhotos={210} onSelectPhoto={vi.fn()} onClose={vi.fn()} /></QueryClientProvider>);
  await waitFor(() => expect(fact("Reference")).toHaveTextContent("AAAAAAAA"));
  expect(counter("1 / 210")).toBeVisible();
});

it.each([[0, false], [3, true], [4, true]])("near the loaded end (photo %i of 5) asks for more = %s", async (at, asks) => {
  login();
  vi.spyOn(api, "get").mockResolvedValue(response([receipt(orderA)]));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const photos = gallery(5);
  const onNeedMore = vi.fn();
  render(<QueryClientProvider client={client}><PhotographerPurchasePreview selection={{ photoId: photos[at].id }} photos={photos} totalPhotos={210} onNeedMore={onNeedMore} onSelectPhoto={vi.fn()} onClose={vi.fn()} /></QueryClientProvider>);
  await screen.findByRole("dialog");
  expect(onNeedMore).toHaveBeenCalledTimes(asks ? 1 : 0);
});

it("does not ask for more once every purchased photo is loaded", async () => {
  login();
  vi.spyOn(api, "get").mockResolvedValue(response([receipt(orderA)]));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const photos = gallery(5);
  const onNeedMore = vi.fn();
  render(<QueryClientProvider client={client}><PhotographerPurchasePreview selection={{ photoId: photos[4].id }} photos={photos} totalPhotos={5} onNeedMore={onNeedMore} onSelectPhoto={vi.fn()} onClose={vi.fn()} /></QueryClientProvider>);
  await screen.findByRole("dialog");
  expect(counter("5 / 5")).toBeVisible();
  expect(onNeedMore).not.toHaveBeenCalled();
});
