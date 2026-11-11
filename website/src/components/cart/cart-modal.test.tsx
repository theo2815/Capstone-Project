import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { CartModal } from "@/components/cart/cart-modal";
import { useCartStore } from "@/store/cart-store";
import { formatPrice } from "@/lib/utils";

// AA-R16b: a guest cart reads live per-photo prices (a photographer may charge
// above the race price), and a row says when its price rose since it was added.
const http = vi.hoisted(() => ({
  get: vi.fn(async (path: string) =>
    path.startsWith("/events/") ? { pricePerPhoto: 125 } : { items: [], total: 0, offset: 0, limit: 200 },
  ),
  post: vi.fn(async (path: string, body: { photoIds?: string[] }) =>
    path === "/coupons/preview"
      ? { code: null, items: [], discountTotal: 0, eligibleCount: 0, livePrices: (body.photoIds ?? []).map((photoId) => ({ photoId, price: 150 })) }
      : {},
  ),
}));
vi.mock("@/lib/api", async (importOriginal) => ({ ...(await importOriginal<typeof import("@/lib/api")>()), api: http }));
vi.mock("next/navigation", () => ({
  usePathname: () => "/events/race", useSearchParams: () => new URLSearchParams(), useRouter: () => ({ push: vi.fn() }),
}));

afterEach(() => {
  cleanup();
  useCartStore.setState({ items: [], syncEnabled: false });
  vi.clearAllMocks();
});

function Wrapper({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>;
}

it("guest refresh uses live per-photo prices", async () => {
  // A photographer's raised ₱150 photo; the event-level price (the race floor) is ₱125.
  useCartStore.setState({ items: [
    { photoId: "raised", eventId: "e1", eventSlug: "race", eventName: "Race", bib: "101", price: 150, priceAtAdd: 150, thumbnailUrl: "" },
  ] });

  render(<CartModal isOpen onClose={() => {}} />, { wrapper: Wrapper });

  await waitFor(() => expect(http.post).toHaveBeenCalledWith("/coupons/preview", { photoIds: ["raised"] }));
  await new Promise((r) => setTimeout(r, 0));
  expect(useCartStore.getState().items[0].price).toBe(150);
  expect(screen.getByText("Subtotal").parentElement).toHaveTextContent(formatPrice(150));
});

it("note shows when price rose since add", () => {
  useCartStore.setState({ syncEnabled: true, items: [
    { photoId: "up", eventId: "e1", eventName: "Race", bib: "101", price: 150, priceAtAdd: 125, thumbnailUrl: "" },
    { photoId: "same", eventId: "e1", eventName: "Race", bib: "202", price: 125, priceAtAdd: 125, thumbnailUrl: "" },
  ] });

  render(<CartModal isOpen onClose={() => {}} />, { wrapper: Wrapper });

  const row = screen.getByRole("button", { name: "Remove 101 from cart" }).closest("li")!;
  expect(within(row).getByText(`Price went up from ${formatPrice(125)} to ${formatPrice(150)}`)).toBeInTheDocument();
  expect(screen.getAllByText(/Price went up/)).toHaveLength(1);
});
