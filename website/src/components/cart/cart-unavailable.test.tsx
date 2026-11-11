import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, renderHook, screen, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { CartModal } from "@/components/cart/cart-modal";
import { useCheckoutWorkflow } from "@/hooks/use-checkout-workflow";
import { useCartStore } from "@/store/cart-store";
import { formatPrice } from "@/lib/utils";

// AA-I20a: a suspended photographer's photo comes back from GET /me/cart?includeUnavailable=true
// with available: false. It stays visible (with Remove) but is never priced or checked out.
const http = vi.hoisted(() => ({ get: vi.fn(async () => ({ items: [], total: 0, offset: 0, limit: 200 })), post: vi.fn(async () => ({})) }));
vi.mock("@/lib/api", async (importOriginal) => ({ ...(await importOriginal<typeof import("@/lib/api")>()), api: http }));
vi.mock("next/navigation", () => ({
  usePathname: () => "/events/race", useSearchParams: () => new URLSearchParams(), useRouter: () => ({ push: vi.fn() }),
}));

afterEach(() => {
  cleanup();
  useCartStore.setState({ items: [] });
});

function Wrapper({ children }: { children: ReactNode }) {
  return <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>;
}

it("an unavailable row shows 'No longer available' and is left out of the total and the checkout items", () => {
  useCartStore.setState({ items: [
    { photoId: "live", eventId: "e1", eventName: "Race", bib: "101", price: 125, thumbnailUrl: "" },
    { photoId: "gone", eventId: "e1", eventName: "Race", bib: "202", price: 150, thumbnailUrl: "", available: false },
  ] });

  render(<CartModal isOpen onClose={() => {}} />, { wrapper: Wrapper });

  const goneRow = screen.getByRole("button", { name: "Remove 202 from cart" }).closest("li")!;
  expect(within(goneRow).getByText("No longer available")).toBeInTheDocument();
  expect(screen.getAllByText("No longer available")).toHaveLength(1);
  expect(screen.getByText("Subtotal").parentElement).toHaveTextContent(formatPrice(125));

  const { result } = renderHook(() => useCheckoutWorkflow(true), { wrapper: Wrapper });
  expect(result.current.items.map((i) => i.photoId)).toEqual(["live"]);
  expect(result.current.payable).toBe(125);
});
