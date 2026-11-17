import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState, type ComponentType } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { CartModal } from "@/components/cart/cart-modal";
import { CheckoutModal } from "@/components/cart/checkout-modal";
import { FloatingCart } from "@/components/cart/floating-cart";
import { useCartStore } from "@/store/cart-store";
import { useUiStore } from "@/store/ui-store";

// R8 review I3 (2026-10-10): neither panel took focus when it opened, so a
// keyboard or screen-reader user stayed on the page behind it.
const http = vi.hoisted(() => ({ get: vi.fn(async () => ({ items: [], total: 0, offset: 0, limit: 200 })), post: vi.fn(async () => ({})) }));
vi.mock("@/lib/api", async (importOriginal) => ({ ...(await importOriginal<typeof import("@/lib/api")>()), api: http }));
vi.mock("next/navigation", () => ({
  usePathname: () => "/events/race", useSearchParams: () => new URLSearchParams(), useRouter: () => ({ push: vi.fn() }),
}));

afterEach(() => {
  cleanup();
  useCartStore.getState().setItems([]);
});

type Panel = ComponentType<{ isOpen: boolean; onClose: () => void }>;

function Harness({ Panel }: { Panel: Panel }) {
  const [open, setOpen] = useState(false);
  return (
    <QueryClientProvider client={new QueryClient()}>
      <button type="button" onClick={() => setOpen(true)}>Open panel</button>
      <Panel isOpen={open} onClose={() => setOpen(false)} />
    </QueryClientProvider>
  );
}

it.each([
  ["cart", CartModal as Panel],
  ["checkout", CheckoutModal as Panel],
])("the %s panel takes focus when it opens and gives it back when it closes", async (_name, Panel) => {
  useCartStore.getState().setItems([{ photoId: "p1", eventId: "e1", eventName: "Race", price: 125 } as never]);
  const user = userEvent.setup();
  render(<Harness Panel={Panel} />);
  const opener = screen.getByRole("button", { name: "Open panel" });

  await user.click(opener);
  const dialog = await screen.findByRole("dialog");
  await waitFor(() => expect(dialog.contains(document.activeElement)).toBe(true));

  await user.keyboard("{Escape}");
  await waitFor(() => expect(opener).toHaveFocus());
});

// The I3 path: the cart closes (and hands focus back to the tile) at the
// moment the checkout opens, so focus must still end inside the checkout.
it("continuing from the cart to checkout leaves focus inside the checkout", async () => {
  useCartStore.getState().setItems([{ photoId: "p1", eventId: "e1", eventName: "Race", price: 125 } as never]);
  useUiStore.setState({ cartOpen: false, checkoutOpen: false });
  const user = userEvent.setup();
  render(
    <QueryClientProvider client={new QueryClient()}>
      <button type="button">Gallery tile</button>
      <FloatingCart />
    </QueryClientProvider>,
  );
  const tile = screen.getByRole("button", { name: "Gallery tile" });
  tile.focus();
  act(() => useUiStore.getState().openCart());

  await user.click(await screen.findByRole("button", { name: /Continue to checkout/ }));
  const checkout = await screen.findByRole("dialog", { name: "Checkout" });
  await waitFor(() => expect(checkout.contains(document.activeElement)).toBe(true));

  await user.keyboard("{Escape}");
  await waitFor(() => expect(tile).toHaveFocus());
});
