import { render } from "@testing-library/react";
import { createPortal } from "react-dom";
import { useRef } from "react";
import { afterEach, expect, it } from "vitest";
import { useInertBackground } from "@/hooks/use-inert-background";

function Overlay({ open, label }: { open: boolean; label: string }) {
  const ref = useRef<HTMLDivElement>(null);
  useInertBackground(ref, open);
  return open ? createPortal(<div ref={ref} role="dialog" aria-label={label} />, document.body) : null;
}

const page = () => document.getElementById("page")!;
const dialog = (label: string) => document.querySelector(`[aria-label="${label}"]`)!.parentElement === document.body
  ? document.querySelector(`[aria-label="${label}"]`)!
  : null;

afterEach(() => { document.body.innerHTML = ""; });

// Runner audit P3-7: with the cart or checkout open, keyboard focus could reach the page behind.
it("makes the page behind an open overlay inert, and gives it back on close", () => {
  const { rerender } = render(<div id="page"><button>Buy</button></div>);
  rerender(<div id="page"><button>Buy</button><Overlay open label="Cart" /></div>);
  expect(dialog("Cart")).not.toHaveAttribute("inert");
  expect(document.querySelector("#page")!.closest("[inert]")).not.toBeNull();
  rerender(<div id="page"><button>Buy</button><Overlay open={false} label="Cart" /></div>);
  expect(document.querySelector("#page")!.closest("[inert]")).toBeNull();
});

it("keeps only the newest overlay live when two are open, and restores correctly as they close", () => {
  const ui = (cart: boolean, checkout: boolean) => (
    <div id="page"><Overlay open={cart} label="Cart" /><Overlay open={checkout} label="Checkout" /></div>
  );
  const { rerender } = render(ui(true, false));
  rerender(ui(true, true));
  expect(dialog("Checkout")).not.toHaveAttribute("inert");
  expect(dialog("Cart")).toHaveAttribute("inert");
  // The cart closes first (the hand-off to checkout): the page must stay inert.
  rerender(ui(false, true));
  expect(page().closest("[inert]")).not.toBeNull();
  expect(dialog("Checkout")).not.toHaveAttribute("inert");
  rerender(ui(false, false));
  expect(page().closest("[inert]")).toBeNull();
});

it("leaves alone anything another component made inert", () => {
  const { rerender } = render(<div id="page" />);
  const other = document.createElement("div");
  other.setAttribute("inert", "");
  document.body.appendChild(other);
  rerender(<div id="page"><Overlay open label="Cart" /></div>);
  rerender(<div id="page"><Overlay open={false} label="Cart" /></div>);
  expect(other).toHaveAttribute("inert");
});
