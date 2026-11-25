import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import { CollapsibleReviewSection, CollapsibleReviewSlab } from "./admin-collapse";

afterEach(cleanup);

// FT-43 (B-26): aria-controls may only point at a body that is in the page.
it.each([
  ["slab", () => <CollapsibleReviewSlab number="01" title="Payouts"><p>Body</p></CollapsibleReviewSlab>],
  ["section", () => <CollapsibleReviewSection kicker="Payouts"><p>Body</p></CollapsibleReviewSection>],
])("a closed %s has no aria-controls; an open one points at its body", (_, ui) => {
  render(ui());
  const toggle = screen.getByRole("button");
  expect(toggle).toHaveAttribute("aria-expanded", "false");
  expect(toggle).not.toHaveAttribute("aria-controls");
  fireEvent.click(toggle);
  expect(document.getElementById(toggle.getAttribute("aria-controls")!)).toHaveTextContent("Body");
});

// FU-21: a 44 px header whose name says its state, and a caption on phones too.
it.each([
  ["slab", () => <CollapsibleReviewSlab number="01" title="Payouts" caption="Where the money goes"><p>Body</p></CollapsibleReviewSlab>],
  ["section", () => <CollapsibleReviewSection kicker="Payouts"><p>Body</p></CollapsibleReviewSection>],
])("a %s header is 44 px tall and its name carries Show or Hide", (_, ui) => {
  render(ui());
  const toggle = screen.getByRole("button", { name: /Payouts.*Show/ });
  expect(toggle).toHaveClass("min-h-11");
  expect(toggle).toHaveAttribute("aria-expanded", "false");
  fireEvent.click(toggle);
  expect(screen.getByRole("button", { name: /Payouts.*Hide/ })).toHaveAttribute("aria-expanded", "true");
});

it("shows the slab caption at every width", () => {
  render(<CollapsibleReviewSlab number="01" title="Completeness" caption="What the photographer has filled in"><p>Body</p></CollapsibleReviewSlab>);
  expect(screen.getByText("What the photographer has filled in")).not.toHaveClass("hidden");
});
