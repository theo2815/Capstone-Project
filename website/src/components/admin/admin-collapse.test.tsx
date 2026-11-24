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
