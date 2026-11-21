import { fireEvent, render, screen } from "@testing-library/react";
import { expect, it } from "vitest";
import { SkipLink } from "@/components/layout/skip-link";

// Runner audit P3-2: keyboard users tabbed through the header on every page.
it("is the first thing a keyboard user reaches and moves focus to the page's main content", () => {
  render(
    <>
      <SkipLink />
      <header><button type="button">Events</button></header>
      <main><h1>Orders.</h1></main>
    </>,
  );
  const link = screen.getByRole("link", { name: "Skip to content" });
  fireEvent.click(link);
  expect(document.activeElement).toBe(document.querySelector("main"));
});

it("does nothing harmful on a page without a main element", () => {
  render(<SkipLink />);
  fireEvent.click(screen.getByRole("link", { name: "Skip to content" }));
  expect(document.activeElement).toBe(document.body);
});
