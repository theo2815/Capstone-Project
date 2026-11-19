import { render, screen, within } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import RunnersPage from "@/app/runners/page";

vi.mock("@/components/layout/site-header", () => ({ SiteHeader: () => null }));
vi.mock("@/components/layout/footer", () => ({ Footer: () => null }));

// Runner audit P2-5: the hero's main button only scrolled to the explainer.
it("the hero's main button takes a runner straight to the events", () => {
  render(<RunnersPage />);
  const hero = screen.getByRole("heading", { level: 1 }).closest("section")!;
  const links = within(hero).getAllByRole("link");
  expect(links[0]).toHaveTextContent("Find my photos");
  expect(links[0]).toHaveAttribute("href", "/events");
  expect(within(hero).getByRole("link", { name: "See how" })).toHaveAttribute("href", "#step-1");
});
