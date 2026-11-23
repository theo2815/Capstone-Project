import { render, screen } from "@testing-library/react";
import { expect, it } from "vitest";
import { ROUTES } from "@/lib/constants";
import { AdminSectionIntro } from "./admin-section-intro";

it("shows the intro, the 'use … instead' link and the words, at every width", () => {
  const { container } = render(<AdminSectionIntro href={ROUTES.ADMIN_VERIFICATIONS} />);

  const intro = screen.getByText(/^Two lists\. Profiles:/);
  expect(intro.className).not.toMatch(/(^|\s)hidden(\s|$)/);

  expect(screen.getByRole("link", { name: "Photographers" })).toHaveAttribute("href", ROUTES.ADMIN_PHOTOGRAPHERS);
  expect(container).toHaveTextContent("To look up, suspend or message any photographer, use Photographers.");

  const terms = container.querySelector("dl");
  expect(terms).toHaveTextContent(
    "Seller details: the legal name, ID and payout account a photographer sends before they may upload.",
  );
  expect(terms).toHaveTextContent("Held: sent back once already; after the photographer fixes it, it waits for your approval.");
});

it("a page with no look-alikes and no words shows only its intro", () => {
  const { container } = render(<AdminSectionIntro href={ROUTES.ADMIN_SECURITY} />);
  expect(container.querySelectorAll("a")).toHaveLength(0);
  expect(container.querySelector("dl")).not.toBeNull();
  const { container: c2 } = render(<AdminSectionIntro href={ROUTES.ADMIN_EVENT_REQUESTS} />);
  expect(c2.querySelector("dl")).toBeNull();
  expect(c2.querySelectorAll("p")).toHaveLength(2);
});
