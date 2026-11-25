import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import { adminUser } from "@/test/fixtures/admin";
import { AdminPhotographerCard } from "./admin-photographer-card";

afterEach(cleanup);

it("Pending uses the warning tone", () => {
  render(<AdminPhotographerCard row={adminUser} />);
  expect(screen.getByText("Pending")).toHaveClass("text-warning-text");
});

// FU-18 (B-37): touch has no hover, so a long name and place wrap instead of hiding behind a title.
it("a long name and region wrap, with no title repeating them", () => {
  const { container } = render(<AdminPhotographerCard row={adminUser} />);
  expect(screen.getByRole("heading", { name: "Test Studio" })).not.toHaveClass("truncate");
  expect(screen.getByText(/^Cebu · Cebu$/).closest(".truncate")).toBeNull();
  expect(container.querySelector("[title]")).toBeNull();
});

// AX-39 (UA7): row values read as typed; mono caps stay for labels only.
it("row values are not upper-cased, and the join month is in sentence case", () => {
  render(<AdminPhotographerCard row={adminUser} />);
  expect(screen.getByText("@test-studio").closest(".uppercase")).toBeNull();
  expect(screen.getByText(/since Sep 2026/).closest(".uppercase")).toBeNull();
  expect(screen.getByText(/fields$/).closest(".uppercase")).toBeNull();
});
