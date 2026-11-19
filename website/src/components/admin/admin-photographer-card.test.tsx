import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import { adminUser } from "@/test/fixtures/admin";
import { AdminPhotographerCard } from "./admin-photographer-card";

afterEach(cleanup);

it("Pending uses the warning tone", () => {
  render(<AdminPhotographerCard row={adminUser} />);
  expect(screen.getByText("Pending")).toHaveClass("text-warning-text");
});

it("a long name and region keep their full text in a title", () => {
  render(<AdminPhotographerCard row={adminUser} />);
  expect(screen.getByRole("heading", { name: "Test Studio" })).toHaveAttribute("title", "Test Studio");
  expect(screen.getByTitle(/^Cebu · Cebu$/)).toBeInTheDocument();
});
