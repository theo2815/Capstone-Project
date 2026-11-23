import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import AdminError from "./error";

// X5: a render error under /admin stays inside the admin shell with a Retry.
describe("AdminError", () => {
  it("shows the alert and calls retry on click", () => {
    const retry = vi.fn();
    render(<AdminError error={new Error("boom")} retry={retry} />);
    expect(screen.getByRole("alert")).toHaveTextContent("This admin page could not load.");
    expect(screen.getByRole("alert")).toHaveTextContent("It may be your connection. Press Retry, or reload the page.");
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(retry).toHaveBeenCalledTimes(1);
  });
});
