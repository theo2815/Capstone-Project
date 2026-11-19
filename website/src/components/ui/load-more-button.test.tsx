import { render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { LoadMoreButton } from "./load-more-button";

it("says nothing under a list that was complete from the start", () => {
  const { container } = render(<LoadMoreButton shown={2} total={2} increment={10} onLoadMore={vi.fn()} />);
  expect(container).toBeEmptyDOMElement();
});

it("ends a list that was loaded in steps with the all-loaded line", () => {
  const { rerender } = render(<LoadMoreButton shown={10} total={12} increment={10} onLoadMore={vi.fn()} />);
  expect(screen.getByRole("button", { name: "Load 2 more" })).toBeInTheDocument();
  rerender(<LoadMoreButton shown={12} total={12} increment={10} onLoadMore={vi.fn()} />);
  expect(screen.getByText("All 12 loaded")).toBeInTheDocument();
});

it("always shows an explicit terminal label", () => {
  render(<LoadMoreButton shown={3} total={3} increment={10} onLoadMore={vi.fn()} terminalLabel="All available photos loaded" />);
  expect(screen.getByText("All available photos loaded")).toBeInTheDocument();
});
