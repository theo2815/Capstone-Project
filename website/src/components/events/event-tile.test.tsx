import { render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { EventTile } from "@/components/events/event-tile";
import type { ListEvent } from "@/types/event";

vi.mock("@/components/events/save-button", () => ({ SaveButton: () => null }));

const base: ListEvent = {
  id: "e1", slug: "cebu-run", name: "Cebu Run", date: "2026-11-12", location: "IT Park, Cebu City",
  photoCount: 35, participantCount: 0, status: "ACTIVE", state: "open", city: "Cebu City",
  pricePerPhoto: 80, pricingMode: "paid",
};

it("tells a runner how many photos a paid race has and what one costs", () => {
  render(<EventTile event={base} />);
  expect(screen.getByText("35 photos · ₱80 each")).toBeInTheDocument();
  expect(screen.getByText("Photos ready")).toBeInTheDocument();
});

it("says a free race is free", () => {
  render(<EventTile event={{ ...base, photoCount: 5, pricingMode: "free", pricePerPhoto: 0 }} />);
  expect(screen.getByText("5 photos · Free")).toBeInTheDocument();
});

it("shows the starting price when photographers in the race charge different prices", () => {
  render(<EventTile event={{ ...base, priceVaries: true }} />);
  expect(screen.getByText("35 photos · From ₱80")).toBeInTheDocument();
});

it("shows only the count when no price is known", () => {
  render(<EventTile event={{ ...base, photoCount: 1, pricePerPhoto: undefined, pricingMode: undefined }} />);
  expect(screen.getByText("1 photo")).toBeInTheDocument();
});

it("does not promise photos for a race that has none", () => {
  render(<EventTile event={{ ...base, photoCount: 0 }} />);
  expect(screen.queryByText("Photos ready")).not.toBeInTheDocument();
  expect(screen.getByText("No photos yet")).toBeInTheDocument();
  expect(screen.queryByText(/^0 photos/)).not.toBeInTheDocument();
});

it("keeps the upcoming chip for a race that has not happened", () => {
  render(<EventTile event={{ ...base, photoCount: 0, state: "upcoming" }} />);
  expect(screen.getByText("Save the date")).toBeInTheDocument();
});
