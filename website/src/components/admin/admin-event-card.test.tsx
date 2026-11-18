import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { adminEvent } from "@/test/fixtures/admin";
import { AdminEventCard } from "./admin-event-card";

afterEach(cleanup);

const review = { owner: "@runner", status: "approved" as const, visibility: "public" as const, listed: false };

function mount(extra: Partial<Parameters<typeof AdminEventCard>[0]> = {}) {
  render(
    <AdminEventCard event={adminEvent} review={review} onEdit={vi.fn()} onDelete={vi.fn()}
      onSetListing={vi.fn()} raceHidden={false} onRaceListing={vi.fn()} {...extra} />,
  );
}

it("names every row action after its event (EV2)", () => {
  mount();
  for (const name of ["Edit Cebu Race", "Delete Cebu Race", "Hide Cebu Race"]) {
    expect(screen.getByRole("button", { name })).toBeInTheDocument();
  }
  expect(screen.getByRole("link", { name: "View Cebu Race" })).toBeInTheDocument();
  expect(screen.getByRole("checkbox", { name: "Show on /events: Cebu Race" })).toBeInTheDocument();
});

it("styles Delete as the house danger button and the row has no hover fill (EV3, EV5)", () => {
  mount();
  expect(screen.getByRole("button", { name: "Delete Cebu Race" })).toHaveClass("text-error", "rounded-full", "min-h-[44px]");
  expect(screen.getByRole("button", { name: "Edit Cebu Race" })).toHaveClass("rounded-full", "min-h-[44px]");
  expect(screen.getByRole("article").className).not.toMatch(/hover:bg/);
});

it("keeps green off the Live row text; only the dot is fresh (UA6)", () => {
  mount();
  const state = screen.getByText("Live now");
  expect(state).toHaveClass("text-ink");
  expect(state.querySelector("[aria-hidden]")).toHaveClass("bg-fresh");
});
