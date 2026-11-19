import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";
import { BibEmptyResult } from "@/components/events/bib-empty-result";
import type { EventDetail } from "@/types/event";

const event = (status: EventDetail["status"]) =>
  ({ id: "e1", slug: "cebu-run", name: "Cebu Run", status }) as EventDetail;

it("does not offer an email alert it cannot send for an open race", () => {
  render(<BibEmptyResult event={event("ACTIVE")} bib="1234" onClear={() => {}} />);
  expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /notify me/i })).not.toBeInTheDocument();
  expect(screen.queryByText(/we'll (email|ping) you/i)).not.toBeInTheDocument();
});

it("does not claim photographers are still uploading when nothing says so", () => {
  render(<BibEmptyResult event={event("ACTIVE")} bib="1234" onClear={() => {}} />);
  expect(screen.queryByText(/still uploading|still working/i)).not.toBeInTheDocument();
  expect(screen.getByText(/no photos with bib 1234 yet/i)).toBeInTheDocument();
});

it("keeps the way back to the full gallery", async () => {
  const onClear = vi.fn();
  render(<BibEmptyResult event={event("ACTIVE")} bib="1234" onClear={onClear} />);
  await userEvent.setup().click(screen.getByRole("button", { name: /skim the full gallery/i }));
  expect(onClear).toHaveBeenCalledOnce();
});

it("keeps the finished-race copy", () => {
  render(<BibEmptyResult event={event("COMPLETED")} bib="1234" onClear={() => {}} />);
  expect(screen.getByText("Bib not found.")).toBeInTheDocument();
});
