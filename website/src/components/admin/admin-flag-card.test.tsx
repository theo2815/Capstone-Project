import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, expect, it, vi } from "vitest";
import { canConfirmHold, canHide, type Flag } from "@/lib/admin-flags";
import { AdminFlagCard, type FlagCardActions } from "./admin-flag-card";

// The queue owns the action hook and modals; a lone card gets inert ones.
const actions: FlagCardActions = { isPending: () => false, pendingAction: () => null, onHide: vi.fn(), onEscalate: vi.fn(), onRestore: vi.fn(), onConfirm: vi.fn(async () => true) };

// System flags (SystemFlagService) target events and photographers, not only photos.
const base: Flag = {
  id: "flag-9", photoId: null, eventId: null, eventName: null, photographerHandle: "juanshots",
  reportedBy: "system", reason: "possible_duplicate", note: "Looks like Cebu Marathon 2027", status: "open",
  reportedAt: "2026-11-08T00:00:00Z", reviewedAt: null, reviewedBy: null, reviewerNote: null,
  photoSnapshot: { alt: "Flag", kmMark: null, bib: null, thumbnailUrl: null },
  targetKind: "event", targetId: "event-9", payoutHoldAt: null,
};
const eventFlag: Flag = { ...base, eventId: "event-9", eventName: "Cebu City Marathon 2027" };
const userFlag: Flag = { ...base, reason: "profile_image_flagged", targetKind: "user", targetId: "user-9" };

function renderCard(flag: Flag) {
  const client = new QueryClient();
  render(<QueryClientProvider client={client}><AdminFlagCard flag={flag} actions={actions} /></QueryClientProvider>);
}
afterEach(cleanup);

// AX-64: the title is the row's reasonLabel; a row without one shows its code.
it("the card shows the backend's label, or the code when there is none", () => {
  renderCard({ ...eventFlag, reasonLabel: "Possible duplicate of another race" });
  expect(screen.getByRole("heading", { name: "Possible duplicate of another race" })).toBeInTheDocument();
  cleanup();
  renderCard(eventFlag);
  expect(screen.getByRole("heading", { name: "possible_duplicate" })).toBeInTheDocument();
});

it("only photo flags can be hidden", () => {
  expect(canHide({ ...base, targetKind: "photo", targetId: "photo-1", photoId: "photo-1" })).toBe(true);
  expect(canHide(eventFlag)).toBe(false);
  expect(canHide(userFlag)).toBe(false);
});

it("an event flag shows the event name and note, with no Hide button", () => {
  renderCard(eventFlag);
  expect(screen.getByText(/Cebu City Marathon 2027/)).toBeInTheDocument();
  expect(screen.getByText("Looks like Cebu Marathon 2027")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Hide…" })).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Dismiss" })).toBeInTheDocument();
});

// FT-19 (C-64, C-66): the card says who raised the flag, and its dismiss promises nothing about a photo.
it("a flag the system raised says Automatic check, and its dismiss sentence names no photo", async () => {
  renderCard(eventFlag);
  expect(screen.getByText(/Automatic check/)).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Dismiss" }));
  const sentence = screen.getByText("Close this flag as a false alarm. Nothing changes for the race or the photographer, and nobody is told. The dismissal is logged.");
  expect(sentence.textContent).not.toMatch(/\bphoto\b/i);
});

it("a photographer flag links to their admin page and names no event", () => {
  renderCard(userFlag);
  expect(screen.getByRole("link", { name: "@juanshots" })).toHaveAttribute("href", "/admin/photographers/juanshots");
  expect(screen.queryByText(/Unknown event/)).not.toBeInTheDocument();
});

const relevanceFlag: Flag = {
  ...base, id: "flag-r1", reason: "batch_relevance_low", targetKind: "user", targetId: "user-9",
  note: "event e1: copy 5/8", eventName: "Cebu City Marathon 2027", reasonLabel: "Uploads may not belong to the event",
};

it("a relevance flag shows Hold payouts and no Hide button", () => {
  renderCard(relevanceFlag);
  expect(screen.getByText("Uploads may not belong to the event")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Hold payouts" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Resolve" })).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Hide…" })).not.toBeInTheDocument();
  expect(screen.queryByText(/Payouts on hold since/)).not.toBeInTheDocument();
});

it("a confirmed hold shows since when and offers no second confirm", () => {
  renderCard({ ...relevanceFlag, payoutHoldAt: "2026-11-12T03:00:00Z" });
  expect(screen.getByText(/Payouts on hold since .*2026/)).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Hold payouts" })).not.toBeInTheDocument();
  expect(canConfirmHold({ ...relevanceFlag, payoutHoldAt: "2026-11-12T03:00:00Z" })).toBe(false);
  expect(canConfirmHold({ ...relevanceFlag, status: "dismissed" })).toBe(false);
  expect(canConfirmHold(userFlag)).toBe(false);
});

it("a relevance flag offers Resolve and both confirms say what happens to the hold and the photos", async () => {
  renderCard({ ...relevanceFlag, payoutHoldAt: "2026-11-12T03:00:00Z" });
  await userEvent.click(screen.getByRole("button", { name: "Resolve" }));
  expect(screen.getByText("Ends the payout hold. Photos this flag hid stay hidden; you can restore them one by one. If this flag holds the photographer's payouts and no other flag does, they are told the hold is lifted.")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Confirm resolve" })).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
  await userEvent.click(screen.getByRole("button", { name: "Dismiss" }));
  expect(screen.getByText("False alarm. Ends the payout hold and puts back on sale every photo this flag hid. If this flag holds the photographer's payouts and no other flag does, they are told the hold is lifted.")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Confirm dismiss" })).toBeInTheDocument();
});

it("other flags have no Resolve button", () => {
  renderCard(userFlag);
  expect(screen.queryByRole("button", { name: "Resolve" })).not.toBeInTheDocument();
});

it("an openable card opens through a button named by its title; inner actions do not open it", async () => {
  const onOpen = vi.fn();
  const photoFlag: Flag = { ...base, reason: "inappropriate", reasonLabel: "Inappropriate or NSFW", targetKind: "photo", targetId: "photo-9", photoId: "photo-9" };
  const { container } = render(<QueryClientProvider client={new QueryClient()}><AdminFlagCard flag={photoFlag} actions={actions} onOpen={onOpen} /></QueryClientProvider>);
  expect(container.querySelector('[role="button"]')).toBeNull();
  await userEvent.click(screen.getByRole("button", { name: "Dismiss" }));
  expect(onOpen).not.toHaveBeenCalled();
  await userEvent.click(screen.getByRole("button", { name: "Inappropriate or NSFW" }));
  expect(onOpen).toHaveBeenCalledTimes(1);
});

// AX-39 (UA7): row values read as typed; mono caps stay for labels only.
it("row values are not upper-cased, and the date is in sentence case", () => {
  renderCard(eventFlag);
  expect(screen.getByText("flag-9").closest(".uppercase")).toBeNull();
  expect(screen.getByText(/Nov \d\d · 2026/).closest(".uppercase")).toBeNull();
});

// FU-10 review: an inline "Confirm dismiss" left open must not survive the flag turning hidden (it would skip the Restore confirm).
it("an open dismiss confirm goes away when the flag becomes hidden", async () => {
  const photoFlag: Flag = { ...base, targetKind: "photo", targetId: "photo-9", photoId: "photo-9" };
  const client = new QueryClient();
  const { rerender } = render(<QueryClientProvider client={client}><AdminFlagCard flag={photoFlag} actions={actions} /></QueryClientProvider>);
  await userEvent.click(screen.getByRole("button", { name: "Dismiss" }));
  expect(screen.getByRole("button", { name: "Confirm dismiss" })).toBeInTheDocument();
  rerender(<QueryClientProvider client={client}><AdminFlagCard flag={{ ...photoFlag, status: "hidden" }} actions={actions} /></QueryClientProvider>);
  expect(screen.queryByRole("button", { name: "Confirm dismiss" })).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Restore photo & dismiss" })).toBeInTheDocument();
});
