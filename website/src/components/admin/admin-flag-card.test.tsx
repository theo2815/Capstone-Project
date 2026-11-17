import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, expect, it } from "vitest";
import { canConfirmHold, canHide, flagReasonLabel, type Flag } from "@/lib/admin-flags";
import { AdminFlagCard } from "./admin-flag-card";

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
  render(<QueryClientProvider client={client}><AdminFlagCard flag={flag} /></QueryClientProvider>);
}
afterEach(cleanup);

it("labels the system reasons in plain words", () => {
  expect(flagReasonLabel("same_bib_duplicate")).toBe("Possible duplicate: same bibs as another race");
  expect(flagReasonLabel("profile_image_flagged")).not.toBe("profile_image_flagged");
  expect(flagReasonLabel("something_new")).toBe("something_new");
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

it("a photographer flag links to their admin page and names no event", () => {
  renderCard(userFlag);
  expect(screen.getByRole("link", { name: "@juanshots" })).toHaveAttribute("href", "/admin/photographers/juanshots");
  expect(screen.queryByText(/Unknown event/)).not.toBeInTheDocument();
});

const relevanceFlag: Flag = {
  ...base, id: "flag-r1", reason: "batch_relevance_low", targetKind: "user", targetId: "user-9",
  note: "event e1: copy 5/8", eventName: "Cebu City Marathon 2027",
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
  expect(screen.getByText("Ends the payout hold. Photos this flag hid stay hidden; you can restore them one by one.")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Confirm resolve" })).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
  await userEvent.click(screen.getByRole("button", { name: "Dismiss" }));
  expect(screen.getByText("False alarm. Ends the payout hold and puts back on sale every photo this flag hid.")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Confirm dismiss" })).toBeInTheDocument();
});

it("other flags have no Resolve button", () => {
  renderCard(userFlag);
  expect(screen.queryByRole("button", { name: "Resolve" })).not.toBeInTheDocument();
});
