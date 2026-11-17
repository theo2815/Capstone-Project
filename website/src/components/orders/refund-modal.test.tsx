import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, expect, it, vi } from "vitest";
import { RefundModal } from "./refund-modal";
import { REFUND_POLICY_BULLETS } from "@/lib/refund-policy";
import { DISPUTE_REASON_LABEL } from "@/lib/admin-disputes";
import { SUPPORT_EMAIL } from "@/lib/constants";

afterEach(cleanup);

const ORDER = { id: "order-1", eventId: "event-1", photoIds: ["photo-1"], total: 100, paymentMethod: "gcash", paidAt: "2026-09-09T00:00:00Z" };

// Refund policy v2 (owner-approved 2026-11-08): ten points, in this order.
it("states the ten v2 policy points in order", () => {
  expect(REFUND_POLICY_BULLETS.map((b) => b.kicker)).toEqual([
    "Our promise", "Download problems", "Blurred photos", "Charged more than once", "Checkout errors",
    "What we don't refund", "How to ask", "How long it takes", "How much comes back", "Photos bought together",
  ]);
  expect(REFUND_POLICY_BULLETS[6].body).toContain(SUPPORT_EMAIL);
});

it("offers the v2 reasons, without Wrong runner, and still labels old wrong_runner requests", async () => {
  const user = userEvent.setup();
  render(<QueryClientProvider client={new QueryClient()}><RefundModal mode="request" isOpen onClose={vi.fn()} eventName="Cebu Race" order={ORDER} /></QueryClientProvider>);
  await user.click(screen.getByRole("button", { name: "Refund reason" }));
  expect(screen.getAllByRole("menuitem").map((m) => m.textContent)).toEqual([
    "Too blurred to recognise", "Won't download or file is broken", "Charged more than once", "Checkout error or something else",
  ]);
  expect(DISPUTE_REASON_LABEL.wrong_runner).toMatch(/^Wrong runner/);
});

it("closes only the reason list on Escape and keeps the refund form and its note", async () => {
  const user = userEvent.setup(); const onClose = vi.fn();
  render(<QueryClientProvider client={new QueryClient()}><RefundModal mode="request" isOpen onClose={onClose} eventName="Cebu Race"
    order={{ id: "order-1", eventId: "event-1", photoIds: ["photo-1"], total: 100, paymentMethod: "gcash", paidAt: "2026-09-09T00:00:00Z" }} /></QueryClientProvider>);
  const note = within(screen.getByRole("dialog", { name: "Request a refund" })).getByRole("textbox");
  await user.type(note, "Keep this note");
  await user.click(screen.getByRole("button", { name: "Refund reason" }));
  expect(screen.getAllByRole("menuitem").length).toBeGreaterThan(0);
  await user.keyboard("{Escape}");
  expect(screen.queryAllByRole("menuitem")).toEqual([]);
  expect(onClose).not.toHaveBeenCalled();
  expect(note).toHaveValue("Keep this note");
  await user.keyboard("{Escape}");
  expect(onClose).toHaveBeenCalledTimes(1);
});
