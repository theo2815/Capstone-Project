import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, expect, it, vi } from "vitest";
import { RefundModal } from "./refund-modal";

afterEach(cleanup);

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
