import { act, cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { admin, adminEvent } from "@/test/fixtures/admin";
import { adminEventKeys } from "@/hooks/use-admin-events";
import { useAuthStore } from "@/store/auth-store";
import { useToastStore } from "@/store/toast-store";
import { EventRequestsQueue } from "./event-requests-queue";
let client: QueryClient;
const key = adminEventKeys.list(admin.id, { review: "queue" });
beforeEach(() => {
  useAuthStore.getState().setUser(admin); useToastStore.getState().clear();
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  client.setQueryData(key, [{ ...adminEvent, status: "DRAFT", reviewStatus: "pending", createdByName: "Photographer" }]);
  vi.spyOn(api, "get").mockResolvedValue({ items: [] });
});
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); vi.restoreAllMocks(); });
it.each(["pending", "change_pending"] as const)("retains %s rejection input through failure and retries", async (reviewStatus) => {
  client.setQueryData(key, [{ ...adminEvent, reviewStatus }]);
  const user = userEvent.setup(); let fail!: (error: Error) => void;
  const post = vi.spyOn(api, "post").mockReturnValueOnce(new Promise((_, reject) => { fail = reject; }));
  render(<QueryClientProvider client={client}><EventRequestsQueue /></QueryClientProvider>);
  const label = reviewStatus === "pending" ? "Send back" : "Decline";
  await user.click(screen.getByRole("button", { name: label }));
  const dialog = screen.getByRole("dialog"); const reason = within(dialog).getByLabelText("Reason (max 500)");
  await user.type(reason, "Please clarify pricing"); await user.click(within(dialog).getByRole("button", { name: label }));
  expect(reason).toBeDisabled(); await user.keyboard("{Escape}{Enter}"); expect(post).toHaveBeenCalledTimes(1);
  expect(within(dialog).getByRole("button", { name: reviewStatus === "pending" ? "Sending back…" : "Declining…" })).toBeDisabled();
  expect(screen.getByRole("button", { name: "Approve", hidden: true })).toBeDisabled();
  await act(async () => { fail(new Error("Unavailable")); });
  expect(reason).toHaveValue("Please clarify pricing"); await waitFor(() => expect(reason).toBeEnabled()); expect(dialog).toBeInTheDocument();
  post.mockResolvedValueOnce({ ...adminEvent, reviewStatus: reviewStatus === "pending" ? "rejected" : "approved" });
  await user.click(within(dialog).getByRole("button", { name: label }));
  await waitFor(() => expect(dialog).not.toBeInTheDocument());
  expect(client.getQueryData(key)).toEqual([]);
  expect(post).toHaveBeenLastCalledWith(`/admin/events/${adminEvent.id}/reject`, { reason: "Please clarify pricing", expectedRevision: 4 });
});
it("awaits approval before success and prevents duplicate submission", async () => {
  let confirm!: (value: unknown) => void;
  const post = vi.spyOn(api, "post").mockReturnValueOnce(new Promise((resolve) => { confirm = resolve; }));
  render(<QueryClientProvider client={client}><EventRequestsQueue /></QueryClientProvider>);
  const submit = screen.getByRole("button", { name: "Approve" });
  await userEvent.click(submit); await userEvent.click(submit);
  expect(submit).toBeDisabled(); expect(post).toHaveBeenCalledTimes(1); expect(useToastStore.getState().toasts).toEqual([]);
  await act(async () => { confirm(adminEvent); });
  await waitFor(() => expect(client.getQueryData(key)).toEqual([]));
  expect(useToastStore.getState().toasts[0].kind).toBe("success");
});
it("shows everything the photographer filled in", () => {
  client.setQueryData(key, [{
    ...adminEvent, status: "DRAFT", reviewStatus: "pending", visibility: "unlisted",
    createdByName: "Ana Cruz", createdByHandle: "anashoots", organizerName: "Cebu Runners Club",
    bannerUrl: "https://cdn.example.test/cover.jpg", description: "Start 5 AM.\nThree loops.",
  }]);
  render(<QueryClientProvider client={client}><EventRequestsQueue /></QueryClientProvider>);
  expect(screen.getByRole("img", { name: "Cover photo for Cebu Race" })).toHaveAttribute("src", "https://cdn.example.test/cover.jpg");
  expect(screen.getByText("Ana Cruz · @anashoots")).toBeInTheDocument();
  expect(screen.getByText("Cebu Runners Club")).toBeInTheDocument();
  expect(screen.getByText("Unlisted · link only")).toBeInTheDocument();
  expect(screen.getByText(/Start 5 AM\.\s+Three loops\./)).toBeInTheDocument();
});
// AA-R17b: race context from the queue fields (AA-R7).
const target = {
  id: "race-0", slug: "cebu-marathon", name: "Cebu Marathon", date: "2026-09-10", location: "SRP, Cebu City",
  regionCode: "region-vii", provinceCode: "cebu", organizerName: "Cebu Runners", eventType: "ROAD_RACE",
  bannerUrl: null, reviewStatus: "approved", strength: "strong" as const,
};
const race = {
  ...adminEvent, status: "DRAFT" as const, reviewStatus: "pending" as const, eventType: "ROAD_RACE",
  submittedByHandle: "anashoots", submittedByName: "Ana Cruz", checkReasons: ["POSSIBLE_DUPLICATE"], possibleDuplicateOf: target,
};
it("shows duplicate candidates and decides only after the API answers", async () => {
  client.setQueryData(key, [race]); const user = userEvent.setup();
  let confirm!: (value: unknown) => void;
  const post = vi.spyOn(api, "post").mockReturnValueOnce(new Promise((resolve) => { confirm = resolve; }));
  render(<QueryClientProvider client={client}><EventRequestsQueue /></QueryClientProvider>);
  expect(screen.getByRole("link", { name: "Cebu Marathon" })).toHaveAttribute("href", "/events/cebu-marathon");
  expect(screen.getByText("It may be the same race as one already listed.")).toBeInTheDocument();
  expect(screen.getByText("Ana Cruz · @anashoots")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Different race" })).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Same race → join" }));
  expect(post).toHaveBeenCalledWith(`/admin/events/${race.id}/same-race`, { targetEventId: target.id, expectedRevision: 4 });
  expect(screen.getByRole("button", { name: "Different race" })).toBeDisabled();
  expect(client.getQueryData(key)).toEqual([race]); expect(useToastStore.getState().toasts).toEqual([]);
  await act(async () => { confirm(target); });
  await waitFor(() => expect(client.getQueryData(key)).toEqual([]));
  expect(useToastStore.getState().toasts[0]).toEqual(expect.objectContaining({ kind: "success", message: expect.stringMatching(/joined Cebu Marathon/) }));
});
it("a held race edit shows the changed fields", () => {
  client.setQueryData(key, [{
    ...adminEvent, eventType: "ROAD_RACE", reviewStatus: "change_pending", submittedByHandle: "anashoots", checkReasons: ["BAD_WORDS"],
    pendingChange: { kind: "race_edit", fields: { title: "Cebu City Marathon", date: "2026-09-11" }, coverS3Key: "events/x/cover.jpg", requestedBy: "u", requestedAt: "2026-09-09T00:00:00Z" },
  }]);
  render(<QueryClientProvider client={client}><EventRequestsQueue /></QueryClientProvider>);
  expect(screen.getByText(/Race edit · live race/)).toBeInTheDocument();
  const title = screen.getByText("Title").closest("div")!;
  expect(title).toHaveTextContent("Cebu Race"); expect(title).toHaveTextContent("Cebu City Marathon");
  expect(screen.getByText("Date").closest("div")).toHaveTextContent("2026-09-11");
  expect(screen.getByText("A new cover photo")).toBeInTheDocument();
  expect(screen.getByText("The name or notes use a word an admin needs to check.")).toBeInTheDocument();
  expect(screen.queryByText(/Requested/)).not.toBeInTheDocument();
});
const HELD_EDIT = {
  ...adminEvent, eventType: "ROAD_RACE", reviewStatus: "change_pending", submittedByHandle: "anashoots",
  pendingChange: { kind: "race_edit", fields: { title: "Cebu City Marathon" }, coverS3Key: "events/x/cover.jpg", requestedBy: "u", requestedAt: "2026-09-09T00:00:00Z" },
};
it("a held race edit with a cover link shows the image", () => {
  client.setQueryData(key, [{ ...HELD_EDIT, heldCoverUrl: "https://r2.example/held-cover.jpg" }]);
  render(<QueryClientProvider client={client}><EventRequestsQueue /></QueryClientProvider>);
  const img = screen.getByRole("img", { name: `New cover photo for ${adminEvent.name}` });
  expect(img).toHaveAttribute("src", "https://r2.example/held-cover.jpg");
  expect(img).toHaveAttribute("loading", "lazy");
});
it("a held cover with no link keeps the words", () => {
  client.setQueryData(key, [{ ...HELD_EDIT, heldCoverUrl: null }]);
  render(<QueryClientProvider client={client}><EventRequestsQueue /></QueryClientProvider>);
  expect(screen.getByText("A new cover photo")).toBeInTheDocument();
  expect(screen.queryByRole("img", { name: /^New cover photo for/ })).not.toBeInTheDocument();
});
it("an Other request shows the race it looks like", () => {
  client.setQueryData(key, [{ ...adminEvent, status: "DRAFT", reviewStatus: "pending", createdByHandle: "anashoots", lookalike: { ...target, strength: "medium" } }]);
  render(<QueryClientProvider client={client}><EventRequestsQueue /></QueryClientProvider>);
  expect(screen.getByText(/Looks like a shared race:/).closest("p")).toHaveTextContent("Cebu Marathon");
});
