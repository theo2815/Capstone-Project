import { act, cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { admin, adminEvent } from "@/test/fixtures/admin";
import { adminEventKeys } from "@/hooks/use-admin-events";
import { useAuthStore } from "@/store/auth-store";
import { useToastStore } from "@/store/toast-store";
import { describePricing } from "@/lib/photographer-events";
import { EventRequestsQueue } from "./event-requests-queue";
// A tiny in-memory URL: the kind filter lives in ?kind=.
const nav = vi.hoisted(() => {
  let search = "";
  const listeners = new Set<() => void>();
  return {
    get: () => search,
    set: (next: string) => { search = next; listeners.forEach((l) => l()); },
    subscribe: (l: () => void) => { listeners.add(l); return () => listeners.delete(l); },
  };
});
vi.mock("next/navigation", async () => {
  const { useSyncExternalStore } = await import("react");
  return {
    usePathname: () => "/admin/event-requests",
    useSearchParams: () => new URLSearchParams(useSyncExternalStore(nav.subscribe, nav.get)),
    useRouter: () => ({ replace: (url: string) => nav.set(url.split("?")[1] ?? ""), push: vi.fn() }),
  };
});
let client: QueryClient;
const key = adminEventKeys.list(admin.id, { review: "queue" });
beforeEach(() => {
  useAuthStore.getState().setUser(admin); useToastStore.getState().clear();
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  client.setQueryData(key, [{ ...adminEvent, status: "DRAFT", reviewStatus: "pending", createdByName: "Photographer" }]);
  vi.spyOn(api, "get").mockResolvedValue({ items: [] });
});
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); vi.restoreAllMocks(); nav.set(""); });
it.each(["pending", "change_pending"] as const)("retains %s rejection input through failure and retries", async (reviewStatus) => {
  client.setQueryData(key, [{ ...adminEvent, reviewStatus }]);
  const user = userEvent.setup(); let fail!: (error: Error) => void;
  const post = vi.spyOn(api, "post").mockReturnValueOnce(new Promise((_, reject) => { fail = reject; }));
  render(<QueryClientProvider client={client}><EventRequestsQueue /></QueryClientProvider>);
  const label = reviewStatus === "pending" ? "Send back" : "Decline";
  await user.click(screen.getByRole("button", { name: label }));
  const dialog = screen.getByRole("dialog"); const reason = within(dialog).getByLabelText("Reason (max 500)");
  expect(dialog).toHaveTextContent(/sees your reason in their inbox. Write what they should change./);
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
  await userEvent.click(screen.getByRole("button", { name: "Approve" }));
  const submit = within(screen.getByRole("dialog")).getByRole("button", { name: "Approve" });
  await userEvent.click(submit); await userEvent.click(submit);
  expect(submit).toBeDisabled(); expect(post).toHaveBeenCalledTimes(1); expect(useToastStore.getState().toasts).toEqual([]);
  await act(async () => { confirm(adminEvent); });
  await waitFor(() => expect(client.getQueryData(key)).toEqual([]));
  expect(useToastStore.getState().toasts[0].kind).toBe("success");
});
// ER2: nothing goes public, and no price changes, in one unseen click.
it("Approve asks first and says what goes public", async () => {
  const post = vi.spyOn(api, "post").mockResolvedValue(adminEvent); const user = userEvent.setup();
  render(<QueryClientProvider client={client}><EventRequestsQueue /></QueryClientProvider>);
  await user.click(screen.getByRole("button", { name: "Approve" }));
  expect(post).not.toHaveBeenCalled();
  const dialog = screen.getByRole("dialog", { name: "Approve this event?" });
  expect(dialog).toHaveTextContent("Cebu Race goes live and uploads open.");
  expect(dialog).toHaveTextContent("Public · link and profile");
  await user.click(within(dialog).getByRole("button", { name: "Cancel" }));
  expect(post).not.toHaveBeenCalled();
  await user.click(screen.getByRole("button", { name: "Approve" }));
  await user.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Approve" }));
  expect(post).toHaveBeenCalledWith(`/admin/events/${adminEvent.id}/approve`, { expectedRevision: 4 });
});
it("approving a price change names the old and new pricing", async () => {
  const change = { pricingMode: "paid" as const, pricePerPhoto: "120", watermarkPolicy: "own" as const, requestedAt: "2026-09-09T00:00:00Z" };
  client.setQueryData(key, [{ ...adminEvent, reviewStatus: "change_pending", pendingChange: change }]);
  const post = vi.spyOn(api, "post"); const user = userEvent.setup();
  render(<QueryClientProvider client={client}><EventRequestsQueue /></QueryClientProvider>);
  await user.click(screen.getByRole("button", { name: "Approve" }));
  const dialog = screen.getByRole("dialog", { name: "Apply this pricing change?" });
  expect(dialog).toHaveTextContent(describePricing({ pricingMode: "paid", pricePerPhoto: 80, watermarkPolicy: "platform" }));
  expect(dialog).toHaveTextContent(describePricing(change));
  expect(post).not.toHaveBeenCalled();
});
// ER3: the queue arrives whole (one request, oldest first), so the kind filter is client-side.
it("the kind filter splits new requests from changes and keeps it in the URL", async () => {
  client.setQueryData(key, [
    { ...adminEvent, id: "new-1", name: "New Race", status: "DRAFT", reviewStatus: "pending" },
    { ...adminEvent, id: "chg-1", name: "Changed Race", reviewStatus: "change_pending",
      pendingChange: { pricingMode: "free", pricePerPhoto: "0", watermarkPolicy: "platform", requestedAt: "2026-09-09T00:00:00Z" } },
  ]);
  const user = userEvent.setup();
  render(<QueryClientProvider client={client}><EventRequestsQueue /></QueryClientProvider>);
  const tabs = screen.getByRole("group", { name: "Request kind" });
  expect(within(tabs).getByRole("button", { name: "All 2" })).toHaveAttribute("aria-pressed", "true");
  await user.click(within(tabs).getByRole("button", { name: "Changes 1" }));
  expect(screen.queryByText("New Race")).not.toBeInTheDocument();
  expect(screen.getByText("Changed Race")).toBeInTheDocument();
  expect(nav.get()).toBe("kind=changes");
  await user.click(within(tabs).getByRole("button", { name: "New 1" }));
  expect(screen.getByText("New Race")).toBeInTheDocument();
  expect(screen.queryByText("Changed Race")).not.toBeInTheDocument();
});
it.each([[3, /Waiting 3 days/], [1, /Waiting 1 day\b/], [0, /Waiting since today/]])("a change request %i days old says how long it has waited", (days, text) => {
  const requestedAt = new Date(Date.now() - days * 86_400_000 - 60_000).toISOString();
  client.setQueryData(key, [{ ...adminEvent, reviewStatus: "change_pending",
    pendingChange: { pricingMode: "free", pricePerPhoto: "0", watermarkPolicy: "platform", requestedAt } }]);
  render(<QueryClientProvider client={client}><EventRequestsQueue /></QueryClientProvider>);
  expect(screen.getByText(text)).toBeInTheDocument();
});
it("a request time in the future or unreadable shows no age", () => {
  client.setQueryData(key, [{ ...adminEvent, reviewStatus: "change_pending",
    pendingChange: { pricingMode: "free", pricePerPhoto: "0", watermarkPolicy: "platform", requestedAt: "not a date" } }]);
  render(<QueryClientProvider client={client}><EventRequestsQueue /></QueryClientProvider>);
  expect(screen.queryByText(/Waiting/)).not.toBeInTheDocument();
});
it("a new request created three days ago says how long it has waited", () => {
  const createdAt = new Date(Date.now() - 3 * 86_400_000 - 60_000).toISOString();
  client.setQueryData(key, [{ ...adminEvent, reviewStatus: "pending", pendingChange: null, createdAt }]);
  render(<QueryClientProvider client={client}><EventRequestsQueue /></QueryClientProvider>);
  expect(screen.getByText(/Waiting 3 days/)).toBeInTheDocument();
});
it("a race sent again counts from its resend, not its first submission", () => {
  const createdAt = new Date(Date.now() - 10 * 86_400_000).toISOString();
  const resubmittedAt = new Date(Date.now() - 2 * 86_400_000 - 60_000).toISOString();
  client.setQueryData(key, [{ ...adminEvent, reviewStatus: "pending", pendingChange: null, createdAt, resubmittedAt }]);
  render(<QueryClientProvider client={client}><EventRequestsQueue /></QueryClientProvider>);
  expect(screen.getByText(/Waiting 2 days/)).toBeInTheDocument();
});
it("a race never sent again (resend time null) counts from its submission", () => {
  const createdAt = new Date(Date.now() - 3 * 86_400_000 - 60_000).toISOString();
  client.setQueryData(key, [{ ...adminEvent, reviewStatus: "pending", pendingChange: null, createdAt, resubmittedAt: null }]);
  render(<QueryClientProvider client={client}><EventRequestsQueue /></QueryClientProvider>);
  expect(screen.getByText(/Waiting 3 days/)).toBeInTheDocument();
});
it("a new request without a submission time (older backend) shows no age", () => {
  client.setQueryData(key, [{ ...adminEvent, reviewStatus: "pending", pendingChange: null, createdAt: undefined }]);
  render(<QueryClientProvider client={client}><EventRequestsQueue /></QueryClientProvider>);
  expect(screen.queryByText(/Waiting/)).not.toBeInTheDocument();
});
it("a change request counts from its request time, not the event's submission time", () => {
  const requestedAt = new Date(Date.now() - 86_400_000 - 60_000).toISOString();
  client.setQueryData(key, [{ ...adminEvent, reviewStatus: "change_pending", createdAt: "2026-01-01T00:00:00Z",
    resubmittedAt: "2026-02-01T00:00:00Z", pendingChange: { pricingMode: "free", pricePerPhoto: "0", watermarkPolicy: "platform", requestedAt } }]);
  render(<QueryClientProvider client={client}><EventRequestsQueue /></QueryClientProvider>);
  expect(screen.getByText(/Waiting 1 day\b/)).toBeInTheDocument();
});
it("approving a race edit names the changed fields", async () => {
  client.setQueryData(key, [{ ...adminEvent, eventType: "ROAD_RACE", reviewStatus: "change_pending",
    pendingChange: { kind: "race_edit", fields: { title: "Cebu City Marathon", date: "2026-09-11" }, coverS3Key: "events/x/cover.jpg", requestedBy: "u", requestedAt: "2026-09-09T00:00:00Z" } }]);
  render(<QueryClientProvider client={client}><EventRequestsQueue /></QueryClientProvider>);
  await userEvent.click(screen.getByRole("button", { name: "Approve" }));
  expect(screen.getByRole("dialog", { name: "Apply this race edit?" })).toHaveTextContent("Cebu Race takes the new title, date, cover photo.");
});
it("a failed approve keeps the confirm open and lets the admin try again", async () => {
  const post = vi.spyOn(api, "post").mockRejectedValueOnce(new Error("Unavailable")); const user = userEvent.setup();
  render(<QueryClientProvider client={client}><EventRequestsQueue /></QueryClientProvider>);
  await user.click(screen.getByRole("button", { name: "Approve" }));
  const dialog = screen.getByRole("dialog");
  await user.click(within(dialog).getByRole("button", { name: "Approve" }));
  await waitFor(() => expect(within(dialog).getByRole("button", { name: "Approve" })).toBeEnabled());
  expect(dialog).toBeInTheDocument(); expect(post).toHaveBeenCalledTimes(1);
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
  await user.click(screen.getByRole("button", { name: "Approve as a new race" }));
  const approveDialog = screen.getByRole("dialog", { name: "Approve as a new race?" });
  expect(approveDialog).toHaveTextContent("It stays its own race, apart from Cebu Marathon.");
  await user.click(within(approveDialog).getByRole("button", { name: "Cancel" }));
  expect(post).not.toHaveBeenCalled();
  await user.click(screen.getByRole("button", { name: "Same race as an existing one…" }));
  // Joining merges and removes the request, so nothing is sent until the admin confirms.
  expect(post).not.toHaveBeenCalled();
  expect(screen.getByText(/will cover/)).toHaveTextContent("Ana Cruz will cover Cebu Marathon");
  await user.click(screen.getByRole("button", { name: "Join race" }));
  expect(post).toHaveBeenCalledWith(`/admin/events/${race.id}/same-race`, { targetEventId: target.id, expectedRevision: 4 });
  expect(screen.getByRole("button", { name: "Approve as a new race" })).toBeDisabled();
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
it("join confirm sends the request's current revision, not the one it had when the modal opened", async () => {
  client.setQueryData(key, [race]); const user = userEvent.setup();
  const post = vi.spyOn(api, "post").mockResolvedValue(target);
  render(<QueryClientProvider client={client}><EventRequestsQueue /></QueryClientProvider>);
  await user.click(screen.getByRole("button", { name: "Same race as an existing one…" }));
  act(() => { client.setQueryData(key, [{ ...race, reviewRevision: 9 }]); });
  await user.click(screen.getByRole("button", { name: "Join race" }));
  expect(post).toHaveBeenCalledWith(`/admin/events/${race.id}/same-race`, { targetEventId: target.id, expectedRevision: 9 });
});
it("join confirm closes when the request is gone", async () => {
  client.setQueryData(key, [race]); const user = userEvent.setup();
  const post = vi.spyOn(api, "post");
  render(<QueryClientProvider client={client}><EventRequestsQueue /></QueryClientProvider>);
  await user.click(screen.getByRole("button", { name: "Same race as an existing one…" }));
  expect(screen.getByRole("dialog")).toBeInTheDocument();
  act(() => { client.setQueryData(key, []); });
  await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  expect(post).not.toHaveBeenCalled();
});

// AX-39 (UA7): row values read as typed; mono caps stay for labels only.
it("a request card's values are not upper-cased", () => {
  client.setQueryData(key, [{ ...adminEvent, status: "DRAFT", reviewStatus: "pending", createdByHandle: "anashoots" }]);
  render(<QueryClientProvider client={client}><EventRequestsQueue /></QueryClientProvider>);
  expect(screen.getByText(/· @anashoots/).closest(".uppercase")).toBeNull();
  expect(screen.getByText(/^Sep 10 · 2026 · /)).toBeInTheDocument();
  cleanup();
  client.setQueryData(key, [race]);
  render(<QueryClientProvider client={client}><EventRequestsQueue /></QueryClientProvider>);
  expect(screen.getByText(/· Sep 10 · 2026 · SRP, Cebu City · strong match/)).toBeInTheDocument();
});

it("an empty queue says no event requests are waiting", () => {
  client.setQueryData(key, []);
  render(<QueryClientProvider client={client}><EventRequestsQueue /></QueryClientProvider>);
  expect(screen.getByText("No event requests waiting. You're caught up.")).toBeInTheDocument();
});
