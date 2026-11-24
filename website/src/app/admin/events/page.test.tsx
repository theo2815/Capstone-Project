import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { webcrypto } from "node:crypto";
import { clearAdminActionIntents } from "@/lib/admin-action-intents";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api, ApiError } from "@/lib/api";
import { ConfirmationOverlay } from "@/components/ui/confirmation-overlay";
import { useConfirmationStore } from "@/store/confirmation-store";
import { useToastStore } from "@/store/toast-store";
import { useAuthStore } from "@/store/auth-store";
import { admin, adminEvent, eventRegions } from "@/test/fixtures/admin";
import { adminEventKeys } from "@/hooks/use-admin-events";
import AdminEventsPage from "./page";
import { useAdminEventActions } from "@/hooks/use-admin-events";
import { expectSectionIntro } from "@/test/admin-section-intro";
import { ROUTES } from "@/lib/constants";

// A tiny in-memory URL: the page keeps its status tab in ?status=.
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
    usePathname: () => "/admin/events",
    useSearchParams: () => new URLSearchParams(useSyncExternalStore(nav.subscribe, nav.get)),
    useRouter: () => ({ replace: (url: string) => nav.set(url.split("?")[1] ?? ""), push: vi.fn() }),
  };
});

let client: QueryClient;
let rows = [adminEvent];
const key = adminEventKeys.list(admin.id);
function mount() { render(<QueryClientProvider client={client}><AdminEventsPage /><ConfirmationOverlay /></QueryClientProvider>); }
// AX-21: the tab list comes from the server, so wait for its first page.
async function mountLoaded() { mount(); await waitFor(() => expect(client.isFetching()).toBe(0)); }
// The fake server pages each tab and counts every state, as GET /admin/events and /admin/events/counts do.
function serve(path: string) {
  if (path === "/regions") return eventRegions;
  const p = new URLSearchParams(path.split("?")[1] ?? "");
  const q = (p.get("q") ?? "").toLowerCase();
  const found = rows.filter((row) => row.name.toLowerCase().includes(q));
  if (path.startsWith("/admin/events/counts")) {
    return { counts: Object.fromEntries(["upcoming", "live", "open", "past"].map((s) => [s, found.filter((row) => row.state === s).length])) };
  }
  const state = p.get("state"); const offset = Number(p.get("offset") ?? 0); const limit = Number(p.get("limit") ?? 200);
  const all = found.filter((row) => !state || row.state === state);
  return { items: all.slice(offset, offset + limit), total: all.length, offset, limit };
}
function deferred() {
  let resolve!: (value: unknown) => void; let reject!: (error: Error) => void;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
beforeEach(() => {
  vi.stubGlobal("crypto", webcrypto); clearAdminActionIntents();
  useAuthStore.getState().setUser(admin); rows = [adminEvent];
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } }); client.setQueryData(key, rows);
  client.setQueryData(["regions"], eventRegions);
  vi.spyOn(api, "get").mockImplementation(async (path) => serve(path));
});
afterEach(() => {
  nav.set("");
  clearAdminActionIntents(); vi.unstubAllGlobals();
  cleanup(); client.clear(); useAuthStore.getState().logout();
  useConfirmationStore.getState().close(useConfirmationStore.getState().active?.id ?? "", false); useToastStore.getState().clear(); vi.restoreAllMocks();
});
it("waits in delete confirmation, retains the event on failure, and allows retry", async () => {
  const user = userEvent.setup(); const request = deferred();
  const remove = vi.spyOn(api, "delete").mockReturnValueOnce(request.promise); await mountLoaded();
  await user.click(screen.getByRole("button", { name: /delete/i }));
  const dialog = screen.getByRole("dialog", { name: "Delete event?" });
  const submit = within(dialog).getByRole("button", { name: "Delete event" });
  await user.click(submit); await user.click(submit);
  expect(screen.getByRole("button", { name: "Edit Cebu Race" })).toBeDisabled();
  expect(submit).toBeDisabled(); expect(remove).toHaveBeenCalledTimes(1);
  expect(client.getQueryData(key)).toEqual([adminEvent]); expect(useToastStore.getState().toasts).toEqual([]);
  await act(async () => { request.reject(new Error("Event is still in use.")); });
  expect(dialog).toBeInTheDocument(); expect(within(dialog).getByRole("alert")).toHaveTextContent("Event is still in use.");
  expect(submit).toBeEnabled(); expect(useToastStore.getState().toasts.map((t) => t.kind)).toEqual(["error"]);
  remove.mockImplementationOnce(async () => { rows = []; return { removed: true }; });
  await user.click(submit); await waitFor(() => expect(dialog).not.toBeInTheDocument());
  expect(screen.queryByText("Cebu Race")).not.toBeInTheDocument(); expect(client.getQueryData(key)).toEqual([]);
  expect(useToastStore.getState().toasts.map((t) => t.kind)).toEqual(["error", "success"]);
});
it.each(["create", "edit"])("retains %s form input and canonical rows after failure, then retries", async (mode) => {
  const user = userEvent.setup(); const request = deferred();
  const mutation = vi.spyOn(api, mode === "create" ? "post" : "fetch").mockReturnValueOnce(request.promise); await mountLoaded();
  await user.click(screen.getByRole("button", { name: mode === "create" ? /new event/i : "Edit Cebu Race" }));
  const dialog = screen.getByRole("dialog", { name: mode === "create" ? "Create event" : "Edit event" });
  const title = within(dialog).getByLabelText(/^Title/); await user.clear(title); await user.type(title, "Draft title");
  if (mode === "create") {
    await user.type(within(dialog).getByLabelText(/^Location/), "Cebu");
    fireEvent.change(within(dialog).getByLabelText(/^Date/), { target: { value: adminEvent.date } });
    await user.clear(within(dialog).getByLabelText(/^Price per photo · PHP/));
    await user.type(within(dialog).getByLabelText(/^Price per photo · PHP/), "80");
    await user.click(within(dialog).getByRole("button", { name: /^Region:/ }));
    await user.click(screen.getByRole("menuitem", { name: eventRegions[0].name }));
    await user.click(within(dialog).getByRole("button", { name: /^Province:/ }));
    await user.click(screen.getByRole("menuitem", { name: "Cebu" }));
  }
  const submitLabel = mode === "create" ? "Create event" : "Save changes";
  await user.click(within(dialog).getByRole("button", { name: submitLabel }));
  expect(within(dialog).getByRole("button", { name: mode === "create" ? "Creating…" : "Saving…" })).toHaveAttribute("aria-busy", "true");
  expect(title).toBeDisabled(); expect(within(dialog).getByRole("button", { name: "Cancel" })).toBeDisabled();
  await user.keyboard("{Escape}{Enter}"); expect(mutation).toHaveBeenCalledTimes(1);
  expect(client.getQueryData(key)).toEqual([adminEvent]);
  await act(async () => { request.reject(new Error("Try again")); });
  expect(title).toHaveValue("Draft title"); expect(title).toBeEnabled(); expect(dialog).toBeInTheDocument();
  await user.click(within(dialog).getByRole("button", { name: "Cancel" }));
  await user.click(screen.getByRole("button", { name: mode === "create" ? /new event/i : "Edit Cebu Race" }));
  expect(within(screen.getByRole("dialog")).getByLabelText(/^Title/)).toHaveValue("Draft title");
  expect(screen.getByRole("status")).toHaveTextContent(/unresolved|unchanged|original/i);
  const confirmed = { ...adminEvent, name: "Backend title", slug: "backend-title" };
  mutation.mockImplementationOnce(async () => { rows = [confirmed]; return confirmed; });
  await user.click(within(screen.getByRole("dialog")).getByRole("button", { name: submitLabel }));
  await waitFor(() => expect(dialog).not.toBeInTheDocument());
  expect(client.getQueryData(key)).toEqual([confirmed]);
  await waitFor(() => expect(screen.getByRole("link", { name: /view/i })).toHaveAttribute("href", "/events/backend-title"));
  if (mode === "edit") {
    const form = (mutation.mock.calls[0][1] as RequestInit).body as FormData;
    expect(form.get("title")).toBe("Draft title"); expect(form.has("pricePerPhoto")).toBe(false);
    expect(form.get("expectedRevision")).toBe("4");
    expect(form.has("regionCode")).toBe(false); expect(form.has("provinceCode")).toBe(false);
  } else {
    const form = mutation.mock.calls[0][1] as FormData;
    expect(form.get("regionCode")).toBe("region-vii"); expect(form.get("provinceCode")).toBe("cebu");
  }
});
it("shows loading and a retryable query error instead of an empty catalog", async () => {
  client.clear(); let failed = false;
  vi.mocked(api.get).mockImplementation(async (path) => {
    if (path === "/regions") return eventRegions;
    if (!failed && String(path).startsWith("/admin/events?")) { failed = true; throw new Error("Offline"); }
    return serve(path);
  }); mount();
  // EV5: a skeleton list, like Photographers and Event requests, not a bare line.
  expect(document.querySelector('[aria-busy="true"] .animate-pulse')).toBeInTheDocument();
  expect(screen.queryByText("Loading events…")).not.toBeInTheDocument();
  await screen.findByRole("alert"); expect(screen.queryByText("No live events right now.")).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Retry" }));
  await screen.findByRole("button", { name: "Edit Cebu Race" });
});
it("identifies a lost edit snapshot after a committed response is lost and the editor remounts", async () => {
  const update = vi.spyOn(api, "fetch").mockImplementationOnce(async () => {
    rows = [{ ...adminEvent, name: "Committed draft", reviewRevision: 5 }];
    throw new Error("Response lost after commit");
  });
  await mountLoaded(); const user = userEvent.setup();
  await user.click(screen.getByRole("button", { name: "Edit Cebu Race" }));
  await user.clear(screen.getByLabelText(/^Title/)); await user.type(screen.getByLabelText(/^Title/), "Committed draft");
  await user.click(screen.getByRole("button", { name: "Save changes" }));
  await waitFor(() => expect(screen.getByRole("button", { name: "Save changes" })).toBeEnabled());
  const reference = (update.mock.calls[0][1]?.headers as Record<string, string>)["Idempotency-Key"];
  cleanup(); client.removeQueries({ queryKey: adminEventKeys.all }); await mountLoaded();
  await user.click(screen.getByRole("button", { name: "Edit Committed draft" }));
  const dialog = screen.getByRole("dialog");
  expect(dialog).toHaveTextContent(/original edit snapshot/i);
  expect(dialog).toHaveTextContent(/reconcil/i); expect(dialog).toHaveTextContent(reference);
  await user.click(within(dialog).getByRole("button", { name: "Save changes" }));
  await waitFor(() => expect(useToastStore.getState().toasts.at(-1)?.message).toMatch(/reconcil/i));
  expect(update).toHaveBeenCalledTimes(1);
  expect(dialog).toBeInTheDocument();
});

it("retains changed geography and cover after a conflict refresh, then sends the original snapshot on retry", async () => {
  const user = userEvent.setup();
  rows = [{ ...adminEvent, regionCode: "region-vii", provinceCode: "cebu" }]; client.setQueryData(key, rows);
  const update = vi.spyOn(api, "fetch").mockRejectedValueOnce(new ApiError([{ code: "CONFLICT", field: "expectedRevision", message: "Event changed" }], 409));
  const objectUrl = vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:cover");
  vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
  await mountLoaded(); await user.click(screen.getByRole("button", { name: "Edit Cebu Race" }));
  const dialog = screen.getByRole("dialog");
  await user.click(within(dialog).getByRole("button", { name: /^Region:/ }));
  await user.click(screen.getByRole("menuitem", { name: eventRegions[1].name }));
  await user.click(within(dialog).getByRole("button", { name: /^Province:/ }));
  await user.click(screen.getByRole("menuitem", { name: "Leyte" }));
  await user.upload(within(dialog).getByLabelText("Upload cover"), new File(["cover"], "cover.jpg", { type: "image/jpeg" }));
  rows = [{ ...rows[0], name: "Other admin", reviewRevision: 5 }];
  await user.click(within(dialog).getByRole("button", { name: "Save changes" }));
  await waitFor(() => expect(within(dialog).getByRole("button", { name: "Save changes" })).toBeEnabled());
  await screen.findByText("Other admin");
  expect(within(dialog).getByLabelText(/^Title/)).toHaveValue("Cebu Race");
  expect(within(dialog).getByRole("button", { name: "Province: Leyte" })).toBeInTheDocument();
  expect(within(dialog).getByAltText("Cover preview")).toHaveAttribute("src", "blob:cover");
  expect(useToastStore.getState().toasts.at(-1)?.message).toMatch(/event changed.*reopen the event/i);
  const confirmed = { ...rows[0], regionCode: "region-viii", provinceCode: "leyte" };
  update.mockResolvedValue(confirmed);
  await user.click(within(dialog).getByRole("button", { name: "Save changes" }));
  await waitFor(() => expect(dialog).not.toBeInTheDocument());
  for (const [, options] of update.mock.calls) {
    const form = options?.body as FormData;
    expect(form.get("regionCode")).toBe("region-viii"); expect(form.get("provinceCode")).toBe("leyte");
    expect(form.get("cover")).toBeInstanceOf(File); expect(form.get("expectedRevision")).toBe("4");
  }
  // A refused stale edit is released, so the next save is a new request.
  expect(update.mock.calls[0][1]?.headers).not.toEqual(update.mock.calls[1][1]?.headers);
  expect(objectUrl).toHaveBeenCalledOnce();
});
it("keeps a delete confirmation open when another view owns the event", async () => {
  const request = deferred(); vi.spyOn(api, "post").mockReturnValue(request.promise);
  const remove = vi.spyOn(api, "delete");
  let approve!: () => Promise<boolean>;
  function OtherView() { const actions = useAdminEventActions(); approve = () => actions.approve(adminEvent.id, 4); return null; }
  render(<QueryClientProvider client={client}><AdminEventsPage /><OtherView /><ConfirmationOverlay /></QueryClientProvider>);
  await waitFor(() => expect(client.isFetching()).toBe(0));
  await userEvent.click(screen.getByRole("button", { name: /delete/i }));
  let command!: Promise<boolean>; act(() => { command = approve(); });
  const dialog = screen.getByRole("dialog");
  await userEvent.click(within(dialog).getByRole("button", { name: "Delete event" }));
  await waitFor(() => expect(within(dialog).getByRole("alert")).toHaveTextContent(/another|pending/i));
  expect(dialog).toBeInTheDocument(); expect(remove).not.toHaveBeenCalled();
  await act(async () => { request.resolve(adminEvent); await command; });
});
// "Show on /events" (V89): a live Public photographer event can be put on the public list.
const owned = { ...adminEvent, id: "owned", slug: "nrw-fun-run", name: "NRW Fun Run", createdByHandle: "quickpitikfree", createdByName: "QuickPitik Free" };
it("shows the /events switch only on a live public photographer event", async () => {
  rows = [adminEvent, { ...owned, id: "private", slug: "private", name: "Private Run", visibility: "unlisted" as const },
    { ...owned, id: "draft", slug: "draft", name: "Draft Run", status: "DRAFT" as const, reviewStatus: "pending" as const }, owned];
  client.setQueryData(key, rows); await mountLoaded();
  expect(screen.getAllByRole("checkbox", { name: "Show on /events: NRW Fun Run" })).toHaveLength(1);
  expect(screen.getByRole("checkbox", { name: "Show on /events: NRW Fun Run" }).closest("article")).toHaveTextContent("@quickpitikfree");
});
it("lists a photographer event from the server's answer and refreshes the public catalog", async () => {
  rows = [owned]; client.setQueryData(key, rows); client.setQueryData(["public", "events", "catalog"], []);
  const request = deferred(); const patch = vi.spyOn(api, "fetch").mockReturnValueOnce(request.promise); await mountLoaded();
  const box = screen.getByRole("checkbox", { name: "Show on /events: NRW Fun Run" });
  await userEvent.click(box);
  // EV5: it asks first, as Relist does, and the box waits for the answer.
  expect(patch).not.toHaveBeenCalled(); expect(box).not.toBeChecked();
  const dialog = screen.getByRole("dialog", { name: "Show this event on /events?" });
  await userEvent.click(within(dialog).getByRole("button", { name: "Show on /events" }));
  expect(patch).toHaveBeenCalledWith(`/admin/events/${owned.id}/listing`, { method: "PATCH", body: JSON.stringify({ listed: true }) });
  expect(box).toBeDisabled();
  await act(async () => { rows = [{ ...owned, listedOnEvents: true }]; request.resolve(rows[0]); });
  await waitFor(() => expect(screen.getByRole("checkbox", { name: "Show on /events: NRW Fun Run" })).toBeChecked());
  expect(client.getQueryState(["public", "events", "catalog"])?.isInvalidated).toBe(true);
});
it("keeps the switch as the server has it when listing fails", async () => {
  rows = [owned]; client.setQueryData(key, rows);
  vi.spyOn(api, "fetch").mockRejectedValueOnce(new ApiError([{ code: "VALIDATION_ERROR", message: "This event is Private. The photographer must make it Public first." }], 400));
  await mountLoaded();
  await userEvent.click(screen.getByRole("checkbox", { name: "Show on /events: NRW Fun Run" }));
  const dialog = screen.getByRole("dialog", { name: "Show this event on /events?" });
  await userEvent.click(within(dialog).getByRole("button", { name: "Show on /events" }));
  await waitFor(() => expect(within(dialog).getByRole("alert")).toHaveTextContent("must make it Public first"));
  await waitFor(() => expect(useToastStore.getState().toasts).toEqual([expect.objectContaining({ kind: "error", message: "This event is Private. The photographer must make it Public first." })]));
  expect(screen.getByRole("checkbox", { name: "Show on /events: NRW Fun Run" })).not.toBeChecked();
  expect(screen.getByRole("checkbox", { name: "Show on /events: NRW Fun Run" })).toBeEnabled();
});
// AA-R17b: a hidden shared race is relisted through the listing switch.
it("relist waits for the API before the chip clears", async () => {
  const hidden = { ...adminEvent, eventType: "ROAD_RACE", hiddenAt: "2026-09-12T00:00:00Z" };
  rows = [hidden]; client.setQueryData(key, rows);
  const request = deferred(); const patch = vi.spyOn(api, "fetch").mockReturnValueOnce(request.promise); await mountLoaded();
  expect(screen.getByText("Hidden")).toBeInTheDocument();
  expect(screen.getByText(/link still opens/)).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Relist Cebu Race" }));
  await userEvent.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Relist race" }));
  expect(patch).toHaveBeenCalledWith(`/admin/events/${hidden.id}/listing`, { method: "PATCH", body: JSON.stringify({ listed: true }) });
  expect(screen.getByText("Hidden")).toBeInTheDocument();
  await act(async () => { rows = [{ ...hidden, hiddenAt: null, adminApproved: true }]; request.resolve(rows[0]); });
  await waitFor(() => expect(screen.queryByText("Hidden")).not.toBeInTheDocument());
});
it("a race still waiting for review offers no Hide or Relist", async () => {
  rows = [{ ...adminEvent, eventType: "ROAD_RACE", status: "DRAFT", reviewStatus: "pending", hiddenAt: "2026-09-12T00:00:00Z" }];
  client.setQueryData(key, rows); await mountLoaded();
  expect(screen.queryByRole("button", { name: /^(Hide|Relist) / })).not.toBeInTheDocument();
});
it("pages a tab 50 at a time: 51 events show Load more, and it appends (AX-21)", async () => {
  rows = Array.from({ length: 51 }, (_, i) => ({ ...adminEvent, id: `e${i}`, slug: `race-${i}`, name: `Race ${i}` }));
  client.removeQueries({ queryKey: key }); await mountLoaded();
  expect(screen.getAllByRole("button", { name: /^Edit Race/ })).toHaveLength(50);
  await userEvent.click(screen.getByRole("button", { name: "Load 1 more" }));
  await screen.findByRole("button", { name: "Edit Race 50" });
  expect(screen.getAllByRole("button", { name: /^Edit Race/ })).toHaveLength(51);
});
it("searches on the server and counts the tabs for that search (AX-21)", async () => {
  rows = [adminEvent, { ...adminEvent, id: "e2", slug: "mandaue", name: "Mandaue Run" }];
  await mountLoaded();
  expect(screen.getByRole("button", { name: "Live now 2", pressed: true })).toBeInTheDocument();
  await userEvent.type(screen.getByRole("searchbox", { name: "Search events" }), "mandaue");
  await waitFor(() => expect(screen.queryByRole("button", { name: "Edit Cebu Race" })).not.toBeInTheDocument());
  expect(api.get).toHaveBeenCalledWith("/admin/events?state=live&q=mandaue&offset=0&limit=50", expect.anything());
  expect(api.get).toHaveBeenCalledWith("/admin/events/counts?q=mandaue", expect.anything());
  expect(screen.getByRole("button", { name: "Edit Mandaue Run" })).toBeInTheDocument();
  await waitFor(() => expect(screen.getByRole("button", { name: "Live now 1", pressed: true })).toBeInTheDocument());
});
it("does not close the delete confirm while the deleted event is still listed (AX-21)", async () => {
  await mountLoaded(); const refetch = deferred();
  vi.spyOn(api, "delete").mockImplementationOnce(async () => {
    rows = [];
    vi.mocked(api.get).mockImplementation(async (path) => { await refetch.promise; return serve(path); });
    return { removed: true };
  });
  await userEvent.click(screen.getByRole("button", { name: /delete/i }));
  const dialog = screen.getByRole("dialog", { name: "Delete event?" });
  await userEvent.click(within(dialog).getByRole("button", { name: "Delete event" }));
  await act(async () => { await new Promise((r) => setTimeout(r, 50)); });
  // A slow list refresh: closed-with-the-row-still-there is the bug.
  expect(!dialog.isConnected && !!screen.queryByText("Cebu Race")).toBe(false);
  await act(async () => { refetch.resolve(undefined); });
  await waitFor(() => expect(dialog).not.toBeInTheDocument());
  expect(screen.queryByText("Cebu Race")).not.toBeInTheDocument();
});
it("keeps the count in the tab and does not repeat it under the tabs (EV6)", async () => {
  await mountLoaded();
  expect(screen.getByRole("button", { name: "Live now 1", pressed: true })).toBeInTheDocument();
  expect(screen.queryByText("1 event")).not.toBeInTheDocument();
});
it("a search link with no tab opens the first tab that holds the match (FT-4)", async () => {
  rows = [{ ...adminEvent, id: "e2", slug: "mandaue", name: "Mandaue Run", state: "open" }];
  nav.set("q=Mandaue%20Run");
  await mountLoaded();
  await waitFor(() => expect(screen.getByRole("button", { name: "Recent 1", pressed: true })).toBeInTheDocument());
  expect(new URLSearchParams(nav.get()).get("status")).toBe("recent");
  // An empty Live tab can still be opened: the switch happens once.
  await userEvent.click(screen.getByRole("button", { name: /^Live now/ }));
  await act(async () => { await new Promise((r) => setTimeout(r, 50)); });
  expect(screen.getByRole("button", { name: /^Live now/, pressed: true })).toBeInTheDocument();
});
it("a search link that names a tab keeps that tab (FT-4)", async () => {
  rows = [{ ...adminEvent, state: "open" }];
  nav.set("status=live&q=Cebu");
  await mountLoaded();
  expect(screen.getByRole("button", { name: /^Live now/, pressed: true })).toBeInTheDocument();
  expect(new URLSearchParams(nav.get()).get("status")).toBe("live");
});

it("shows its guide intro, look-alike links and definitions without hover", () => {
  mount();
  expectSectionIntro(ROUTES.ADMIN_EVENTS);
});

it("the delete dialog names what deleting does and that it cannot be undone (FT-15, B-30)", async () => {
  await mountLoaded();
  await userEvent.click(screen.getByRole("button", { name: /delete/i }));
  const dialog = screen.getByRole("dialog", { name: "Delete event?" });
  expect(dialog).toHaveTextContent("Removes Cebu Race from the public events list, the photographer upload picker and this page. Its photos can no longer be bought. Runners who already paid can still download. Nobody is told. This cannot be undone from the dashboard.");
});
it("a row in the Recent tab reads Recent, its tab name (FT-15, B-31)", async () => {
  rows = [{ ...adminEvent, state: "open" }];
  nav.set("status=recent");
  await mountLoaded();
  expect(screen.getByRole("article")).toHaveTextContent("Recent");
  expect(screen.queryByText("Open for sale")).not.toBeInTheDocument();
});

// FT-22 (R-02): after a create the page opens the tab the server filed the event under.
async function createFromForm(name: string, date: string) {
  const user = userEvent.setup();
  await user.click(screen.getByRole("button", { name: /new event/i }));
  const dialog = screen.getByRole("dialog", { name: "Create event" });
  await user.type(within(dialog).getByLabelText(/^Title/), name);
  await user.type(within(dialog).getByLabelText(/^Location/), "Cebu");
  fireEvent.change(within(dialog).getByLabelText(/^Date/), { target: { value: date } });
  await user.type(within(dialog).getByLabelText(/^Price per photo · PHP/), "80");
  await user.click(within(dialog).getByRole("button", { name: /^Region:/ }));
  await user.click(screen.getByRole("menuitem", { name: eventRegions[0].name }));
  await user.click(within(dialog).getByRole("button", { name: /^Province:/ }));
  await user.click(screen.getByRole("menuitem", { name: "Cebu" }));
  await user.click(within(dialog).getByRole("button", { name: "Create event" }));
  await waitFor(() => expect(dialog).not.toBeInTheDocument());
}
it("creating an event dated next month opens Upcoming and the toast names the tab (FT-22, R-02)", async () => {
  const next = new Date(); next.setMonth(next.getMonth() + 1);
  const date = next.toISOString().slice(0, 10);
  const created = { ...adminEvent, id: "e-new", slug: "next-race", name: "Next Race", date, state: "upcoming" as const };
  vi.spyOn(api, "post").mockImplementation(async () => { rows = [...rows, created]; return created; });
  nav.set("q=Cebu");
  await mountLoaded();
  await createFromForm("Next Race", date);
  await waitFor(() => expect(screen.getByRole("button", { name: /^Upcoming/, pressed: true })).toBeInTheDocument());
  expect(new URLSearchParams(nav.get()).get("status")).toBe("upcoming");
  await waitFor(() => expect(new URLSearchParams(nav.get()).has("q")).toBe(false));
  expect(await screen.findByText("Next Race")).toBeInTheDocument();
  expect(useToastStore.getState().toasts.at(-1)?.message).toBe("Created Next Race. It is under Upcoming.");
});
it("a past-dated event opens the tab the server put it under, not a guess from its date (FT-22)", async () => {
  const past = new Date(); past.setDate(past.getDate() - 2);
  const date = past.toISOString().slice(0, 10);
  const created = { ...adminEvent, id: "e-past", slug: "past-race", name: "Past Race", date, state: "open" as const };
  vi.spyOn(api, "post").mockImplementation(async () => { rows = [...rows, created]; return created; });
  await mountLoaded();
  await createFromForm("Past Race", date);
  await waitFor(() => expect(screen.getByRole("button", { name: /^Recent/, pressed: true })).toBeInTheDocument());
  expect(useToastStore.getState().toasts.at(-1)?.message).toBe("Created Past Race. It is under Recent.");
});
it("with no event in any tab the empty text says how to add the first one (FT-22, B-29)", async () => {
  rows = []; client.setQueryData(key, rows);
  await mountLoaded();
  expect(await screen.findByText("No events yet. Use + New event to add the first one.")).toBeInTheDocument();
  expect(screen.queryByText("No live events right now.")).not.toBeInTheDocument();
});
it("an empty tab keeps its own text while another tab holds an event (FT-22)", async () => {
  rows = [{ ...adminEvent, state: "upcoming" }]; client.setQueryData(key, rows);
  await mountLoaded();
  expect(await screen.findByText("No live events right now.")).toBeInTheDocument();
  expect(screen.queryByText(/No events yet/)).not.toBeInTheDocument();
});

it("each tab's caption says the server's rule: Live is race day only, Archive is archived events (FU-4, U4.1 to U4.4)", async () => {
  await mountLoaded();
  const captions: Array<[RegExp, string]> = [
    [/^Live now/, "Race day"],
    [/^Upcoming/, "Race day ahead, or not published yet"],
    [/^Recent/, "After race day · photos on sale"],
    [/^Archive/, "Archived events"],
  ];
  for (const [tab, caption] of captions) {
    await userEvent.click(screen.getByRole("button", { name: tab }));
    expect(await screen.findByText(caption, { selector: "p" })).toBeInTheDocument();
  }
  for (const old of ["Currently accepting uploads · 4-day grace from race day", "Race day pending", "Galleries open for sale · upload window closed", "Read-only history"]) {
    expect(screen.queryByText(old)).not.toBeInTheDocument();
  }
});
