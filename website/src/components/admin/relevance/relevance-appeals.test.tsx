import { act, cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api, ApiError } from "@/lib/api";
import type { RelevanceAppealRow, RelevanceCaseDetail, RelevanceCaseRow } from "@/lib/admin-relevance";
import { admin } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import { useToastStore } from "@/store/toast-store";
import { RelevanceCases } from "./relevance-cases";

// RV-3.5: the Appeals tab. An admin reads the reason, opens the samples (recorded once), then decides.
const fixtures = vi.hoisted(() => ({ search: "", router: { replace: vi.fn() } }));
vi.mock("next/navigation", () => ({
  useRouter: () => fixtures.router,
  usePathname: () => "/admin/relevance",
  useSearchParams: () => new URLSearchParams(fixtures.search),
}));

const caseRow: RelevanceCaseRow = {
  caseId: "c1", eventId: "e1", eventName: "Cebu City Marathon 2003", photographerHandle: "juanshots",
  photographerName: "Juan Cruz", verdict: "JUNK_DUMP", verdictNote: null, action: "HIDE_ALL", appliedMode: "AUTO_HIDE",
  wouldHave: false, batchPhotos: 420, junkPercent: 81, decidedAt: "2003-02-14T03:00:00Z",
};
const detail: RelevanceCaseDetail = {
  case: caseRow, stage: "APPLIED", sampleSize: 300, measured: 300, blurry: 100, dark: 10, notRace: 10, junk: 120, band: "OVER",
  modelVerdict: "JUNK_DUMP", adminReason: null, evidenceUsed: [], summarySent: null, model: null, promptVersion: null, wordsPhotos: 0,
  topWords: [], alerts: [], cleanSamples: [], confirmedAt: null, photographerStrikes: 0, payoutHoldAt: null, payoutHoldPlaced: false, closeReason: null,
  junkSamples: [{ photoId: "p1", thumbnailUrl: null, fgSharpness: 2, fgBrightness: 40, sharpness: 2, brightness: 40, blurry: true, dark: false, notRace: false }],
};
const openAppeal: RelevanceAppealRow = {
  appealId: "a1", caseId: "c1", batchId: "b1", eventId: "e1", eventName: "Cebu City Marathon 2003", photographerName: "Juan Cruz",
  photographerHandle: "juanshots", verdict: "JUNK_DUMP", verdictNote: null, action: "HIDE_ALL", caseStage: "APPLIED", status: "OPEN",
  reason: "<b>These are my race photos</b>\nfrom km 21", createdAt: "2003-02-16T00:00:00Z", evidenceOpenedAt: null, evidenceOpenedBy: null,
  decidedAt: null, decidedBy: null, decisionNote: null,
};

let lists: Record<string, RelevanceAppealRow[]>;
let post: (url: string, body?: unknown) => Promise<unknown>;
let client: QueryClient;
beforeEach(() => {
  fixtures.search = "tab=appeals&row=a1";
  useAuthStore.getState().setUser(admin);
  useToastStore.getState().clear();
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  lists = { OPEN: [openAppeal], WON: [], LOST: [] };
  post = async (url) => (url.endsWith("/evidence-opened") ? { ...openAppeal, evidenceOpenedAt: "2003-02-17T00:00:00Z", evidenceOpenedBy: admin.id } : {});
  vi.spyOn(api, "get").mockImplementation(((url: string) => {
    const status = url.match(/^\/admin\/relevance\/appeals\?status=(\w+)/)?.[1];
    if (status) return Promise.resolve({ items: lists[status], total: lists[status].length, offset: 0, limit: 50 });
    if (url.startsWith("/admin/relevance/cases/c1")) return Promise.resolve(url.includes("/hidden-photos") ? { items: [], total: 0, offset: 0, limit: 100 } : detail);
    return Promise.resolve({ items: [], total: 0, offset: 0, limit: 50 });
  }) as typeof api.get);
  vi.spyOn(api, "post").mockImplementation(((url: string, body?: unknown) => post(url, body)) as typeof api.post);
});
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); vi.restoreAllMocks(); });
const mount = () => render(<QueryClientProvider client={client}><RelevanceCases /></QueryClientProvider>);
const drawer = () => screen.findByRole("dialog", { name: "Cebu City Marathon 2003" });
const toasts = () => useToastStore.getState().toasts.map((t) => t.message);
const posts = (suffix: string) => vi.mocked(api.post).mock.calls.filter(([url]) => String(url).endsWith(suffix));

it("lists open appeals first and shows the open count on the tab", async () => {
  lists.WON = [{ ...openAppeal, appealId: "a2", status: "WON", eventName: "Won Race" }];
  fixtures.search = "tab=appeals";
  mount();
  const open = await screen.findByRole("region", { name: "Open" });
  expect(await within(open).findByText("Cebu City Marathon 2003")).toBeInTheDocument();
  expect(await within(screen.getByRole("region", { name: "Won" })).findByText("Won Race")).toBeInTheDocument();
  const sections = screen.getAllByRole("region").map((r) => r.getAttribute("aria-label"));
  expect(sections).toEqual(["Open", "Won", "Lost"]);
  expect(screen.getByRole("button", { name: /Appeals/ })).toHaveTextContent("Appeals 1");
});

it("the drawer shows the photographer's reason as plain text beside the case evidence", async () => {
  mount();
  const d = await drawer();
  const reason = await within(d).findByText(/These are my race photos/);
  expect(reason.textContent).toBe("<b>These are my race photos</b>\nfrom km 21");
  expect(d.querySelector("b")).toBeNull();
  expect(await within(d).findByText("Junk dump", { selector: "p" })).toBeInTheDocument();
});

it("the decide buttons are disabled until the evidence was opened", async () => {
  const user = userEvent.setup();
  mount();
  const d = await drawer();
  const won = await within(d).findByRole("button", { name: "Won" });
  expect(won).toBeDisabled();
  expect(within(d).getByRole("button", { name: "Lost" })).toBeDisabled();
  expect(within(d).queryByText("Junk samples")).not.toBeInTheDocument();
  await user.click(within(d).getByRole("button", { name: "Open the samples" }));
  await waitFor(() => expect(won).toBeEnabled());
  expect(within(d).getByRole("button", { name: "Lost" })).toBeEnabled();
  expect(within(d).getByText("Junk samples")).toBeInTheDocument();
});

it("opening the evidence is recorded once", async () => {
  mount();
  const d = await drawer();
  const open = await within(d).findByRole("button", { name: "Open the samples" });
  act(() => { open.click(); open.click(); });
  await within(d).findByText("Junk samples");
  expect(posts("/evidence-opened")).toEqual([["/admin/relevance/appeals/a1/evidence-opened"]]);
  expect(within(d).queryByRole("button", { name: "Open the samples" })).not.toBeInTheDocument();
});

it("a decided appeal leaves the open list", async () => {
  const user = userEvent.setup();
  post = async (url) => {
    if (url.endsWith("/decide")) { lists = { OPEN: [], WON: [{ ...openAppeal, status: "WON" }], LOST: [] }; return { ...openAppeal, status: "WON" }; }
    return { ...openAppeal, evidenceOpenedBy: admin.id };
  };
  mount();
  const d = await drawer();
  await user.click(await within(d).findByRole("button", { name: "Open the samples" }));
  await user.type(within(d).getByLabelText("Note (admins only, optional)"), "  Real race photos  ");
  await user.click(await within(d).findByRole("button", { name: "Won" }));
  await user.click(await screen.findByRole("button", { name: "Confirm won" }));
  expect(posts("/decide")).toEqual([["/admin/relevance/appeals/a1/decide", { outcome: "WON", note: "Real race photos" }]]);
  await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  expect(await within(screen.getByRole("region", { name: "Open" })).findByText("No open appeals.")).toBeInTheDocument();
  expect(toasts()).toContain("Appeal won: Cebu City Marathon 2003");
});

it("the server's refusal is shown", async () => {
  const user = userEvent.setup();
  post = async (url) => {
    if (url.endsWith("/decide")) throw new ApiError([{ code: "EVIDENCE_NOT_OPENED", message: "Open this appeal's evidence before deciding it" }], 409);
    return { ...openAppeal, evidenceOpenedBy: admin.id };
  };
  mount();
  const d = await drawer();
  await user.click(await within(d).findByRole("button", { name: "Open the samples" }));
  await user.click(await within(d).findByRole("button", { name: "Lost" }));
  await user.click(await screen.findByRole("button", { name: "Confirm lost" }));
  await waitFor(() => expect(toasts()).toContain("Open this appeal's evidence before deciding it"));
  expect(screen.getByRole("dialog", { name: "Cebu City Marathon 2003" })).toBeInTheDocument();
  expect(within(screen.getByRole("region", { name: "Open" })).getByText("Cebu City Marathon 2003")).toBeInTheDocument();
});

it("a refused evidence look keeps the decide buttons locked and shows the server's message", async () => {
  const user = userEvent.setup();
  post = async () => { throw new ApiError([{ code: "RELEVANCE_ACTIONS_OFF", message: "Deciding appeals needs the relevance mode auto-hide" }], 409); };
  mount();
  const d = await drawer();
  await user.click(await within(d).findByRole("button", { name: "Open the samples" }));
  await waitFor(() => expect(toasts()).toContain("Deciding appeals needs the relevance mode auto-hide"));
  expect(within(d).getByRole("button", { name: "Won" })).toBeDisabled();
});

it("Won and Lost send nothing until the confirm, and cancelling keeps the note", async () => {
  const user = userEvent.setup();
  mount();
  const d = await drawer();
  await user.click(await within(d).findByRole("button", { name: "Open the samples" }));
  await user.type(within(d).getByLabelText("Note (admins only, optional)"), "Checked");
  await user.click(await within(d).findByRole("button", { name: "Lost" }));
  const confirm = await screen.findByRole("dialog", { name: "Appeal lost" });
  expect(within(confirm).getByText(/counts against the photographer\. This cannot be undone\./)).toBeInTheDocument();
  expect(posts("/decide")).toEqual([]);
  await user.click(within(confirm).getByRole("button", { name: "Cancel" }));
  await waitFor(() => expect(screen.queryByRole("dialog", { name: "Appeal lost" })).not.toBeInTheDocument());
  expect(posts("/decide")).toEqual([]);
  expect(within(d).getByLabelText("Note (admins only, optional)")).toHaveValue("Checked");
});

it("two confirms in one tick send one decision", async () => {
  const user = userEvent.setup();
  post = async (url) => (url.endsWith("/decide") ? new Promise(() => {}) : { ...openAppeal, evidenceOpenedBy: admin.id });
  mount();
  const d = await drawer();
  await user.click(await within(d).findByRole("button", { name: "Open the samples" }));
  await user.click(await within(d).findByRole("button", { name: "Won" }));
  const ok = await screen.findByRole("button", { name: "Confirm won" });
  act(() => { ok.click(); ok.click(); });
  await waitFor(() => expect(posts("/decide")).toHaveLength(1));
  await act(async () => {});
  expect(posts("/decide")).toHaveLength(1);
});

it("when another admin opened the samples since, the samples can be opened again", async () => {
  const user = userEvent.setup();
  post = async (url) => {
    if (url.endsWith("/decide")) throw new ApiError([{ code: "EVIDENCE_NOT_OPENED", message: "Open this appeal's evidence before deciding it" }], 409);
    return { ...openAppeal, evidenceOpenedBy: admin.id };
  };
  mount();
  const d = await drawer();
  await user.click(await within(d).findByRole("button", { name: "Open the samples" }));
  await user.click(await within(d).findByRole("button", { name: "Won" }));
  await user.click(await screen.findByRole("button", { name: "Confirm won" }));
  expect(await within(d).findByRole("button", { name: "Open the samples" })).toBeEnabled();
  expect(within(d).getByRole("button", { name: "Won" })).toBeDisabled();
});

it("the open count is read with one row, and the Appeals tab reads no cases", async () => {
  fixtures.search = "tab=appeals";
  mount();
  await screen.findByRole("region", { name: "Open" });
  await waitFor(() => expect(api.get).toHaveBeenCalledWith("/admin/relevance/appeals?status=OPEN&offset=0&limit=1", expect.anything()));
  expect(vi.mocked(api.get).mock.calls.map(([url]) => String(url)).filter((u) => u.startsWith("/admin/relevance/cases?"))).toEqual([]);
  expect(await screen.findByRole("button", { name: /Appeals/ })).toHaveTextContent("Appeals 1");
});

it("the confirm of an appeal on a closed case says it only records the decision", async () => {
  const user = userEvent.setup();
  mount();
  let d = await drawer();
  await user.click(await within(d).findByRole("button", { name: "Open the samples" }));
  await user.click(await within(d).findByRole("button", { name: "Won" }));
  expect(within(await screen.findByRole("dialog", { name: "Appeal won" })).getByText(/^Marks the appeal won and tells the photographer\./)).toBeInTheDocument();
  cleanup(); client.clear();
  lists.OPEN = [{ ...openAppeal, caseStage: "CLOSED" }];
  mount();
  d = await drawer();
  await user.click(await within(d).findByRole("button", { name: "Open the samples" }));
  for (const [button, title] of [["Won", "Appeal won"], ["Lost", "Appeal lost"]] as const) {
    await user.click(within(d).getByRole("button", { name: button }));
    const confirm = await screen.findByRole("dialog", { name: title });
    expect(within(confirm).getByText("The case is already closed, so this only records your decision. The photographer is not told. This cannot be undone.")).toBeInTheDocument();
    expect(confirm.textContent).not.toMatch(/tells the photographer/);
    await user.click(within(confirm).getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(screen.queryByRole("dialog", { name: title })).not.toBeInTheDocument());
  }
});

// RV-4.3: a won appeal also ends a Copied case's payout hold (RelevanceAppealService.decide).
it("the Won confirm says a won appeal ends a copied case's payout hold", async () => {
  const user = userEvent.setup();
  mount();
  const d = await drawer();
  await user.click(await within(d).findByRole("button", { name: "Open the samples" }));
  await user.click(await within(d).findByRole("button", { name: "Won" }));
  const confirm = await screen.findByRole("dialog", { name: "Appeal won" });
  expect(confirm.textContent).toContain("A Copied case's payout hold ends; payouts resume once the photographer has no other hold.");
});
