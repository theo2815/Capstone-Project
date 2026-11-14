import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import type { RelevanceCaseDetail, RelevanceCaseRow } from "@/lib/admin-relevance";
import { admin } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import { RelevanceCases } from "./relevance-cases";

const fixtures = vi.hoisted(() => ({ search: "", router: { replace: vi.fn() } }));
vi.mock("next/navigation", () => ({
  useRouter: () => fixtures.router,
  usePathname: () => "/admin/relevance",
  useSearchParams: () => new URLSearchParams(fixtures.search),
}));

const row: RelevanceCaseRow = {
  caseId: "c1", eventId: "e1", eventName: "Cebu City Marathon 2003", photographerHandle: "juanshots",
  photographerName: "Juan Cruz", verdict: "JUNK_DUMP", verdictNote: null, action: "HIDE_ALL", appliedMode: "RECORD_ONLY",
  wouldHave: true, batchPhotos: 420, junkPercent: 81, decidedAt: "2003-02-14T03:00:00Z",
};
const detail = (r: RelevanceCaseRow = row, extra: Partial<RelevanceCaseDetail> = {}): RelevanceCaseDetail => ({
  case: r, stage: "APPLIED", sampleSize: 300, measured: 300, blurry: 200, dark: 30, notRace: 13, junk: 243, band: "OVER",
  modelVerdict: r.verdict, adminReason: "Most of the sample is blurry.", evidenceUsed: ["junk_share"],
  summarySent: '{"case":"c1"}', model: "claude-sonnet-5-5", promptVersion: "rv1", wordsPhotos: 380,
  topWords: [{ word: "CEBU", photos: 300 }],
  junkSamples: [{ photoId: "j1", thumbnailUrl: "https://img/j1.jpg", fgSharpness: 4.5, fgBrightness: 50, sharpness: 5, brightness: 48, blurry: true, dark: false, notRace: false }],
  cleanSamples: [{ photoId: "k1", thumbnailUrl: null, fgSharpness: 80, fgBrightness: 60, sharpness: 82, brightness: 61, blurry: null, dark: null, notRace: false }],
  alerts: [],
  ...extra,
});

let list: (url: string) => Promise<unknown>;
let one: () => Promise<RelevanceCaseDetail>;
let client: QueryClient;
beforeEach(() => {
  fixtures.search = "";
  fixtures.router.replace.mockClear();
  useAuthStore.getState().setUser(admin);
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  list = async () => ({ items: [row], total: 1, offset: 0, limit: 50 });
  one = async () => detail();
  vi.spyOn(api, "get").mockImplementation(((url: string) =>
    url.startsWith("/admin/relevance/cases?") ? list(url) : one()) as typeof api.get);
});
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); vi.restoreAllMocks(); });
const mount = () => render(<QueryClientProvider client={client}><RelevanceCases /></QueryClientProvider>);

it("the tab comes from the URL and changing it keeps the row closed", async () => {
  const user = userEvent.setup();
  fixtures.search = "tab=cant_tell";
  mount();
  expect(screen.getByRole("button", { name: "Can't tell" })).toHaveAttribute("aria-pressed", "true");
  await user.click(await screen.findByRole("button", { name: /Cebu City Marathon 2003/ }));
  expect(api.get).toHaveBeenCalledWith(expect.stringContaining("tab=cant_tell"), expect.anything());
  expect(fixtures.router.replace).toHaveBeenLastCalledWith("/admin/relevance?tab=cant_tell&row=c1", { scroll: false });
  expect(await screen.findByRole("dialog")).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "All" }));
  expect(fixtures.router.replace).toHaveBeenLastCalledWith("/admin/relevance?tab=all", { scroll: false });
  await waitFor(() => expect(api.get).toHaveBeenCalledWith(expect.stringContaining("tab=all"), expect.anything()));
  expect(screen.getByRole("button", { name: "All" })).toHaveAttribute("aria-pressed", "true");
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
});

it("a URL with both tab and row selects that tab and opens that case", async () => {
  fixtures.search = "tab=all&row=c1";
  mount();
  expect(screen.getByRole("button", { name: "All" })).toHaveAttribute("aria-pressed", "true");
  expect(screen.getByRole("button", { name: "Hidden" })).toHaveAttribute("aria-pressed", "false");
  const drawer = await screen.findByRole("dialog");
  expect(within(drawer).getByText("Case · c1")).toBeInTheDocument();
  expect(await within(drawer).findByRole("heading", { name: "Cebu City Marathon 2003" })).toBeInTheDocument();
  expect(api.get).toHaveBeenCalledWith("/admin/relevance/cases/c1", expect.anything());
});

it("a record-only case is labelled would have hidden and shows no action", async () => {
  fixtures.search = "row=c1";
  mount();
  const card = await screen.findByRole("button", { name: /Cebu City Marathon 2003/ });
  expect(within(card).getByText("Would have hidden")).toBeInTheDocument();
  const drawer = await screen.findByRole("dialog");
  expect(await within(drawer).findByText(/Would have hidden: Hide the whole batch/)).toBeInTheDocument();
  expect(within(drawer).getByText(/Record-only: nothing was hidden/)).toBeInTheDocument();
  expect(within(drawer).getAllByRole("button").map((b) => b.getAttribute("aria-label") ?? b.textContent)).toEqual(["Close"]);
  // Server codes read as words: the evidence gets a label, and the stage (always APPLIED here) is not shown.
  expect(within(drawer).getByText("Evidence used: Junk share")).toBeInTheDocument();
  expect(within(drawer).queryByText("APPLIED")).not.toBeInTheDocument();
  // A clean sample whose cut-offs are unset is "not judged", and a sample with no preview says it is gone.
  expect(within(drawer).getByText("Not judged")).toBeInTheDocument();
  expect(within(drawer).getByRole("img", { name: /no longer available/ })).toBeInTheDocument();
});

it("a case shows the model's reason and the note when code overrode it", async () => {
  const overridden = { ...row, verdict: "CANT_TELL", verdictNote: "CONTRADICTS", action: "NONE", wouldHave: false };
  list = async () => ({ items: [overridden], total: 1, offset: 0, limit: 50 });
  one = async () => detail(overridden, { modelVerdict: "JUNK_DUMP", adminReason: "Nearly every photo is a lens cap shot." });
  fixtures.search = "row=c1";
  mount();
  const drawer = await screen.findByRole("dialog");
  expect(await within(drawer).findByText("Nearly every photo is a lens cap shot.")).toBeInTheDocument();
  expect(within(drawer).getByText(/The model said Junk dump; code stored Can't tell/)).toBeInTheDocument();
  expect(within(drawer).getByText("The model's verdict contradicts the measured numbers or the copy rule.")).toBeInTheDocument();
  expect(within(drawer).queryByText(/Would have hidden/)).not.toBeInTheDocument();
});

it("an unknown verdict from the server renders as text, not a crash", async () => {
  const odd = { ...row, verdict: "NEW_VERDICT", verdictNote: "NEW_NOTE", action: "HIDE_SOME" };
  list = async () => ({ items: [odd], total: 1, offset: 0, limit: 50 });
  one = async () => detail(odd, { band: "NEW_BAND", modelVerdict: "OTHER" });
  fixtures.search = "row=c1";
  mount();
  const card = await screen.findByRole("button", { name: /Cebu City Marathon 2003/ });
  expect(within(card).getByText("NEW_VERDICT")).toBeInTheDocument();
  const drawer = await screen.findByRole("dialog");
  expect(await within(drawer).findByText("NEW_NOTE")).toBeInTheDocument();
  expect(within(drawer).getAllByText("NEW_VERDICT").length).toBeGreaterThan(0);
  expect(within(drawer).getByText("NEW_BAND")).toBeInTheDocument();
});

it("a tab shows loading, failed with retry, empty, and Showing S of T with Load more", async () => {
  const user = userEvent.setup();
  let release!: (v: unknown) => void;
  list = () => new Promise((r) => { release = r; });
  mount();
  expect(screen.getByText("Loading cases…")).toBeInTheDocument();
  release({ items: [], total: 0, offset: 0, limit: 50 });
  expect(await screen.findByText("No batch is hidden, or would have been hidden.")).toBeInTheDocument();
  cleanup();
  client.clear();
  list = async () => { throw new Error("down"); };
  mount();
  expect(await screen.findByText("Could not load the cases.")).toBeInTheDocument();
  const rows = [row, { ...row, caseId: "c2", eventName: "Second Race" }, { ...row, caseId: "c3", eventName: "Third Race" }];
  list = async (url) => {
    const offset = Number(new URL(`http://x${url}`).searchParams.get("offset"));
    return { items: rows.slice(offset, offset + 2), total: 3, offset, limit: 2 };
  };
  await user.click(screen.getByRole("button", { name: "Retry" }));
  expect(await screen.findByText(/Showing 2 of 3/)).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: /Load/ }));
  expect(await screen.findByText("Third Race")).toBeInTheDocument();
  expect(screen.getByText(/All 3 loaded/)).toBeInTheDocument();
});

it("the Appeals tab says appeals arrive with the photographer side", async () => {
  fixtures.search = "tab=appeals";
  list = async () => ({ items: [], total: 0, offset: 0, limit: 50 });
  mount();
  expect(await screen.findByText("No appeals yet. Appeals arrive when photographers can answer a decision.")).toBeInTheDocument();
});
