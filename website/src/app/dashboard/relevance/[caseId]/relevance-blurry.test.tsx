import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api, ApiError } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import type { PhotographerRelevanceCase } from "@/lib/api-photographer";
import { RelevanceCase } from "./relevance-case";

// RV-3.4b: a Fits case with blurry photos. The photographer hides some and can undo it; both grids change only after
// the server answers (each read again), never before.
const owner = { id: "owner-a", role: "PHOTOGRAPHER" as const, name: "Owner A", email: "a@example.test", createdAt: "2026-01-01T00:00:00Z" };
const fits: PhotographerRelevanceCase = {
  caseId: "case-1", eventId: "event-1", eventName: "Cebu Marathon", verdict: "FITS", action: "NONE", current: true,
  batchPhotos: 400, sampleSize: 300, junkCount: 3, hiddenPhotos: 0, scanMeasured: 400, scanBlurry: 2,
  noticeAt: null, appealDeadline: null, appeal: null, eventDate: "2026-11-08", closeReason: null,
};
const tile = (id: string) => ({ photoId: id, thumbnailUrl: `https://cdn.test/${id}.jpg` });
let blurry: string[];
let hidden: string[];
let caseRead: () => Promise<PhotographerRelevanceCase>;
let client: QueryClient;

beforeEach(() => {
  useAuthStore.getState().startSession(owner);
  client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  blurry = ["p1", "p2"];
  hidden = [];
  caseRead = async () => fits;
  vi.spyOn(api, "get").mockImplementation(((url: string) => {
    const page = (ids: string[]) => Promise.resolve({ items: ids.map(tile), total: ids.length, offset: 0, limit: 50 });
    if (url.includes("/blurry?")) return page(blurry);
    if (url.includes("/hidden?")) return page(hidden);
    return caseRead();
  }) as typeof api.get);
});
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); vi.restoreAllMocks(); });

const renderCase = () => render(<QueryClientProvider client={client}><RelevanceCase caseId="case-1" /></QueryClientProvider>);
const toReview = () => screen.findByRole("region", { name: "Review and hide" });
const hiddenList = () => screen.findByRole("region", { name: "Hidden" });

it("shows the whole blurry-photos note above the grid", async () => {
  renderCase();
  expect(await screen.findByText(/They are still on sale\. Review them, and hide any you do not want to sell\. You can show them again later\./)).toBeVisible();
  const review = await toReview();
  expect(await within(review).findAllByText("Looks blurry")).toHaveLength(2);
  expect(within(review).getByText(/Showing 2 of 2|All 2 loaded/)).toBeInTheDocument();
});

it("hiding selected blurry photos waits for the server", async () => {
  const user = userEvent.setup();
  let answer!: (v: { affected: number }) => void;
  const post = vi.spyOn(api, "post").mockImplementation(() => new Promise((r) => { answer = r as typeof answer; }));
  renderCase();
  const review = await toReview();
  await user.click(await within(review).findByRole("button", { name: "Photo 1" }));
  await user.click(within(review).getByRole("button", { name: "Hide" }));
  expect(post).toHaveBeenCalledWith("/me/photographer/relevance/cases/case-1/hide", { photoIds: ["p1"] });
  // Nothing moves before the answer.
  expect(within(review).getAllByText("Looks blurry")).toHaveLength(2);
  expect(screen.queryByRole("region", { name: "Hidden" })).not.toBeInTheDocument();
  blurry = ["p2"]; hidden = ["p1"];
  await act(async () => answer({ affected: 1 }));
  await waitFor(() => expect(within(review).getAllByText("Looks blurry")).toHaveLength(1));
  expect(await within(await hiddenList()).findByRole("button", { name: "Undo photo 1" })).toBeInTheDocument();
});

it("a second click while hiding sends one request", async () => {
  const post = vi.spyOn(api, "post").mockImplementation(() => new Promise(() => {}));
  renderCase();
  const review = await toReview();
  fireEvent.click(await within(review).findByRole("button", { name: "Photo 1" }));
  const hide = within(review).getByRole("button", { name: "Hide" });
  act(() => { hide.click(); hide.click(); });
  await waitFor(() => expect(post).toHaveBeenCalled());
  await act(async () => {});
  expect(post).toHaveBeenCalledTimes(1);
});

it("undo brings a photo back into the grid", async () => {
  const user = userEvent.setup();
  blurry = []; hidden = ["p1"];
  const post = vi.spyOn(api, "post").mockImplementation(async () => { blurry = ["p1"]; hidden = []; return { affected: 1 }; });
  renderCase();
  await user.click(await within(await hiddenList()).findByRole("button", { name: "Undo photo 1" }));
  expect(post).toHaveBeenCalledWith("/me/photographer/relevance/cases/case-1/unhide", { photoIds: ["p1"] });
  expect(await within(await toReview()).findByRole("button", { name: "Photo 1" })).toBeInTheDocument();
  await waitFor(() => expect(screen.queryByRole("button", { name: "Undo photo 1" })).not.toBeInTheDocument());
});

it("a refused hide shows the server's message", async () => {
  const user = userEvent.setup();
  vi.spyOn(api, "post").mockRejectedValue(new ApiError([{ code: "RELEVANCE_ACTIONS_OFF",
    message: "Hiding photos needs a case applied in the relevance mode auto-hide" }], 409));
  renderCase();
  const review = await toReview();
  await user.click(await within(review).findByRole("button", { name: "Photo 2" }));
  await user.click(within(review).getByRole("button", { name: "Hide" }));
  expect(await screen.findByText("Hiding photos needs a case applied in the relevance mode auto-hide")).toBeVisible();
  expect(within(review).getAllByText("Looks blurry")).toHaveLength(2);
});

it("a thumbnail that fails to load keeps its tile selectable", async () => {
  renderCase();
  const review = await toReview();
  const select = await within(review).findByRole("button", { name: "Photo 1" });
  fireEvent.error(select.querySelector("img")!);
  fireEvent.click(select);
  expect(within(review).getByRole("button", { name: "Photo 1" })).toHaveAttribute("aria-pressed", "true");
  expect(within(review).getByRole("button", { name: "Hide" })).toBeEnabled();
});

it("the grid is absent for a record-only case", async () => {
  // The server answers a record-only case's read with not found (RelevanceAppealService.caseFor).
  caseRead = () => Promise.reject(new ApiError([{ code: "NOT_FOUND", message: "Case not found" }], 404));
  renderCase();
  expect(await screen.findByText("Case not found")).toBeVisible();
  expect(screen.queryByRole("region", { name: "Review and hide" })).not.toBeInTheDocument();
  expect(vi.mocked(api.get).mock.calls.map(([url]) => String(url)).filter((u) => /\/(blurry|hidden)\?/.test(u))).toEqual([]);
});

it("each tile keeps one name and its pressed state says whether it is selected", async () => {
  renderCase();
  const review = await toReview();
  const tile = await within(review).findByRole("button", { name: "Photo 1" });
  expect(tile).toHaveAttribute("aria-pressed", "false");
  fireEvent.click(tile);
  expect(within(review).getByRole("button", { name: "Photo 1" })).toHaveAttribute("aria-pressed", "true");
});

it("each list shows that it is loading, and offers a retry when its read failed", async () => {
  const user = userEvent.setup();
  let failBlurry = true;
  const pending = new Promise(() => {});
  vi.mocked(api.get).mockImplementation(((url: string) => {
    if (url.includes("/blurry?")) return failBlurry ? Promise.reject(new Error("network")) : Promise.resolve({ items: blurry.map(tile), total: blurry.length, offset: 0, limit: 50 });
    if (url.includes("/hidden?")) return pending;
    return caseRead();
  }) as typeof api.get);
  renderCase();
  expect(await within(await hiddenList()).findByText("Loading…")).toBeInTheDocument();
  const review = await toReview();
  expect(await within(review).findByText("Couldn't load")).toBeInTheDocument();
  failBlurry = false;
  await user.click(within(review).getByRole("button", { name: "Try again" }));
  expect(await within(review).findByRole("button", { name: "Photo 1" })).toBeInTheDocument();
});

it("an empty list is not drawn", async () => {
  blurry = []; hidden = ["p1"];
  renderCase();
  expect(await within(await hiddenList()).findByRole("button", { name: "Undo photo 1" })).toBeInTheDocument();
  expect(screen.queryByRole("region", { name: "Review and hide" })).not.toBeInTheDocument();
  cleanup(); client.clear();
  blurry = ["p1"]; hidden = [];
  renderCase();
  expect(await within(await toReview()).findByRole("button", { name: "Photo 1" })).toBeInTheDocument();
  await waitFor(() => expect(vi.mocked(api.get).mock.calls.some(([u]) => String(u).includes("/hidden?"))).toBe(true));
  await act(async () => {});
  expect(screen.queryByRole("region", { name: "Hidden" })).not.toBeInTheDocument();
});

it("a selection over 200 is sent in chunks of 200, one after the other, and read again once", async () => {
  blurry = Array.from({ length: 250 }, (_, i) => `p${String(i).padStart(3, "0")}`);
  const answers: ((v: { affected: number }) => void)[] = [];
  const post = vi.spyOn(api, "post").mockImplementation(() => new Promise((r) => { answers.push(r as (v: { affected: number }) => void); }));
  renderCase();
  const review = await toReview();
  // getByRole over hundreds of tiles is slow: the tiles are read straight from the DOM.
  const tiles = () => review.querySelectorAll<HTMLButtonElement>("button[aria-pressed]");
  await waitFor(() => expect(tiles()).toHaveLength(250));
  tiles().forEach((b) => fireEvent.click(b));
  const reads = () => vi.mocked(api.get).mock.calls.filter(([u]) => String(u).includes("/blurry?")).length;
  const before = reads();
  fireEvent.click(within(review).getByRole("button", { name: "Hide" }));
  await waitFor(() => expect(post).toHaveBeenCalledTimes(1));
  expect((post.mock.calls[0][1] as { photoIds: string[] }).photoIds).toHaveLength(200);
  await act(async () => answers[0]({ affected: 200 }));
  await waitFor(() => expect(post).toHaveBeenCalledTimes(2));
  expect((post.mock.calls[1][1] as { photoIds: string[] }).photoIds).toHaveLength(50);
  expect(reads()).toBe(before);
  blurry = [];
  await act(async () => answers[1]({ affected: 50 }));
  await waitFor(() => expect(reads()).toBe(before + 1));
}, 20_000);

it("a refused chunk stops the rest and shows the server's message", async () => {
  blurry = Array.from({ length: 201 }, (_, i) => `p${String(i).padStart(3, "0")}`);
  const post = vi.spyOn(api, "post").mockRejectedValue(new ApiError([{ code: "VALIDATION_ERROR", message: "Send 1 to 200 photo ids" }], 400));
  renderCase();
  const review = await toReview();
  // getByRole over hundreds of tiles is slow: the tiles are read straight from the DOM.
  const tiles = () => review.querySelectorAll<HTMLButtonElement>("button[aria-pressed]");
  await waitFor(() => expect(tiles()).toHaveLength(201));
  tiles().forEach((b) => fireEvent.click(b));
  fireEvent.click(within(review).getByRole("button", { name: "Hide" }));
  expect(await screen.findByText("Send 1 to 200 photo ids")).toBeVisible();
  expect(post).toHaveBeenCalledTimes(1);
}, 20_000);
