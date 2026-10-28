import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { clearTokens, setTokens } from "@/lib/auth";
import type { EventPhotosResult } from "@/lib/api-photos";
import { shrinkSelfie } from "@/lib/image-utils";
import { useAuthStore } from "@/store/auth-store";
import { SelfieSearchPanel } from "./bib-search-panels";

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }), usePathname: () => "/events/race" }));
vi.mock("@/lib/image-utils", async (importOriginal) => ({ ...(await importOriginal<typeof import("@/lib/image-utils")>()), shrinkSelfie: vi.fn() }));
const library = vi.hoisted(() => ({ selfies: [] as { id: string; dataUrl: string; uploadedAt: string; isPrimary: boolean; qualityScore: number; qualityTestStatus: string }[] }));
vi.mock("@/hooks/use-selfies", () => ({
  useSelfiesList: () => ({ selfies: library.selfies, isLoading: false, error: null }),
}));

const runner = { id: "runner-a", role: "RUNNER" as const, name: "Runner", email: "a@example.test", createdAt: "2026-09-10" };
const file = new File(["selfie"], "picked.png", { type: "image/png" });
const selfie = { id: "selfie-1", dataUrl: "/selfie.jpg", uploadedAt: "2026-09-10", isPrimary: true, qualityScore: 0, qualityTestStatus: "untested" };
const ok = () => new Response(JSON.stringify({ success: true, data: selfie }));
const rejected = () => new Response(JSON.stringify({ success: false, errors: [{ code: "SELFIE_REJECTED", message: "No face found." }] }), { status: 422 });
const searchResult: EventPhotosResult = { items: [], total: 0, offset: 0, limit: 200 };
const found = () => new Response(JSON.stringify({ success: true, data: searchResult }));

let client: QueryClient;
beforeEach(() => {
  library.selfies = [];
  vi.mocked(shrinkSelfie).mockReset().mockImplementation(async (picked) => picked);
  useAuthStore.getState().startSession(runner);
  setTokens("access-a", "refresh-a");
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
});
afterEach(() => {
  client.clear();
  useAuthStore.getState().logout();
  clearTokens();
  vi.unstubAllGlobals();
});

function mountAndPick(onSearchSuccess: (result: EventPhotosResult) => void) {
  render(<QueryClientProvider client={client}><SelfieSearchPanel eventSlug="race" onSwitchToBib={() => {}} onSearchSuccess={onSearchSuccess} /></QueryClientProvider>);
  fireEvent.change(screen.getByLabelText(/Upload a selfie/), { target: { files: [file] } });
}

it.each(["resolved", "rejected"])("does not search under replacement credentials after a stale save is %s", async (outcome) => {
  let finish!: (response: Response) => void;
  const first = new Promise<Response>((resolve) => { finish = resolve; });
  const fetch = vi.fn().mockReturnValueOnce(first).mockResolvedValue(found());
  vi.stubGlobal("fetch", fetch);
  const onSearchSuccess = vi.fn();
  const invalidate = vi.spyOn(client, "invalidateQueries");
  mountAndPick(onSearchSuccess);
  await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1));

  act(() => { useAuthStore.getState().startSession({ ...runner, id: "runner-b" }); setTokens("access-b", "refresh-b"); });
  await act(async () => { finish(outcome === "resolved" ? ok() : rejected()); await first; await new Promise((resolve) => setTimeout(resolve, 0)); });
  expect(fetch).toHaveBeenCalledTimes(1);
  expect(invalidate).not.toHaveBeenCalled();
  expect(onSearchSuccess).not.toHaveBeenCalled();
  expect(screen.queryByText(/Saved to your selfie library|Not saved to your library/)).not.toBeInTheDocument();
});

it("keeps save and search working in the current session", async () => {
  const fetch = vi.fn().mockResolvedValueOnce(ok()).mockResolvedValueOnce(found());
  vi.stubGlobal("fetch", fetch);
  const onSearchSuccess = vi.fn();
  mountAndPick(onSearchSuccess);
  await waitFor(() => expect(onSearchSuccess).toHaveBeenCalledWith(searchResult));
  expect(fetch).toHaveBeenCalledTimes(2);
  expect((fetch.mock.calls[1][1] as RequestInit).headers).toMatchObject({ Authorization: "Bearer access-a" });
});

it("falls back to file search after a current-session save rejection", async () => {
  const fetch = vi.fn().mockResolvedValueOnce(rejected()).mockResolvedValueOnce(found());
  vi.stubGlobal("fetch", fetch);
  const onSearchSuccess = vi.fn();
  mountAndPick(onSearchSuccess);
  await waitFor(() => expect(onSearchSuccess).toHaveBeenCalledWith(searchResult));
  expect(fetch).toHaveBeenCalledTimes(2);
  expect(screen.getByText(/Not saved to your library/)).toBeInTheDocument();
});

const shrunk = new File(["small"], "selfie.jpg", { type: "image/jpeg" });
const sentFile = (fetch: ReturnType<typeof vi.fn>, call: number, field: string) =>
  ((fetch.mock.calls[call][1] as RequestInit).body as FormData).get(field);

it("saves the shrunk selfie, not the picked file", async () => {
  vi.mocked(shrinkSelfie).mockResolvedValue(shrunk);
  const fetch = vi.fn().mockResolvedValueOnce(ok()).mockResolvedValueOnce(found());
  vi.stubGlobal("fetch", fetch);
  const onSearchSuccess = vi.fn();
  mountAndPick(onSearchSuccess);
  await waitFor(() => expect(onSearchSuccess).toHaveBeenCalledWith(searchResult));
  expect(shrinkSelfie).toHaveBeenCalledExactlyOnceWith(file);
  expect(sentFile(fetch, 0, "file")).toBe(shrunk);
});

it("searches with the same shrunk selfie after a refused save, shrinking once", async () => {
  vi.mocked(shrinkSelfie).mockResolvedValue(shrunk);
  const fetch = vi.fn().mockResolvedValueOnce(rejected()).mockResolvedValueOnce(found());
  vi.stubGlobal("fetch", fetch);
  const onSearchSuccess = vi.fn();
  mountAndPick(onSearchSuccess);
  await waitFor(() => expect(onSearchSuccess).toHaveBeenCalledWith(searchResult));
  expect(shrinkSelfie).toHaveBeenCalledTimes(1);
  expect(sentFile(fetch, 0, "file")).toBe(shrunk);
  expect(sentFile(fetch, 1, "selfie")).toBe(shrunk);
});

it("sends nothing when authentication changes while the selfie shrinks", async () => {
  let finish!: (out: File) => void;
  vi.mocked(shrinkSelfie).mockReturnValue(new Promise((resolve) => { finish = resolve; }));
  const fetch = vi.fn().mockResolvedValue(found());
  vi.stubGlobal("fetch", fetch);
  mountAndPick(vi.fn());
  act(() => { useAuthStore.getState().startSession({ ...runner, id: "runner-b" }); setTokens("access-b", "refresh-b"); });
  await act(async () => { finish(shrunk); await new Promise((resolve) => setTimeout(resolve, 0)); });
  expect(fetch).not.toHaveBeenCalled();
});

it.each(["all", "selected"])("rejects a late saved-%s search after same-account login", async (selection) => {
  library.selfies = [selfie];
  let finish!: (response: Response) => void;
  const pending = new Promise<Response>((resolve) => { finish = resolve; });
  vi.stubGlobal("fetch", vi.fn().mockReturnValue(pending));
  const onSearchSuccess = vi.fn();
  render(<QueryClientProvider client={client}><SelfieSearchPanel eventSlug="race" onSwitchToBib={() => {}} onSearchSuccess={onSearchSuccess} /></QueryClientProvider>);
  fireEvent.click(screen.getByRole("button", { name: /Use saved selfie/ }));
  fireEvent.click(screen.getByRole("button", { name: selection === "all" ? /Match with all/ : /Search with primary selfie/ }));
  act(() => useAuthStore.getState().startSession(runner));
  await act(async () => { finish(found()); await pending; });
  expect(onSearchSuccess).not.toHaveBeenCalled();
});

it("allows the replacement session to search without an old result clearing its busy state", async () => {
  library.selfies = [selfie];
  let finishOld!: (response: Response) => void, finishNew!: (response: Response) => void;
  const old = new Promise<Response>((resolve) => { finishOld = resolve; });
  const next = new Promise<Response>((resolve) => { finishNew = resolve; });
  vi.stubGlobal("fetch", vi.fn().mockReturnValueOnce(old).mockReturnValueOnce(next));
  const success = vi.fn();
  render(<QueryClientProvider client={client}><SelfieSearchPanel eventSlug="race" onSwitchToBib={() => {}} onSearchSuccess={success} /></QueryClientProvider>);
  fireEvent.click(screen.getByRole("button", { name: /Use saved selfie/ }));
  fireEvent.click(screen.getByRole("button", { name: /Match with all/ }));
  act(() => useAuthStore.getState().startSession(runner));
  await waitFor(() => expect(screen.getByLabelText(/Upload a selfie/)).not.toBeDisabled());
  fireEvent.click(screen.getByRole("button", { name: /Use saved selfie/ }));
  fireEvent.click(screen.getByRole("button", { name: /Match with all/ }));
  await act(async () => { finishOld(found()); await old; });
  expect(success).not.toHaveBeenCalled();
  expect(screen.getByLabelText(/Upload a selfie/)).toBeDisabled();
  await act(async () => { finishNew(found()); await next; });
  expect(success).toHaveBeenCalledTimes(1);
});

it("tells the runner how long to wait when the search is refused with a 429", async () => {
  const refused = () => new Response(JSON.stringify({ success: false, errors: [{ code: "RATE_LIMITED", message: "Too many requests. Please slow down and try again." }] }), { status: 429, headers: { "Retry-After": "19" } });
  vi.stubGlobal("fetch", vi.fn().mockImplementation(async () => refused()));
  mountAndPick(vi.fn());
  expect(await screen.findByText("Too many searches. Try again in about 19 seconds.")).toBeInTheDocument();
});
