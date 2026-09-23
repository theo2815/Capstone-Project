import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { clearTokens, setTokens } from "@/lib/auth";
import type { EventPhotosResult } from "@/lib/api-photos";
import { useAuthStore } from "@/store/auth-store";
import { SelfieSearchPanel } from "./bib-search-panels";

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }), usePathname: () => "/events/race" }));
vi.mock("@/hooks/use-selfies", () => ({
  useSelfiesList: () => ({ selfies: [], isLoading: false, error: null }),
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
  expect(fetch).toHaveBeenCalledTimes(1);

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
