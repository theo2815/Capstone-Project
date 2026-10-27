import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { clearTokens, setTokens } from "@/lib/auth";
import { shrinkSelfie } from "@/lib/image-utils";
import { useAuthStore } from "@/store/auth-store";
import { useToastStore } from "@/store/toast-store";
import { SelfieLibrary } from "./selfie-library";

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }) }));
vi.mock("@/lib/image-utils", async (importOriginal) => ({ ...(await importOriginal<typeof import("@/lib/image-utils")>()), shrinkSelfie: vi.fn() }));
vi.mock("@/hooks/use-selfies", () => ({
  useSelfiesList: () => ({ selfies: [], isLoading: false, error: null }),
}));

const runner = { id: "runner-a", role: "RUNNER" as const, name: "Runner", email: "a@example.test", createdAt: "2026-09-10" };
const files = [
  new File(["first"], "first.png", { type: "image/png" }),
  new File(["second"], "second.png", { type: "image/png" }),
];
const selfie = { id: "selfie-1", dataUrl: "/selfie.jpg", uploadedAt: "2026-09-10", isPrimary: true, qualityScore: 0, qualityTestStatus: "untested" };
const ok = () => new Response(JSON.stringify({ success: true, data: selfie }));
const rejected = () => new Response(JSON.stringify({ success: false, errors: [{ code: "SELFIE_REJECTED", message: "No face found." }] }), { status: 422 });

let client: QueryClient;
beforeEach(() => {
  vi.mocked(shrinkSelfie).mockReset().mockImplementation(async (picked) => picked);
  useAuthStore.getState().startSession(runner);
  setTokens("access-a", "refresh-a");
  useToastStore.getState().clear();
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
});
afterEach(() => {
  client.clear();
  useAuthStore.getState().logout();
  clearTokens();
  vi.unstubAllGlobals();
});

function mountAndPick() {
  render(<QueryClientProvider client={client}><SelfieLibrary returnTo="/events/race" /></QueryClientProvider>);
  fireEvent.change(screen.getByLabelText(/Upload selfies/), { target: { files } });
}

it.each([
  ["resolved", "account switch"], ["rejected", "account switch"],
  ["resolved", "logout"], ["rejected", "logout"],
])("does not upload a later file or publish stale %s work after %s", async (outcome, transition) => {
  let finish!: (response: Response) => void;
  const first = new Promise<Response>((resolve) => { finish = resolve; });
  const fetch = vi.fn().mockReturnValueOnce(first).mockResolvedValue(ok());
  vi.stubGlobal("fetch", fetch);
  const invalidate = vi.spyOn(client, "invalidateQueries");
  mountAndPick();
  await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1));
  expect((fetch.mock.calls[0][1] as RequestInit).headers).toMatchObject({ Authorization: "Bearer access-a" });

  act(() => {
    if (transition === "logout") { useAuthStore.getState().logout(); clearTokens(); }
    else { useAuthStore.getState().startSession({ ...runner, id: "runner-b" }); setTokens("access-b", "refresh-b"); }
  });
  await act(async () => { finish(outcome === "resolved" ? ok() : rejected()); await first; await new Promise((resolve) => setTimeout(resolve, 0)); });
  expect(fetch).toHaveBeenCalledTimes(1);
  expect(invalidate).not.toHaveBeenCalled();
  expect(useToastStore.getState().toasts).toEqual([]);
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
});

it("does not continue a batch after fresh same-account authentication", async () => {
  let finish!: (response: Response) => void;
  const first = new Promise<Response>((resolve) => { finish = resolve; });
  const fetch = vi.fn().mockReturnValueOnce(first).mockResolvedValue(ok());
  vi.stubGlobal("fetch", fetch);
  mountAndPick();
  await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1));

  act(() => { useAuthStore.getState().startSession(runner); setTokens("access-new", "refresh-new"); });
  await act(async () => { finish(ok()); await first; await new Promise((resolve) => setTimeout(resolve, 0)); });
  expect(fetch).toHaveBeenCalledTimes(1);
  expect(useToastStore.getState().toasts).toEqual([]);
});

it.each(["resolved", "rejected"])("continues a current-session batch after the first file is %s", async (outcome) => {
  const fetch = vi.fn().mockResolvedValueOnce(outcome === "resolved" ? ok() : rejected()).mockResolvedValueOnce(ok());
  vi.stubGlobal("fetch", fetch);
  mountAndPick();

  await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
  await waitFor(() => expect(useToastStore.getState().toasts).toHaveLength(1));
  expect((fetch.mock.calls[1][1] as RequestInit).headers).toMatchObject({ Authorization: "Bearer access-a" });
  expect(((fetch.mock.calls[1][1] as RequestInit).body as FormData).get("file")).toMatchObject({ name: "second.png" });
  expect(useToastStore.getState().toasts[0].message).toContain(outcome === "resolved" ? "Added 2 selfies" : "Selfie added");
  if (outcome === "rejected") expect(screen.getByRole("alert")).toHaveTextContent("first.png: No face found.");
});

it("uploads each picked file shrunk", async () => {
  const shrunk = files.map((f) => new File(["small"], `small-${f.name}`, { type: "image/jpeg" }));
  vi.mocked(shrinkSelfie).mockImplementation(async (picked) => shrunk[files.indexOf(picked)]);
  const fetch = vi.fn().mockResolvedValueOnce(ok()).mockResolvedValueOnce(ok());
  vi.stubGlobal("fetch", fetch);
  mountAndPick();

  await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2));
  expect(shrinkSelfie).toHaveBeenCalledTimes(2);
  expect(((fetch.mock.calls[0][1] as RequestInit).body as FormData).get("file")).toBe(shrunk[0]);
  expect(((fetch.mock.calls[1][1] as RequestInit).body as FormData).get("file")).toBe(shrunk[1]);
});
