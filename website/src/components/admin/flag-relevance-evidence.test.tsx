import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api, ApiError } from "@/lib/api";
import type { Flag, HiddenPhoto, RelevanceBatch, RelevanceEvidence } from "@/lib/admin-flags";
import { admin } from "@/test/fixtures/admin";
import { useAuthStore } from "@/store/auth-store";
import { useToastStore } from "@/store/toast-store";
import { FlagRelevanceEvidence } from "./flag-relevance-evidence";

vi.mock("@/lib/constants", async (original) => ({ ...await original<object>(), ADMIN_FLAGS_ENABLED: true }));

const flag: Flag = {
  id: "flag-2", photoId: null, eventId: null, eventName: null, photographerHandle: "juanshots", reportedBy: "system",
  reason: "batch_relevance_low", note: "event e1: copy 5/8", status: "open", reportedAt: "2026-11-08T00:00:00Z",
  reviewedAt: null, reviewedBy: null, reviewerNote: null, photoSnapshot: { alt: "Flag", kmMark: null, bib: null, thumbnailUrl: null },
  targetKind: "user", targetId: "user-1", payoutHoldAt: null,
};
const sample = (i: number) => ({
  photoId: `p${i}`, thumbnailUrl: `https://img/p${i}.jpg`, matchPhotoId: `m${i}`, matchThumbnailUrl: `https://img/m${i}.jpg`,
  matchPhotographerHandle: "maria", distance: 3, pair: null, exact: false, exactPhotoId: null, exactThumbnailUrl: null,
  exactPhotographerHandle: null, exactEventId: null,
});
const batch: RelevanceBatch = {
  eventId: "e1", eventName: "Cebu City Marathon 2027", eventDate: "2027-02-14", sharedRace: true, level: "LOW", copyLow: true, dateLow: false,
  photosLive: 40, lookups: 8, tightHits: 5, weakHits: 2, exactMatches: 1, xmpMatches: 0, datedPhotos: 30, datedOutside: 4,
  metadataPending: 0, bibbedPhotos: null, bibOverlapPhotos: null, scoredAt: "2026-11-12T00:00:00Z",
  copySamples: Array.from({ length: 15 }, (_, i) => sample(i)), dateSamples: [{ photoId: "d1", thumbnailUrl: null, takenOn: "2027-01-02" }], hiddenPhotos: 0,
};
const evidence = (b: RelevanceBatch[]): RelevanceEvidence => ({ flagId: "flag-2", payoutHoldAt: null, batches: b });
const hp = (id: string, restoredAt: string | null = null): HiddenPhoto => ({ photoId: id, eventId: "e1", thumbnailUrl: null, hiddenAt: "2026-11-13T00:00:00Z", restoredAt });

let client: QueryClient;
let relevance: () => Promise<RelevanceEvidence>;
let hidden: () => Promise<{ items: HiddenPhoto[]; total: number }>;
beforeEach(() => {
  useAuthStore.getState().setUser(admin);
  useToastStore.getState().clear();
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  relevance = async () => evidence([batch]);
  hidden = async () => ({ items: [], total: 0 });
  vi.spyOn(api, "get").mockImplementation(((url: string) => (url.includes("/relevance") ? relevance() : hidden())) as typeof api.get);
});
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); vi.restoreAllMocks(); });
const mount = (f: Flag = flag) => render(<QueryClientProvider client={client}><FlagRelevanceEvidence flag={f} /></QueryClientProvider>);
const toasts = () => useToastStore.getState().toasts;
const refusal = (m: string) => new ApiError([{ code: "INVALID_STATE_TRANSITION", message: m }], 409);

it("each batch shows its counts and sample pairs", async () => {
  mount();
  const card = await screen.findByRole("region", { name: "Batch in Cebu City Marathon 2027" });
  expect(within(card).getByText("Copy rule", { exact: false })).toBeInTheDocument();
  expect(within(card).getByText("5 tight of 8")).toBeInTheDocument();
  expect(within(card).getByText("4 of 30")).toBeInTheDocument();
  expect(within(card).getByText("Not available")).toBeInTheDocument();
  expect(within(card).getByText("The earlier upload is presumed to be the original. Check both photographers' upload histories before you decide.")).toBeInTheDocument();
  expect(within(card).getAllByText(/earlier upload by @maria/)).toHaveLength(12);
  expect(within(card).getAllByText(/Distance 3/)).toHaveLength(12);
  expect(within(card).getByText(/Hidden \(0\)/)).toBeInTheDocument();
});

it("an own event shows exact and marker counts only", async () => {
  relevance = async () => evidence([{ ...batch, sharedRace: false, copySamples: [], dateSamples: [], exactMatches: 2, xmpMatches: 3 }]);
  mount();
  const card = await screen.findByRole("region", { name: /Batch in/ });
  expect(within(card).getByText("Exact copies")).toBeInTheDocument();
  expect(within(card).getByText("Marker matches")).toBeInTheDocument();
  expect(within(card).queryByText("Look-ups")).not.toBeInTheDocument();
  expect(within(card).queryByText("Bib overlap")).not.toBeInTheDocument();
  expect(within(card).queryByText("Dated outside window")).not.toBeInTheDocument();
});

it("hide batch asks first and reports how many photos were hidden", async () => {
  const user = userEvent.setup();
  const post = vi.spyOn(api, "post").mockRejectedValueOnce(refusal("This flag is already closed.")).mockImplementation(async () => {
    relevance = async () => evidence([{ ...batch, hiddenPhotos: 12 }]);
    return flag;
  });
  mount();
  await user.click(await screen.findByRole("button", { name: "Hide batch…" }));
  expect(post).not.toHaveBeenCalled();
  expect(screen.getByText(/Takes every photo this photographer uploaded to Cebu City Marathon 2027 off sale/)).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Confirm hide batch" }));
  await waitFor(() => expect(toasts().some((t) => t.kind === "error" && t.message === "This flag is already closed.")).toBe(true));
  await user.click(screen.getByRole("button", { name: "Confirm hide batch" }));
  await waitFor(() => expect(toasts().some((t) => t.message === "12 photos hidden in Cebu City Marathon 2027")).toBe(true));
  expect(post).toHaveBeenLastCalledWith("/admin/flags/flag-2/hide-batch", { eventId: "e1", resolutionNote: null });
  expect(await screen.findByText(/Hidden \(12\)/)).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Confirm hide batch" })).not.toBeInTheDocument();
});

it("a closed flag offers no Hide batch", async () => {
  mount({ ...flag, status: "resolved" });
  await screen.findByRole("region", { name: /Batch in/ });
  expect(screen.queryByRole("button", { name: "Hide batch…" })).not.toBeInTheDocument();
});

it("restoring one photo keeps its row, greyed, and offers no second restore", async () => {
  const user = userEvent.setup();
  hidden = async () => ({ items: [hp("a"), hp("b")], total: 2 });
  const post = vi.spyOn(api, "post").mockRejectedValueOnce(refusal("Another flag still holds this photo.")).mockImplementation(async () => {
    hidden = async () => ({ items: [hp("a", "2026-11-13T05:00:00Z"), hp("b")], total: 2 });
    return flag;
  });
  mount();
  const restore = await screen.findAllByRole("button", { name: "Restore" });
  expect(restore).toHaveLength(2);
  await user.click(restore[0]!);
  await waitFor(() => expect(toasts().some((t) => t.message === "Another flag still holds this photo.")).toBe(true));
  await user.click(screen.getAllByRole("button", { name: "Restore" })[0]!);
  await waitFor(() => expect(screen.getAllByRole("button", { name: "Restore" })).toHaveLength(1));
  expect(post).toHaveBeenLastCalledWith("/admin/flags/flag-2/hidden-photos/a/restore");
  const greyed = document.querySelector('[data-restored="true"]') as HTMLElement;
  expect(greyed).toHaveClass("opacity-50");
  expect(within(greyed).queryByRole("button")).not.toBeInTheDocument();
});

it("loading, empty and error states render", async () => {
  const user = userEvent.setup();
  let release!: (v: RelevanceEvidence) => void;
  relevance = () => new Promise((r) => { release = r; });
  hidden = async () => { throw new Error("boom"); };
  mount();
  expect(screen.getByText("Loading evidence…")).toBeInTheDocument();
  release(evidence([]));
  expect(await screen.findByText("No batch is linked to this flag.")).toBeInTheDocument();
  expect(await screen.findByText("Could not load the hidden photos.")).toBeInTheDocument();
  hidden = async () => ({ items: [], total: 0 });
  await user.click(screen.getByRole("button", { name: "Retry" }));
  expect(await screen.findByText("No photos hidden by this flag.")).toBeInTheDocument();
  cleanup();
  client.clear();
  relevance = async () => { throw new Error("down"); };
  mount();
  expect(await screen.findByText("Could not load the evidence.")).toBeInTheDocument();
  relevance = async () => evidence([batch]);
  await user.click(screen.getAllByRole("button", { name: "Retry" })[0]!);
  expect(await screen.findByRole("region", { name: /Batch in/ })).toBeInTheDocument();
});

it("a hidden list longer than one page shows shown of total, loads the next page and restores a later photo", async () => {
  const user = userEvent.setup();
  const all = ["a", "b", "c", "d", "e"].map((id) => hp(id));
  const offsets: number[] = [];
  vi.mocked(api.get).mockImplementation((async (url: string) => {
    if (url.includes("/relevance")) return relevance();
    const offset = Number(new URL(`http://x${url}`).searchParams.get("offset"));
    offsets.push(offset);
    return { items: all.slice(offset, offset + 2), total: all.length, offset, limit: 2 };
  }) as typeof api.get);
  const post = vi.spyOn(api, "post").mockImplementation(async () => {
    all[4] = hp("e", "2026-11-13T05:00:00Z");
    return flag;
  });
  mount();
  expect(await screen.findByText(/Showing 2 of 5/)).toBeInTheDocument();
  expect(screen.getAllByRole("button", { name: "Restore" })).toHaveLength(2);
  await user.click(screen.getByRole("button", { name: /Load/ }));
  expect(await screen.findByText(/Showing 4 of 5/)).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: /Load/ }));
  expect(await screen.findByText(/All 5 loaded/)).toBeInTheDocument();
  expect(offsets).toEqual(expect.arrayContaining([0, 2, 4]));
  const restore = screen.getAllByRole("button", { name: "Restore" });
  expect(restore).toHaveLength(5);
  await user.click(restore[4]!);
  await waitFor(() => expect(post).toHaveBeenCalledWith("/admin/flags/flag-2/hidden-photos/e/restore"));
  await waitFor(() => expect(screen.getAllByRole("button", { name: "Restore" })).toHaveLength(4));
  expect(document.querySelectorAll('[data-restored="true"]')).toHaveLength(1);
});

it("a batch where both rules fired names both", async () => {
  relevance = async () => evidence([{ ...batch, copyLow: true, dateLow: true }]);
  mount();
  const card = await screen.findByRole("region", { name: /Batch in/ });
  expect(within(card).getByText(/Copy rule and date rule/)).toBeInTheDocument();
});

it("a batch whose event is gone says so and is not called an own event", async () => {
  relevance = async () => evidence([{ ...batch, eventName: "", eventDate: "", sharedRace: false, copySamples: [] }]);
  mount();
  const card = await screen.findByRole("region", { name: "Batch in an event that is no longer available" });
  expect(within(card).getByRole("heading", { name: "Event no longer available" })).toBeInTheDocument();
  expect(within(card).queryByText(/Own event/)).not.toBeInTheDocument();
});
