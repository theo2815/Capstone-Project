import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import * as publicFaceSearchApi from "@/lib/api-photos";
import * as personalSelfieApi from "@/lib/api-selfies";
import { useAuthStore } from "@/store/auth-store";
import { FindPhotosControls } from "./find-photos-controls";
import { FindPhotosModal } from "./find-photos-modal";

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }), usePathname: () => "/events/race" }));
vi.mock("@/hooks/use-selfies", () => ({ useSelfiesList: () => ({ selfies: [], isLoading: false, error: null }) }));
const runner = { id: "runner", role: "RUNNER" as const, name: "Runner", email: "a@example.test", createdAt: "2026-10-08" };
const file = new File(["private selfie"], "selfie.png", { type: "image/png" });
const found = { items: [], total: 0, offset: 0, limit: 200 };
let client: QueryClient;
beforeEach(() => { client = new QueryClient(); useAuthStore.getState().startSession(runner); });
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); vi.restoreAllMocks(); });

it("removalSearchCannotSaveToPersonalLibrary", () => {
  const save = vi.spyOn(personalSelfieApi, "uploadSelfie");
  const search = vi.spyOn(publicFaceSearchApi, "searchEventByFace");
  const onSearch = vi.fn();
  function Removal() { const [selfie, setSelfie] = useState<File | null>(null); const [bib, setBib] = useState(""); return <FindPhotosControls file={selfie} bib={bib} onFileChange={setSelfie} onBibChange={setBib} onSearch={onSearch} busy={false} error={null} allowPersonalLibrary={false} />; }
  render(<Removal />);
  fireEvent.change(screen.getByLabelText(/Upload a selfie/), { target: { files: [file] } });
  expect(screen.getByText("selfie.png")).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Enter your bib number"), { target: { value: "123" } });
  fireEvent.click(screen.getByRole("button", { name: "Search" }));
  expect(onSearch).toHaveBeenCalledTimes(1);
  expect(save).not.toHaveBeenCalled(); expect(search).not.toHaveBeenCalled();
  expect(screen.queryByText(/save.*library|Use saved selfie/i)).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Remove selected selfie" }));
  expect(screen.queryByText("selfie.png")).not.toBeInTheDocument();
});

it("keeps controlled input on failure and disables duplicate work while busy", () => {
  const onSearch = vi.fn(), onFileChange = vi.fn();
  render(<FindPhotosControls file={file} bib="B-12" onFileChange={onFileChange} onBibChange={() => {}} onSearch={onSearch} busy error="Backend refused" allowPersonalLibrary={false} />);
  expect(screen.getByText("selfie.png")).toBeInTheDocument();
  expect(screen.getByDisplayValue("B-12")).toBeDisabled();
  expect(screen.getByRole("button", { name: /Search/ })).toBeDisabled();
  expect(screen.getByText("Backend refused")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: /Search/ }));
  expect(onSearch).not.toHaveBeenCalled();
});

it("publicSearchKeepsItsExistingAdapter", async () => {
  const search = vi.spyOn(publicFaceSearchApi, "searchEventByFace").mockResolvedValue(found);
  const save = vi.spyOn(personalSelfieApi, "uploadSelfie").mockResolvedValue({ id: "saved" } as Awaited<ReturnType<typeof personalSelfieApi.uploadSelfie>>);
  const onSearchSuccess = vi.fn();
  render(<QueryClientProvider client={client}><FindPhotosModal eventSlug="race" eyebrow="Race" photoCount={10} eventPhotoCount={10} onClose={() => {}} onSubmitBib={() => {}} onSearchByFaceSuccess={onSearchSuccess} /></QueryClientProvider>);
  expect(screen.getByLabelText(/Take a selfie/)).toHaveAttribute("capture", "user");
  expect(screen.getByLabelText(/Upload a selfie/)).not.toHaveAttribute("capture");
  expect(screen.getByRole("checkbox")).toBeChecked();
  fireEvent.click(screen.getByRole("checkbox"));
  fireEvent.change(screen.getByLabelText(/Upload a selfie/), { target: { files: [file] } });
  await waitFor(() => expect(onSearchSuccess).toHaveBeenCalledWith(found));
  expect(search).toHaveBeenCalledWith("race", expect.objectContaining({ selfieFile: file }));
  expect(save).not.toHaveBeenCalled();
});

it("preserves public save-first and match-all through the shared file presentation", async () => {
  const search = vi.spyOn(publicFaceSearchApi, "searchEventByFace").mockResolvedValue(found);
  const save = vi.spyOn(personalSelfieApi, "uploadSelfie").mockResolvedValue({ id: "saved" } as Awaited<ReturnType<typeof personalSelfieApi.uploadSelfie>>);
  const success = vi.fn();
  render(<QueryClientProvider client={client}><FindPhotosModal eventSlug="race" eyebrow="Race" photoCount={10} eventPhotoCount={10} onClose={() => {}} onSubmitBib={() => {}} onSearchByFaceSuccess={success} /></QueryClientProvider>);
  fireEvent.change(screen.getByLabelText(/Take a selfie/), { target: { files: [file] } });
  await waitFor(() => expect(success).toHaveBeenCalledWith(found));
  expect(save).toHaveBeenCalledWith(file);
  expect(search).toHaveBeenCalledWith("race", { allSelfies: true });
});

it("keeps the public bib adapter and closes the portal on submit", () => {
  const bib = vi.fn(), close = vi.fn();
  render(<QueryClientProvider client={client}><FindPhotosModal eventSlug="race" eyebrow="Race" photoCount={10} eventPhotoCount={10} onClose={close} onSubmitBib={bib} onSearchByFaceSuccess={() => {}} /></QueryClientProvider>);
  fireEvent.click(screen.getByRole("button", { name: "Search by bib →" }));
  fireEvent.change(screen.getByLabelText("Enter your bib number"), { target: { value: "B-123" } });
  fireEvent.click(screen.getByRole("button", { name: "Search by bib →" }));
  expect(bib).toHaveBeenCalledWith("B-123"); expect(close).toHaveBeenCalledTimes(1);
});
