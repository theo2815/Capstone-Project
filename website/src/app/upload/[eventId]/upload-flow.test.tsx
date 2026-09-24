import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import FocusedUploadPage from "./page";

const { upload, check, hash } = vi.hoisted(() => ({ upload: vi.fn(), check: vi.fn(), hash: vi.fn() }));
vi.mock("next/navigation", () => ({ useParams: () => ({ eventId: "event-1" }), notFound: () => { throw new Error("not found"); }, unstable_rethrow: () => undefined }));
vi.mock("@/components/layout/site-header", () => ({ SiteHeader: () => null }));
vi.mock("@/components/dashboard/verification-banner", () => ({ VerificationBanner: () => null }));
vi.mock("@/hooks/use-can-upload", () => ({ useCanUpload: () => ({ kind: "ok" }) }));
vi.mock("@/lib/photographer-verification-sync", () => ({ usePhotographerVerificationSync: () => undefined }));
vi.mock("@/hooks/use-toast", () => ({ useToast: () => ({ showToast: vi.fn() }) }));
vi.mock("@/hooks/use-public-events", () => ({ usePublicEvents: () => [{
  id: "event-1", slug: "race", name: "Race", date: "2026-09-24", location: "Cebu",
  photoCount: 0, participantCount: 0, status: "ACTIVE", state: "live", city: "Cebu",
}] }));
vi.mock("@/hooks/use-photographer-data", () => ({ usePhotographerEventDetail: () => ({ detail: { ownedByMe: true }, isMissing: false }) }));
vi.mock("@/lib/image-utils", async (importOriginal) => ({ ...(await importOriginal<typeof import("@/lib/image-utils")>()), isHeicFile: async () => false }));
vi.mock("@/lib/api-photographer", async (importOriginal) => ({ ...(await importOriginal<typeof import("@/lib/api-photographer")>()), uploadPhotographerPhoto: upload, checkPhotosExist: check, sha256Hex: hash }));

beforeEach(() => {
  vi.stubGlobal("URL", Object.assign(class extends URL {}, { createObjectURL: vi.fn(() => "blob:test"), revokeObjectURL: vi.fn() }));
  check.mockResolvedValue([{ hash: "a".repeat(64), status: "new", eventName: null }]);
  hash.mockResolvedValue("a".repeat(64));
});
afterEach(() => { upload.mockReset(); check.mockReset(); hash.mockReset(); vi.unstubAllGlobals(); });

it("keeps the row confirming after byte progress and shows the backend processing receipt", async () => {
  let finish!: (value: unknown) => void;
  upload.mockImplementation((_event, _file, progress, options) => {
    options.onPhase("uploading");
    progress({ loaded: 1, total: 1, percent: 100 });
    options.onPhase("confirming");
    return new Promise((resolve) => { finish = resolve; });
  });
  render(<FocusedUploadPage />);
  fireEvent.change(screen.getByRole("button", { name: /browse photos/i }).parentElement!.querySelector("input")!, {
    target: { files: [new File(["a"], "race.jpg", { type: "image/jpeg" })] },
  });
  await waitFor(() => expect(upload).toHaveBeenCalledTimes(1));
  expect(upload.mock.calls[0][3].contentHash).toBe("a".repeat(64));
  expect(screen.getByText(/confirming/i)).toBeInTheDocument();
  expect(screen.queryByText(/^Live$/)).not.toBeInTheDocument();
  await act(async () => { finish({ id: "photo-1", status: "processing", uploadedAt: "now", thumbnailUrl: "", span: "default" }); });
  expect(screen.getByText("Uploaded — preparing preview")).toBeInTheDocument();
});

it("aborts an active workflow when its row is removed", async () => {
  const aborts: AbortSignal[] = [];
  upload.mockImplementation((_event, _file, _progress, options) => {
    aborts.push(options.signal);
    return new Promise(() => {});
  });
  render(<FocusedUploadPage />);
  fireEvent.change(screen.getByRole("button", { name: /browse photos/i }).parentElement!.querySelector("input")!, {
    target: { files: [new File(["a"], "race.jpg", { type: "image/jpeg" })] },
  });
  await waitFor(() => expect(upload).toHaveBeenCalledTimes(1));
  fireEvent.click(screen.getByRole("button", { name: "Remove race.jpg" }));
  expect(aborts[0].aborted).toBe(true);
});

it("holds four whole-workflow slots and frees one on removal", async () => {
  const signals = new Map<string, AbortSignal>();
  upload.mockImplementation((_event, file: File, _progress, options) => {
    signals.set(file.name, options.signal);
    return new Promise(() => {});
  });
  render(<FocusedUploadPage />);
  const files = Array.from({ length: 5 }, (_, index) => new File(["a"], `race-${index}.jpg`, { type: "image/jpeg" }));
  fireEvent.change(screen.getByRole("button", { name: /browse photos/i }).parentElement!.querySelector("input")!, { target: { files } });
  await waitFor(() => expect(upload).toHaveBeenCalledTimes(4));
  expect(signals.has("race-4.jpg")).toBe(false);
  fireEvent.click(screen.getByRole("button", { name: "Remove race-0.jpg" }));
  await waitFor(() => expect(upload).toHaveBeenCalledTimes(5));
  expect(signals.get("race-0.jpg")?.aborted).toBe(true);
  expect(signals.has("race-4.jpg")).toBe(true);
});
