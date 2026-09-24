import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import FocusedUploadPage from "./page";
import { useAuthStore } from "@/store/auth-store";
import { OfflineUploadError } from "@/lib/api-photographer";

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

it("does not start an old queued file after an identity switch", async () => {
  useAuthStore.getState().setUser({ id: "first", role: "PHOTOGRAPHER", name: "First", email: "first@example.test", createdAt: "2026-09-01" });
  const signals: AbortSignal[] = [];
  upload.mockImplementation((_event, _file, _progress, options) => {
    signals.push(options.signal);
    return new Promise(() => {});
  });
  render(<FocusedUploadPage />);
  const files = Array.from({ length: 5 }, (_, index) => new File(["a"], `switch-${index}.jpg`, { type: "image/jpeg" }));
  fireEvent.change(screen.getByRole("button", { name: /browse photos/i }).parentElement!.querySelector("input")!, { target: { files } });
  await waitFor(() => expect(upload).toHaveBeenCalledTimes(4));
  await act(async () => {
    useAuthStore.getState().setUser({ id: "second", role: "PHOTOGRAPHER", name: "Second", email: "second@example.test", createdAt: "2026-09-01" });
  });
  expect(upload).toHaveBeenCalledTimes(4);
  expect(signals.every((signal) => signal.aborted)).toBe(true);
  expect(screen.queryByText("switch-4.jpg")).not.toBeInTheDocument();
  useAuthStore.getState().logout();
});

it("aborts a removed file's preflight and checks the remaining file again", async () => {
  const signals: AbortSignal[] = [];
  check.mockImplementation((_event, _hashes, signal) => {
    signals.push(signal);
    return new Promise(() => {});
  });
  render(<FocusedUploadPage />);
  const files = ["one.jpg", "two.jpg"].map((name) => new File([name], name, { type: "image/jpeg" }));
  fireEvent.change(screen.getByRole("button", { name: /browse photos/i }).parentElement!.querySelector("input")!, { target: { files } });
  await waitFor(() => expect(check).toHaveBeenCalledTimes(1));
  fireEvent.click(screen.getByRole("button", { name: "Remove one.jpg" }));
  await waitFor(() => expect(check).toHaveBeenCalledTimes(2));
  expect(signals[0].aborted).toBe(true);
  expect(signals[1].aborted).toBe(false);
});

it("aborts preflight when the batch is cleared or unmounted", async () => {
  const signals: AbortSignal[] = [];
  check.mockImplementation((_event, _hashes, signal) => {
    signals.push(signal);
    return new Promise(() => {});
  });
  const view = render(<FocusedUploadPage />);
  const input = screen.getByRole("button", { name: /browse photos/i }).parentElement!.querySelector("input")!;
  fireEvent.change(input, { target: { files: [new File(["a"], "clear.jpg", { type: "image/jpeg" })] } });
  await waitFor(() => expect(check).toHaveBeenCalledTimes(1));
  fireEvent.click(screen.getByRole("button", { name: /clear all/i }));
  expect(signals[0].aborted).toBe(true);
  fireEvent.change(input, { target: { files: [new File(["b"], "unmount.jpg", { type: "image/jpeg" })] } });
  await waitFor(() => expect(check).toHaveBeenCalledTimes(2));
  view.unmount();
  expect(signals[1].aborted).toBe(true);
});

it("restarts preflight when a removed checking file is replaced in one render", async () => {
  check.mockImplementation(() => new Promise(() => {}));
  render(<FocusedUploadPage />);
  const input = screen.getByRole("button", { name: /browse photos/i }).parentElement!.querySelector("input")!;
  fireEvent.change(input, { target: { files: [new File(["a"], "old.jpg", { type: "image/jpeg" })] } });
  await waitFor(() => expect(check).toHaveBeenCalledTimes(1));
  act(() => {
    fireEvent.click(screen.getByRole("button", { name: "Remove old.jpg" }));
    fireEvent.change(input, { target: { files: [new File(["b"], "new.jpg", { type: "image/jpeg" })] } });
  });
  await waitFor(() => expect(check).toHaveBeenCalledTimes(2));
});

it("reuses retry limits on reconnect but resets them for a manual retry", async () => {
  const states: unknown[] = [];
  upload.mockImplementation((_event, _file, _progress, options) => {
    states.push(options.retryState);
    if (states.length === 1) return Promise.reject(new OfflineUploadError());
    if (states.length === 2) return Promise.reject(new Error("network failed"));
    return new Promise(() => {});
  });
  render(<FocusedUploadPage />);
  fireEvent.change(screen.getByRole("button", { name: /browse photos/i }).parentElement!.querySelector("input")!, {
    target: { files: [new File(["a"], "recover.jpg", { type: "image/jpeg" })] },
  });
  await waitFor(() => expect(upload).toHaveBeenCalledTimes(1));
  await act(async () => { window.dispatchEvent(new Event("online")); });
  await waitFor(() => expect(upload).toHaveBeenCalledTimes(2));
  expect(states[0]).toBeDefined();
  expect(states[1]).toBe(states[0]);
  await waitFor(() => expect(screen.getByRole("button", { name: /try uploading recover.jpg again/i })).toBeInTheDocument());
  fireEvent.click(screen.getByRole("button", { name: /try uploading recover.jpg again/i }));
  await waitFor(() => expect(upload).toHaveBeenCalledTimes(3));
  expect(states[2]).not.toBe(states[1]);
});
