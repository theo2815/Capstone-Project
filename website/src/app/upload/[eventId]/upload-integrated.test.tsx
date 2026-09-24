import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import FocusedUploadPage from "./page";
import { useAuthStore } from "@/store/auth-store";

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

const response = (data: unknown, status = 200) => new Response(JSON.stringify(status < 400
  ? { success: true, data }
  : { success: false, errors: [{ code: "UPLOAD_OBJECT_MISSING", message: "Not stored" }] }), { status });

let puts: TransferXhr[];
let nextId: number;
let fetchMock: ReturnType<typeof vi.fn>;
let online: boolean;

class TransferXhr {
  static all: TransferXhr[] = [];
  status = 200;
  responseText = "";
  withCredentials = true;
  upload = { onprogress: null };
  onload?: () => void;
  onerror?: () => void;
  onabort?: () => void;
  method = "";
  aborted = false;
  open(method: string) { this.method = method; }
  setRequestHeader() {}
  send() { puts.push(this); TransferXhr.all.push(this); }
  abort() { this.aborted = true; this.onabort?.(); }
  finish(status = 200) { this.status = status; this.onload?.(); }
  failOffline() { online = false; this.onerror?.(); }
}

beforeEach(() => {
  puts = [];
  nextId = 0;
  online = true;
  TransferXhr.all = [];
  vi.stubEnv("NEXT_PUBLIC_R2_UPLOAD_ORIGIN", "https://uploads.example.test");
  const originalCrypto = globalThis.crypto;
  const NativeFile = globalThis.File;
  vi.stubGlobal("File", class extends NativeFile {
    arrayBuffer() { return Promise.resolve(new Uint8Array([1]).buffer); }
  });
  vi.stubGlobal("crypto", {
    subtle: { digest: async () => new Uint8Array(32).buffer },
    randomUUID: () => `request-${++nextId}`,
    getRandomValues: originalCrypto.getRandomValues.bind(originalCrypto),
  });
  vi.stubGlobal("URL", Object.assign(class extends URL {}, { createObjectURL: vi.fn(() => "blob:test"), revokeObjectURL: vi.fn() }));
  vi.stubGlobal("XMLHttpRequest", TransferXhr);
  vi.spyOn(window.navigator, "onLine", "get").mockImplementation(() => online);
  fetchMock = vi.fn(async (url: string) => {
    if (url.endsWith("/photos/exists")) return response({ results: [] });
    if (url.endsWith("/photos/direct")) {
      const id = ++nextId;
      return response({ mode: "direct", photoId: `photo-${id}`, key: `key-${id}`, uploadUrl: `https://uploads.example.test/${id}`, expiresAt: "2099-01-01T00:00:00Z" });
    }
    if (url.endsWith("/photos/direct/commit")) return response({ id: "saved", status: "processing" });
    throw new Error(`Unexpected fetch ${url}`);
  });
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  useAuthStore.getState().logout();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

function select(files: File[]) {
  fireEvent.change(screen.getByRole("button", { name: /browse photos/i }).parentElement!.querySelector("input")!, { target: { files } });
}

it("aborts four real transfers on session replacement without sending the queued fifth", async () => {
  useAuthStore.getState().setUser({ id: "first", role: "PHOTOGRAPHER", name: "First", email: "first@example.test", createdAt: "2026-09-01" });
  render(<FocusedUploadPage />);
  select(Array.from({ length: 5 }, (_, index) => new File([`${index}`], `switch-${index}.jpg`, { type: "image/jpeg" })));
  await waitFor(() => expect(puts).toHaveLength(4));
  await act(async () => {
    useAuthStore.getState().setUser({ id: "second", role: "PHOTOGRAPHER", name: "Second", email: "second@example.test", createdAt: "2026-09-01" });
  });
  expect(puts).toHaveLength(4);
  expect(puts.every((put) => put.aborted)).toBe(true);
  expect(screen.queryByText("switch-4.jpg")).not.toBeInTheDocument();
  // A late completion from the old transfer cannot create a row in the new batch.
  await act(async () => { puts[0].finish(); });
  expect(screen.queryByText("switch-0.jpg")).not.toBeInTheDocument();
  select([new File(["new"], "new.jpg", { type: "image/jpeg" })]);
  await waitFor(() => expect(puts).toHaveLength(5));
  expect(screen.getByText("new.jpg")).toBeInTheDocument();
});

it("aborts real transfers on clear and unmount", async () => {
  const view = render(<FocusedUploadPage />);
  select([new File(["a"], "clear.jpg", { type: "image/jpeg" })]);
  await waitFor(() => expect(puts).toHaveLength(1));
  fireEvent.click(screen.getByRole("button", { name: /clear all/i }));
  expect(puts[0].aborted).toBe(true);
  select([new File(["b"], "unmount.jpg", { type: "image/jpeg" })]);
  await waitFor(() => expect(puts).toHaveLength(2));
  view.unmount();
  expect(puts[1].aborted).toBe(true);
});

it("resumes an offline transfer through its saved receipt before resending bytes", async () => {
  let now = Date.now();
  vi.spyOn(Date, "now").mockImplementation(() => now);
  const nativeTimer = globalThis.setTimeout;
  const caps: number[] = [];
  vi.spyOn(globalThis, "setTimeout").mockImplementation(((callback: TimerHandler, ms?: number, ...args: unknown[]) => {
    if (ms && ms >= 800_000) caps.push(ms);
    return nativeTimer(callback, ms, ...args);
  }) as typeof setTimeout);
  let commits = 0;
  const beginIds: string[] = [];
  fetchMock.mockImplementation(async (url: string, request: RequestInit) => {
    if (url.endsWith("/photos/exists")) return response({ results: [] });
    if (url.endsWith("/photos/direct/commit")) {
      commits++;
      return commits === 1 ? response(null, 422) : response({ id: "saved", status: "processing" });
    }
    if (url.endsWith("/photos/direct")) {
      beginIds.push(JSON.parse(request.body as string).requestId);
      return response({ mode: "direct", photoId: "saved", key: "key", uploadUrl: "https://uploads.example.test/key", expiresAt: "2099-01-01T00:00:00Z" });
    }
    throw new Error(`Unexpected fetch ${url}`);
  });
  render(<FocusedUploadPage />);
  select([new File(["a"], "offline.jpg", { type: "image/jpeg" })]);
  await waitFor(() => expect(puts).toHaveLength(1));
  await act(async () => { puts[0].failOffline(); });
  expect(screen.getByText(/waiting for connection/i)).toBeInTheDocument();
  now += 60_000;
  online = true;
  await act(async () => { window.dispatchEvent(new Event("online")); });
  await waitFor(() => expect(puts).toHaveLength(2));
  expect(commits).toBe(1);
  expect(beginIds).toHaveLength(2);
  expect(beginIds[1]).toBe(beginIds[0]);
  expect(caps).toEqual([900_000, 840_000]);
  await act(async () => { puts[1].finish(); });
  await waitFor(() => expect(screen.getByText("Uploaded — preparing preview")).toBeInTheDocument());
  expect(commits).toBe(2);
});

it("does not duplicate a manual retry when its button is activated twice", async () => {
  let begins = 0;
  fetchMock.mockImplementation(async (url: string) => {
    if (url.endsWith("/photos/exists")) return response({ results: [] });
    if (url.endsWith("/photos/direct")) {
      begins++;
      return begins === 1
        ? response({ mode: "direct" })
        : response({ mode: "direct", photoId: "saved", key: "key", uploadUrl: "https://uploads.example.test/key" });
    }
    if (url.endsWith("/photos/direct/commit")) return response({ id: "saved", status: "processing" });
    throw new Error(`Unexpected fetch ${url}`);
  });
  render(<FocusedUploadPage />);
  select([new File(["a"], "retry.jpg", { type: "image/jpeg" })]);
  const button = await screen.findByRole("button", { name: /try uploading retry.jpg again/i });
  fireEvent.click(button);
  fireEvent.click(button);
  await waitFor(() => expect(puts).toHaveLength(1));
  expect(begins).toBe(2);
  await act(async () => { puts[0].finish(); });
  await waitFor(() => expect(screen.getByText("Uploaded — preparing preview")).toBeInTheDocument());
  expect(puts).toHaveLength(1);
});
