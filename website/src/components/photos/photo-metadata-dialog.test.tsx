import { useRef, useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { api, ApiError } from "@/lib/api";
import type { PhotoMetadataDto } from "@/types/photo-metadata";
import { PhotoMetadataDialog } from "./photo-metadata-dialog";

const ready: PhotoMetadataDto = {
  photoId: "photo-a", status: "ready", originalFilename: "Cebu 🏃 " + "long-".repeat(30) + ".jpg", fileSizeBytes: 1_572_864,
  fields: {
    cameraMake: "Canon", cameraModel: "Canon EOS R6", lens: "RF 24–70mm", aperture: 2.8,
    exposureTime: "1/500", iso: 400, flash: null, exposureMode: "Manual", whiteBalance: "Daylight",
    author: "Aira <script>alert(1)</script>", copyright: "© 2026 Cebu", takenAtLocal: "2026-09-28T06:35:20", takenAtOffset: null,
  },
};
let client: QueryClient;
function Harness({ photoId = "photo-a", initialOpen = false }: { photoId?: string; initialOpen?: boolean }) {
  const [open, setOpen] = useState(initialOpen);
  const [mounted, setMounted] = useState(true);
  const triggerRef = useRef<HTMLButtonElement>(null);
  return <QueryClientProvider client={client}>
    <button ref={triggerRef} onClick={() => setOpen(true)}>Show Metadata</button>
    <button>Outside action</button>
    <button onClick={() => setMounted(false)}>Unmount popup</button>
    {mounted && <PhotoMetadataDialog photoId={photoId} access="public" isOpen={open} onClose={() => setOpen(false)} triggerRef={triggerRef} />}
  </QueryClientProvider>;
}
function setup() {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<Harness />);
}
afterEach(() => { cleanup(); client?.clear(); vi.restoreAllMocks(); });

it("names loading, hides previous values, and shows a single ordered column when ready", async () => {
  let resolve!: (value: PhotoMetadataDto) => void;
  vi.spyOn(api, "get").mockImplementation(() => new Promise((r) => { resolve = r as typeof resolve; }));
  const user = userEvent.setup(); setup();
  await user.click(screen.getByRole("button", { name: "Show Metadata" }));
  const dialog = screen.getByRole("dialog", { name: "Photo metadata" });
  expect(within(dialog).getByRole("status", { name: "Loading metadata" })).toBeInTheDocument();
  expect(within(dialog).queryByText("Canon EOS R6")).not.toBeInTheDocument();
  await act(async () => resolve(ready));
  expect(await within(dialog).findByText("Canon EOS R6")).toBeInTheDocument();
  expect(within(dialog).queryByText("Canon Canon EOS R6")).not.toBeInTheDocument();
  expect(within(dialog).getByText("Not recorded")).toBeInTheDocument();
  expect(within(dialog).getByText("Aira <script>alert(1)</script>")).toBeInTheDocument();
  expect(dialog.querySelector("script")).toBeNull();
  expect(within(dialog).getByText("06:35:20", { exact: false })).toBeInTheDocument();
  expect(dialog.textContent).not.toMatch(/UTC|Manila/);
  expect(within(dialog).getByText("1.5 MiB", { exact: false })).toHaveTextContent("1.5 MiB (1,572,864 bytes)");
  expect(dialog.querySelector(".sr-only")).toHaveTextContent("(1,572,864 bytes)");
  expect([...dialog.querySelectorAll("dt")].map((node) => node.textContent)).toEqual([
    "Model", "Lens", "Aperture", "Exposure", "ISO", "Flash", "Exposure mode", "White balance",
    "Author", "Copyright", "Date", "Name", "Original size",
  ]);
  expect(dialog.querySelector(".break-words")).toBeTruthy();
});

it.each([
  ["pending", "Metadata is being prepared.", true, true],
  ["empty", "No camera metadata was found.", false, true],
  ["unavailable", "Metadata is unavailable for this photo.", false, true],
  ["failed", "Metadata could not be loaded.", false, true],
] as const)("shows %s without stale camera values", async (status, message, refresh, fileFacts) => {
  vi.spyOn(api, "get").mockResolvedValue({ ...ready, status, fields: ready.fields });
  const user = userEvent.setup(); setup();
  await user.click(screen.getByRole("button", { name: "Show Metadata" }));
  const dialog = screen.getByRole("dialog");
  expect(await within(dialog).findByText(message)).toBeInTheDocument();
  expect(within(dialog).queryByText("Canon EOS R6")).not.toBeInTheDocument();
  expect(Boolean(within(dialog).queryByRole("button", { name: "Refresh" }))).toBe(refresh);
  expect(Boolean(within(dialog).queryByText(ready.originalFilename!))).toBe(fileFacts);
});

it("refreshes pending metadata manually", async () => {
  let resolve!: (value: PhotoMetadataDto) => void;
  const get = vi.spyOn(api, "get")
    .mockResolvedValueOnce({ ...ready, status: "pending", fields: null })
    .mockImplementationOnce(() => new Promise((r) => { resolve = r as typeof resolve; }));
  const user = userEvent.setup(); setup();
  await user.click(screen.getByRole("button", { name: "Show Metadata" }));
  await user.click(await screen.findByRole("button", { name: "Refresh" }));
  expect(screen.getByRole("status", { name: "Loading metadata" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Close metadata" })).toHaveFocus();
  await act(async () => resolve(ready));
  expect(await screen.findByText("Canon EOS R6")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Close metadata" })).toHaveFocus();
  expect(get).toHaveBeenCalledTimes(2);
});

it("clears the old photo during a new load and preserves a supplied capture offset", async () => {
  let resolve!: (value: PhotoMetadataDto) => void;
  vi.spyOn(api, "get").mockResolvedValueOnce(ready).mockImplementationOnce(() => new Promise((r) => { resolve = r as typeof resolve; }));
  const user = userEvent.setup();
  const view = setup();
  await user.click(screen.getByRole("button", { name: "Show Metadata" }));
  expect(await screen.findByText("Canon EOS R6")).toBeInTheDocument();
  view.rerender(<Harness photoId="photo-b" />);
  expect(screen.getByRole("status", { name: "Loading metadata" })).toBeInTheDocument();
  expect(screen.queryByText("Canon EOS R6")).not.toBeInTheDocument();
  await act(async () => resolve({ ...ready, photoId: "photo-b", fields: { ...ready.fields!, takenAtOffset: "+08:00" } }));
  expect(await screen.findByText("2026-09-28 06:35:20 +08:00")).toBeInTheDocument();
});

it("shows a retry for network failure and a distinct public 404 without stale values", async () => {
  let resolveRetry!: (value: PhotoMetadataDto) => void;
  const get = vi.spyOn(api, "get")
    .mockRejectedValueOnce(new Error("offline"))
    .mockImplementationOnce(() => new Promise((r) => { resolveRetry = r as typeof resolveRetry; }));
  const user = userEvent.setup(); setup();
  await user.click(screen.getByRole("button", { name: "Show Metadata" }));
  expect(await screen.findByText("Metadata could not be loaded.")).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Retry" }));
  expect(screen.getByRole("status", { name: "Loading metadata" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Close metadata" })).toHaveFocus();
  await act(async () => resolveRetry(ready));
  expect(await screen.findByText("Canon EOS R6")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Close metadata" })).toHaveFocus();
  expect(get).toHaveBeenCalledTimes(2);
  await user.click(screen.getByRole("button", { name: "Close metadata" }));
  get.mockRejectedValueOnce(new ApiError([{ code: "NOT_FOUND", message: "Gone" }], 404));
  await user.click(screen.getByRole("button", { name: "Show Metadata" }));
  expect(await screen.findByText("Metadata is no longer available")).toBeInTheDocument();
  expect(screen.queryByText("Canon EOS R6")).not.toBeInTheDocument();
});

it("contains Tab and Shift+Tab, and restores trigger focus on close, Escape, backdrop, and unmount", async () => {
  vi.spyOn(api, "get").mockResolvedValue({ ...ready, status: "pending", fields: null });
  const user = userEvent.setup();
  const view = setup();
  const trigger = screen.getByRole("button", { name: "Show Metadata" });
  await user.click(trigger);
  await waitFor(() => expect(screen.getByRole("button", { name: "Refresh" })).toBeInTheDocument());
  expect(screen.getByRole("button", { name: "Close metadata" })).toHaveFocus();
  await user.tab({ shift: true });
  expect(screen.getByRole("button", { name: "Refresh" })).toHaveFocus();
  expect(screen.getByRole("dialog").contains(document.activeElement)).toBe(true);
  await user.tab();
  expect(screen.getByRole("button", { name: "Close metadata" })).toHaveFocus();
  await user.keyboard("{Escape}");
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(trigger).toHaveFocus();
  await user.click(trigger);
  await user.click(screen.getByRole("dialog").previousElementSibling as HTMLElement);
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(trigger).toHaveFocus();
  await user.click(trigger);
  await user.click(screen.getByRole("button", { name: "Unmount popup" }));
  expect(trigger).toHaveFocus();
  view.unmount();
});

it("focuses Close metadata when first mounted already open", async () => {
  vi.spyOn(api, "get").mockResolvedValue({ ...ready, status: "pending", fields: null });
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<Harness initialOpen />);
  expect(await screen.findByRole("button", { name: "Close metadata" })).toHaveFocus();
});
