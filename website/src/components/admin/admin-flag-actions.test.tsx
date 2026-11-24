import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { api, ApiError } from "@/lib/api";
import type { Flag } from "@/lib/admin-flags";
import { admin } from "@/test/fixtures/admin";
import { useToastStore } from "@/store/toast-store";
import { useAuthStore } from "@/store/auth-store";
import { AdminFlagCard, type FlagCardActions } from "./admin-flag-card";
import { FlagsQueue } from "./flags-queue";
import { AdminFlagHideModal } from "./admin-flag-hide-modal";

const navigation = vi.hoisted(() => ({ search: "", replace: vi.fn() }));
vi.mock("next/navigation", () => ({ useRouter: () => ({ replace: navigation.replace }), usePathname: () => "/admin/flags", useSearchParams: () => new URLSearchParams(navigation.search) }));
vi.mock("@/lib/constants", async (original) => ({ ...await original<object>(), ADMIN_FLAGS_ENABLED: true }));
const flag: Flag = { id: "flag-1", photoId: "photo-1", eventId: "event-1", eventName: "Race", photographerHandle: "photographer", reportedBy: "runner", reason: "low_quality", note: "Blurry", status: "open", reportedAt: "2026-09-10T00:00:00Z", reviewedAt: null, reviewedBy: null, reviewerNote: null, photoSnapshot: { alt: "Runner", kmMark: null, bib: null, thumbnailUrl: null }, targetKind: "photo", targetId: "photo-1", payoutHoldAt: null };
const relevanceFlag: Flag = { ...flag, id: "flag-2", photoId: null, reason: "batch_relevance_low", note: "event e1: copy 5/8", targetKind: "user", targetId: "user-1" };
const inertActions: FlagCardActions = { isPending: () => false, pendingAction: () => null, onHide: vi.fn(), onEscalate: vi.fn(), onConfirm: vi.fn(async () => true) };
// The flag list honours `status`, as the server does: the history slabs ask by status (AX-33).
const serveFlags = (...flags: Flag[]) => vi.mocked(api.get).mockImplementation((async (url: string) => {
  if (url.includes("/relevance")) return { flagId: "flag-2", payoutHoldAt: null, batches: [] };
  if (url.includes("/hidden-photos")) return { items: [], total: 0 };
  const status = new URL(url, "http://test").searchParams.get("status");
  const items = flags.filter((f) => !status || f.status === status);
  return { items, total: items.length };
}) as typeof api.get);
let client: QueryClient;
beforeEach(() => {
  useAuthStore.getState().setUser(admin);
  useToastStore.getState().clear();
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  navigation.search = "";
  Object.defineProperty(HTMLElement.prototype, "scrollIntoView", { configurable: true, value: vi.fn() });
  vi.spyOn(api, "get");
  serveFlags(flag);
});
afterEach(() => { cleanup(); client.clear(); useAuthStore.getState().logout(); vi.restoreAllMocks(); });

it.each(["card", "drawer"])("%s retains flag reasons on failure and blocks repeat click, Enter and Escape", async (surface) => {
  const user = userEvent.setup();
  let reject!: (error: Error) => void;
  const post = vi.spyOn(api, "post").mockReturnValueOnce(new Promise((_, no) => { reject = no; }));
  render(<QueryClientProvider client={client}><FlagsQueue /></QueryClientProvider>);
  if (surface === "drawer") await user.click(await screen.findByRole("button", { description: "flag-1" }));
  const scope = surface === "drawer" ? within(await screen.findByRole("dialog")) : screen;
  await user.click((await scope.findAllByRole("button", { name: "Hide…" }))[0]!);
  const dialog = screen.getByRole("dialog", { name: "Hide flag-1" });
  await user.click(within(dialog).getByRole("radio", { name: "Violates platform policy" }));
  await user.type(within(dialog).getByRole("textbox"), "Keep this explanation");
  const submit = within(dialog).getByRole("button", { name: "Hide photo" });
  await user.click(submit);
  await waitFor(() => expect(post).toHaveBeenCalledTimes(1));
  expect(dialog).toBeInTheDocument();
  expect(submit).toHaveAttribute("aria-busy", "true");
  expect(submit).toHaveTextContent("Hiding");
  expect(screen.getAllByRole("button", { name: "Hide…" }).every((button) => button.hasAttribute("disabled"))).toBe(true);
  fireEvent.submit(dialog.querySelector("form")!);
  await user.keyboard("{Escape}h");
  expect(post).toHaveBeenCalledTimes(1);
  await act(async () => { reject(new Error("Unavailable")); });
  expect(within(dialog).getByRole("textbox")).toHaveValue("Keep this explanation");
  await waitFor(() => expect(submit).toBeEnabled());
  post.mockResolvedValue(flag);
  await user.click(submit);
  await waitFor(() => expect(dialog).not.toBeInTheDocument());
});

it("distinguishes an accepted hide from a failed queue refresh", async () => {
  const post = vi.spyOn(api, "post").mockResolvedValue({ ...flag, status: "hidden" });
  render(<QueryClientProvider client={client}><FlagsQueue /></QueryClientProvider>);
  await userEvent.click((await screen.findAllByRole("button", { name: "Hide…" }))[0]);
  const dialog = screen.getByRole("dialog", { name: "Hide flag-1" });
  vi.mocked(api.get).mockRejectedValue(new Error("Refresh unavailable"));
  await userEvent.click(within(dialog).getByRole("radio", { name: "Violates platform policy" }));
  await userEvent.click(within(dialog).getByRole("button", { name: "Hide photo" }));
  await waitFor(() => expect(dialog).not.toBeInTheDocument());
  expect(post).toHaveBeenCalledTimes(1);
  expect(useToastStore.getState().toasts.some((toast) => /saved.*could not refresh/.test(toast.message))).toBe(true);
  expect(useToastStore.getState().toasts.some((toast) => toast.kind === "error")).toBe(false);
});

// FT-11 (C-67): no reason starts picked, so a quick Enter never hides with one the admin did not choose.
it("the Hide dialog opens with no reason, says nobody is told, and Other needs a note", async () => {
  const user = userEvent.setup();
  const post = vi.spyOn(api, "post").mockResolvedValue({ ...flag, status: "hidden" });
  render(<QueryClientProvider client={client}><FlagsQueue /></QueryClientProvider>);
  await user.click((await screen.findAllByRole("button", { name: "Hide…" }))[0]!);
  const dialog = screen.getByRole("dialog", { name: "Hide flag-1" });
  expect(within(dialog).getByText("The photo is hidden from search and galleries. Nobody is told automatically: the reason is saved on the flag for admins. To tell the photographer, send them a message from their page.")).toBeInTheDocument();
  expect(within(dialog).getAllByRole("radio").some((radio) => (radio as HTMLInputElement).checked)).toBe(false);
  const submit = within(dialog).getByRole("button", { name: "Hide photo" });
  expect(submit).toBeDisabled();
  const note = within(dialog).getByRole("textbox");
  expect(note).toHaveAttribute("placeholder", "Anything another admin should know");
  await user.type(note, "x");
  fireEvent.submit(dialog.querySelector("form")!);
  expect(post).not.toHaveBeenCalled();
  await user.clear(note);
  await user.click(within(dialog).getByRole("radio", { name: "Other (see note)" }));
  expect(within(dialog).getByLabelText("Note (required, max 280)")).toBe(note);
  fireEvent.submit(dialog.querySelector("form")!);
  expect(await within(dialog).findByText("Add a note: Other needs one.")).toBeInTheDocument();
  expect(post).not.toHaveBeenCalled();
  await user.type(note, "Shows a bib number");
  await user.click(submit);
  await waitFor(() => expect(post).toHaveBeenCalledTimes(1));
  expect(JSON.stringify(post.mock.calls[0])).toContain("Other (see note) — Shows a bib number");
});

const refusal = () => new ApiError([{ code: "INVALID_STATE_TRANSITION", message: "This flag is already closed." }], 409);

it("a refused hold shows the server's reason and can be retried", async () => {
  const user = userEvent.setup();
  const invalidate = vi.spyOn(client, "invalidateQueries");
  const post = vi.spyOn(api, "post").mockRejectedValueOnce(refusal());
  serveFlags(relevanceFlag);
  render(<QueryClientProvider client={client}><FlagsQueue /></QueryClientProvider>);
  await user.click(await screen.findByRole("button", { name: "Hold payouts" }));
  expect(screen.getByText(/Holds this photographer's payouts until you resolve or dismiss this flag/)).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Confirm hold" }));
  await waitFor(() => expect(useToastStore.getState().toasts.some((t) => t.kind === "error" && t.message === "This flag is already closed.")).toBe(true));
  expect(screen.getByRole("button", { name: "Confirm hold" })).toBeEnabled();
  post.mockResolvedValueOnce({ ...relevanceFlag, payoutHoldAt: "2026-11-13T00:00:00Z" });
  await user.click(screen.getByRole("button", { name: "Confirm hold" }));
  await waitFor(() => expect(screen.queryByRole("button", { name: "Confirm hold" })).not.toBeInTheDocument());
  expect(post).toHaveBeenCalledTimes(2);
  expect(post).toHaveBeenLastCalledWith("/admin/flags/flag-2/payout-hold");
  expect(invalidate).toHaveBeenCalledWith({ queryKey: ["admin", "payouts"] }, { throwOnError: true });
});

it.each([
  ["Resolve", "Confirm resolve", "/admin/flags/flag-2/resolve"],
  ["Dismiss", "Confirm dismiss", "/admin/flags/flag-2/dismiss"],
])("%s on a relevance flag confirms first, shows a refusal and then goes through", async (open, confirm, url) => {
  const user = userEvent.setup();
  const post = vi.spyOn(api, "post").mockRejectedValueOnce(refusal()).mockResolvedValue(relevanceFlag);
  serveFlags(relevanceFlag);
  render(<QueryClientProvider client={client}><FlagsQueue /></QueryClientProvider>);
  await user.click(await screen.findByRole("button", { name: open }));
  expect(post).not.toHaveBeenCalled();
  await user.click(screen.getByRole("button", { name: confirm }));
  await waitFor(() => expect(useToastStore.getState().toasts.some((t) => t.message === "This flag is already closed.")).toBe(true));
  await user.click(screen.getByRole("button", { name: confirm }));
  await waitFor(() => expect(screen.queryByRole("button", { name: confirm })).not.toBeInTheDocument());
  expect(post).toHaveBeenLastCalledWith(url, { resolutionNote: null });
  expect(useToastStore.getState().toasts.some((t) => t.kind === "success")).toBe(true);
});

it("buttons inside a clickable card do not open it or swallow other keys", async () => {
  const onOpen = vi.fn();
  const seen = vi.fn((e: KeyboardEvent) => e.key);
  window.addEventListener("keydown", seen);
  render(<QueryClientProvider client={client}><AdminFlagCard flag={relevanceFlag} actions={inertActions} onOpen={onOpen} /></QueryClientProvider>);
  const button = screen.getByRole("button", { name: "Hold payouts" });
  for (const key of ["Enter", " "]) fireEvent.keyDown(button, { key });
  await userEvent.click(button);
  expect(onOpen).not.toHaveBeenCalled();
  // Enter on a control is not defaultPrevented, so the browser still clicks it.
  expect(fireEvent.keyDown(button, { key: "Enter" })).toBe(true);
  expect(fireEvent.keyDown(button, { key: "j" })).toBe(true);
  expect(seen.mock.calls.map(([e]) => e.key)).toContain("j");
  // The card itself still opens on Enter and Space, through its title button.
  screen.getByRole("button", { description: "flag-2" }).focus();
  await userEvent.keyboard("{Enter} ");
  expect(onOpen).toHaveBeenCalledTimes(2);
  window.removeEventListener("keydown", seen);
});

it("the drawer of a relevance flag holds payouts, shows a refusal and keeps the drawer open", async () => {
  const user = userEvent.setup();
  serveFlags(relevanceFlag);
  const post = vi.spyOn(api, "post").mockRejectedValueOnce(refusal()).mockResolvedValue({ ...relevanceFlag, payoutHoldAt: "2026-11-13T00:00:00Z" });
  render(<QueryClientProvider client={client}><FlagsQueue /></QueryClientProvider>);
  await user.click(await screen.findByRole("button", { description: "flag-2" }));
  const drawer = within(await screen.findByRole("dialog"));
  expect(await drawer.findByText("No batch is linked to this flag.")).toBeInTheDocument();
  expect(drawer.queryByRole("button", { name: "Hide…" })).not.toBeInTheDocument();
  await user.click(drawer.getByRole("button", { name: "Hold payouts" }));
  expect(drawer.getByText(/Holds this photographer's payouts/)).toBeInTheDocument();
  await user.click(drawer.getByRole("button", { name: "Confirm hold" }));
  await waitFor(() => expect(useToastStore.getState().toasts.some((t) => t.message === "This flag is already closed.")).toBe(true));
  await user.click(drawer.getByRole("button", { name: "Confirm hold" }));
  await waitFor(() => expect(post).toHaveBeenLastCalledWith("/admin/flags/flag-2/payout-hold"));
  expect(screen.getByRole("dialog")).toBeInTheDocument();
});

// FT-33 (C-31; W33.2): Hide photo stays off until a reason is picked, and the dialog says so.
it("the Hide dialog says why Hide photo is off until a reason is picked", async () => {
  const user = userEvent.setup();
  render(<AdminFlagHideModal open onClose={() => {}} onSubmit={() => {}} flagId="flag-1" />);
  expect(screen.getByRole("button", { name: "Hide photo" })).toBeDisabled();
  expect(screen.getByText("Choose a reason to continue.")).toBeInTheDocument();
  await user.click(screen.getByRole("radio", { name: "Violates platform policy" }));
  expect(screen.queryByText("Choose a reason to continue.")).not.toBeInTheDocument();
});
