import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useAuthStore } from "@/store/auth-store";
import { resetUserScopedStores } from "@/lib/auth-reset";
import { postSocial, withdrawVerification } from "@/lib/api-photographer-settings";
import SettingsPage from "./page";
import { DEFAULT_WATERMARK_PLACEMENT, usePhotographerSettingsStore } from "@/store/photographer-settings-store";
import { putHandle, putWatermarkSettings, postPayoutAccount, patchPayoutAccount, postPayoutQr } from "@/lib/api-photographer-settings";

const mocks = vi.hoisted(() => ({ toast: vi.fn(), confirm: vi.fn(), regions: [] }));
vi.mock("@/hooks/use-toast", () => ({ useToast: () => ({ showToast: mocks.toast }) }));
vi.mock("@/hooks/use-confirmation", () => ({ useConfirmation: () => ({ confirm: mocks.confirm }) }));
vi.mock("@/hooks/use-auth", () => ({ useAuth: () => ({ user: null }) }));
vi.mock("@/hooks/use-regions", () => ({ useRegions: () => ({ regions: mocks.regions, isLoading: false }) }));
vi.mock("@/lib/photographer-verification-sync", () => ({ syncVerificationNow: vi.fn() }));
vi.mock("@/lib/api-photographer-settings", async (importActual) => ({
  ...await importActual<typeof import("@/lib/api-photographer-settings")>(),
  postSocial: vi.fn(), withdrawVerification: vi.fn(), submitVerification: vi.fn(),
  putHandle: vi.fn(),
  putWatermarkSettings: vi.fn(),
  postPayoutAccount: vi.fn(), patchPayoutAccount: vi.fn(), postPayoutQr: vi.fn(),
}));

const photographer = { id: "photographer", role: "PHOTOGRAPHER" as const, name: "Photo", email: "photo@example.test", createdAt: "2026-09-10" };

beforeEach(() => {
  useAuthStore.getState().setUser(photographer);
  vi.clearAllMocks();
  vi.spyOn(console, "error").mockImplementation(() => {}); // expected failed handle request
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  vi.spyOn(HTMLElement.prototype, "clientWidth", "get").mockReturnValue(400);
  vi.spyOn(HTMLElement.prototype, "clientHeight", "get").mockReturnValue(266);
  usePhotographerSettingsStore.getState().reset();
  usePhotographerSettingsStore.setState({
    verificationStatus: "approved", verificationMissing: [], verificationMissingCodes: [],
    handle: "old-handle", watermark: { dataUrl: "data:image/png;base64,AA==", uploadedAt: "" },
  });
  vi.mocked(putWatermarkSettings).mockResolvedValue();
  vi.mocked(putHandle).mockRejectedValue(new Error("Handle already taken"));
});

const payout = { id: "account", method: "gcash" as const, accountName: "Test Recipient", accountNumber: "09171234567", isPrimary: true, qr: null };
const qr = { dataUrl: "data:image/png;base64,AA==", uploadedAt: "2026-09-09T00:00:00Z" };

it("requires a QR before adding an account and invalidates it when its number changes", () => {
  usePhotographerSettingsStore.setState({ payouts: [{ ...payout, qr }] });
  const { container } = render(<SettingsPage />);
  fireEvent.click(screen.getByRole("button", { name: /Edit settings/ }));
  const section = within(container.querySelector("#payout") as HTMLElement);
  fireEvent.click(section.getByRole("button", { name: "Edit" }));
  expect(section.getByRole("button", { name: /Update account/ })).not.toBeDisabled();
  fireEvent.change(section.getByLabelText("Mobile number"), { target: { value: "09170000002" } });
  expect(section.getByRole("button", { name: /Update account/ })).toBeDisabled();
  expect(section.getByText("QR code (required)")).toBeInTheDocument();
  expect(section.getByLabelText(/Upload QR/)).toHaveAttribute("aria-required", "true");
});

it("keeps a failed required QR upload retryable without duplicating the account", async () => {
  vi.mocked(postPayoutAccount).mockResolvedValue(payout);
  vi.mocked(postPayoutQr).mockRejectedValueOnce(new Error("Upload failed")).mockResolvedValue({ ...payout, qr });
  render(<SettingsPage />);
  fireEvent.click(screen.getByRole("button", { name: /Edit settings/ }));
  act(() => usePhotographerSettingsStore.setState({ payouts: [{ ...payout, id: "local", qr }] }));
  fireEvent.click(screen.getByRole("button", { name: /Save changes/ }));
  await waitFor(() => expect(postPayoutQr).toHaveBeenCalledTimes(1));
  await waitFor(() => expect(screen.getByRole("button", { name: /Save changes/ })).not.toBeDisabled());
  expect(usePhotographerSettingsStore.getState().payouts[0].id).toBe("account");
  fireEvent.click(screen.getByRole("button", { name: /Save changes/ }));
  await waitFor(() => expect(screen.getByRole("button", { name: /Edit settings/ })).toBeInTheDocument());
  expect(postPayoutAccount).toHaveBeenCalledTimes(1);
  expect(postPayoutQr).toHaveBeenCalledTimes(2);
});

it("saves a changed number before its QR and only retries the failed upload", async () => {
  usePhotographerSettingsStore.setState({ payouts: [{ ...payout, qr }] });
  const next = { ...payout, accountNumber: "09170000002", qr: null };
  let finishPatch!: (value: typeof next) => void;
  vi.mocked(patchPayoutAccount).mockImplementation(() => new Promise((resolve) => { finishPatch = resolve; }));
  vi.mocked(postPayoutQr).mockRejectedValueOnce(new Error("Upload failed")).mockResolvedValue({ ...next, qr });
  render(<SettingsPage />);
  fireEvent.click(screen.getByRole("button", { name: /Edit settings/ }));
  act(() => usePhotographerSettingsStore.setState({ payouts: [{ ...next, qr: { ...qr, dataUrl: "data:image/png;base64,AQ==" } }] }));
  fireEvent.click(screen.getByRole("button", { name: /Save changes/ }));
  expect(postPayoutQr).not.toHaveBeenCalled();
  await act(async () => { finishPatch(next); });
  await waitFor(() => expect(postPayoutQr).toHaveBeenCalledTimes(1));
  fireEvent.click(screen.getByRole("button", { name: /Save changes/ }));
  await waitFor(() => expect(screen.getByRole("button", { name: /Edit settings/ })).toBeInTheDocument());
  expect(patchPayoutAccount).toHaveBeenCalledTimes(1);
  expect(postPayoutQr).toHaveBeenCalledTimes(2);
});

afterEach(() => {
  cleanup();
  useAuthStore.getState().logout();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

it("keeps a saved placement on retry and Cancel when another settings request fails", async () => {
  render(<SettingsPage />);
  fireEvent.click(screen.getByRole("button", { name: /Edit settings/ }));
  const saved = {
    ...DEFAULT_WATERMARK_PLACEMENT, scale: 0.3, opacity: 0.5,
    landscape: { x: 0.2, y: 0.4 }, portrait: { x: 0.6, y: 0.8 },
  };
  act(() => usePhotographerSettingsStore.setState({ handle: "taken-handle", watermarkPlacement: saved }));
  fireEvent.click(screen.getByRole("button", { name: /Save changes/ }));
  await waitFor(() => expect(putWatermarkSettings).toHaveBeenCalledWith({
    scale: 0.3, opacity: 0.5, landscapeX: 0.2, landscapeY: 0.4, portraitX: 0.6, portraitY: 0.8,
  }));
  await waitFor(() => expect(screen.getByRole("button", { name: "Cancel" })).not.toBeDisabled());

  // Retrying the failed handle must not resend the successful placement PUT.
  fireEvent.click(screen.getByRole("button", { name: /Save changes/ }));
  await waitFor(() => expect(putHandle).toHaveBeenCalledTimes(2));
  await waitFor(() => expect(screen.getByRole("button", { name: "Cancel" })).not.toBeDisabled());
  expect(putWatermarkSettings).toHaveBeenCalledTimes(1);

  // Discard a subsequent unsaved move; Cancel restores the server's latest value.
  act(() => usePhotographerSettingsStore.setState({
    watermarkPlacement: { ...saved, scale: 0.4, landscape: { x: 0, y: 0 } },
  }));
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(usePhotographerSettingsStore.getState().watermarkPlacement).toEqual(saved);
  expect(usePhotographerSettingsStore.getState().handle).toBe("old-handle");
});


it("drops a stale account-create response before committing its ID or uploading its QR", async () => {
  let finish!: (value: typeof payout) => void;
  vi.mocked(postPayoutAccount).mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }));
  render(<SettingsPage />);
  fireEvent.click(screen.getByRole("button", { name: /Edit settings/ }));
  act(() => usePhotographerSettingsStore.setState({ payouts: [{ ...payout, id: "local", qr }] }));
  fireEvent.click(screen.getByRole("button", { name: /Save changes/ }));
  expect(screen.getByRole("button", { name: /Saving/ })).toBeDisabled();
  act(() => {
    useAuthStore.getState().logout(); resetUserScopedStores();
    useAuthStore.getState().setUser(photographer);
  });
  await act(async () => { finish(payout); });
  expect(usePhotographerSettingsStore.getState().payouts).toEqual([]);
  expect(postPayoutQr).not.toHaveBeenCalled();
  expect(mocks.toast).not.toHaveBeenCalled();
  expect(screen.queryByRole("button", { name: /Saving/ })).not.toBeInTheDocument();
});

it.each(["success", "failure"])("does not apply a stale Settings batch or toast (%s)", async (outcome) => {
  let finish!: (value: { id: string; platform: "website"; url: string }) => void;
  let fail!: (error: Error) => void;
  vi.mocked(postSocial).mockImplementationOnce(() => new Promise((resolve, reject) => { finish = resolve; fail = reject; }));
  render(<SettingsPage />); fireEvent.click(screen.getByRole("button", { name: /Edit settings/ }));
  act(() => usePhotographerSettingsStore.setState({ socials: [{ id: "local", platform: "website", url: "https://example.test" }] }));
  fireEvent.click(screen.getByRole("button", { name: /Save changes/ }));
  act(() => { useAuthStore.getState().logout(); resetUserScopedStores(); });
  await act(async () => {
    if (outcome === "success") finish({ id: "server-id", platform: "website", url: "https://example.test" });
    else fail(new Error("Late failure"));
  });
  expect(usePhotographerSettingsStore.getState().socials).toEqual([]);
  expect(mocks.toast).not.toHaveBeenCalled();
});

it("ignores an old verification withdrawal after account switching", async () => {
  usePhotographerSettingsStore.setState({ verificationStatus: "pending" });
  let finish!: (value: { status: "incomplete" }) => void;
  mocks.confirm.mockResolvedValue(true);
  vi.mocked(withdrawVerification).mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }));
  render(<SettingsPage />); fireEvent.click(screen.getByRole("button", { name: "Withdraw application" }));
  await waitFor(() => expect(withdrawVerification).toHaveBeenCalledTimes(1));
  act(() => {
    useAuthStore.getState().logout(); resetUserScopedStores();
    useAuthStore.getState().setUser({ ...photographer, id: "other" });
    usePhotographerSettingsStore.setState({ verificationStatus: "approved" });
  });
  await act(async () => { finish({ status: "incomplete" }); });
  expect(usePhotographerSettingsStore.getState().verificationStatus).toBe("approved");
  expect(mocks.toast).not.toHaveBeenCalled();
});
