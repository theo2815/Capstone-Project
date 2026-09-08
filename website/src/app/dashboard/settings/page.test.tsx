import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import SettingsPage from "./page";
import { DEFAULT_WATERMARK_PLACEMENT, usePhotographerSettingsStore } from "@/store/photographer-settings-store";
import { putHandle, putWatermarkSettings } from "@/lib/api-photographer-settings";

const mocks = vi.hoisted(() => ({ toast: vi.fn(), confirm: vi.fn(), regions: [] }));
vi.mock("@/hooks/use-toast", () => ({ useToast: () => ({ showToast: mocks.toast }) }));
vi.mock("@/hooks/use-confirmation", () => ({ useConfirmation: () => ({ confirm: mocks.confirm }) }));
vi.mock("@/hooks/use-auth", () => ({ useAuth: () => ({ user: null }) }));
vi.mock("@/hooks/use-regions", () => ({ useRegions: () => ({ regions: mocks.regions, isLoading: false }) }));
vi.mock("@/lib/photographer-verification-sync", () => ({ syncVerificationNow: vi.fn() }));
vi.mock("@/lib/api-photographer-settings", async (importActual) => ({
  ...await importActual<typeof import("@/lib/api-photographer-settings")>(),
  putHandle: vi.fn(),
  putWatermarkSettings: vi.fn(),
}));

beforeEach(() => {
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

afterEach(() => {
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
