import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { ToastContainer } from "@/components/ui/toast";
import { api, ApiError } from "@/lib/api";
import { postCover } from "@/lib/api-photographer-settings";
import { fitToDataUrl, inspectImageDimensions } from "@/lib/image-utils";
import { useAuthStore } from "@/store/auth-store";
import { usePhotographerSettingsStore } from "@/store/photographer-settings-store";
import { useToastStore } from "@/store/toast-store";
import SettingsPage from "./page";

vi.mock("@/hooks/use-auth", () => ({ useAuth: () => ({ user: null }) }));
vi.mock("@/hooks/use-regions", () => ({
  useRegions: () => ({ regions: [], isLoading: false, error: null, retry: vi.fn() }),
}));
vi.mock("@/lib/photographer-verification-sync", () => ({ syncVerificationNow: vi.fn() }));
vi.mock("@/lib/api-photographer-settings", async (importActual) => ({
  ...await importActual<typeof import("@/lib/api-photographer-settings")>(),
  postCover: vi.fn(),
}));
vi.mock("@/lib/image-utils", async (importActual) => ({
  ...await importActual<typeof import("@/lib/image-utils")>(),
  inspectImageDimensions: vi.fn(), fitToDataUrl: vi.fn(),
}));

beforeEach(() => {
  useAuthStore.getState().setUser({
    id: "photographer", role: "PHOTOGRAPHER", name: "Photo",
    email: "photo@example.test", createdAt: "2026-09-10",
  });
  usePhotographerSettingsStore.getState().reset();
  usePhotographerSettingsStore.setState({
    brandName: "Original", verificationStatus: "approved",
    hydratedSessionVersion: useAuthStore.getState().sessionVersion,
  });
  useToastStore.getState().clear();
  vi.spyOn(console, "error").mockImplementation(() => {});
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
  vi.spyOn(HTMLElement.prototype, "clientWidth", "get").mockReturnValue(375);
  vi.spyOn(HTMLElement.prototype, "clientHeight", "get").mockReturnValue(250);
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  useToastStore.getState().clear();
});

function openSettings() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  client.setQueryData([
    "photographer", "settings", "metadata", "photographer", "PHOTOGRAPHER",
    useAuthStore.getState().sessionVersion,
  ], { metadataPublicByDefault: false });
  render(
    <QueryClientProvider client={client}>
      <SettingsPage />
      <ToastContainer />
    </QueryClientProvider>,
  );
  fireEvent.click(screen.getByRole("button", { name: /Edit settings/ }));
}

it("a refused brand name shows the server message and stays in edit mode", async () => {
  vi.spyOn(api, "put").mockRejectedValue(new ApiError([{
    code: "TEXT_NOT_ALLOWED", field: "brandName",
    message: "Your brand name has a word we don't allow. Please change it.",
  }], 400));
  openSettings();
  fireEvent.change(screen.getByLabelText("Display name"), { target: { value: "Refused name" } });
  fireEvent.click(screen.getByRole("button", { name: /Save changes/ }));

  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Couldn't save Brand. Your brand name has a word we don't allow. Please change it.",
  );
  expect(screen.getByRole("button", { name: /Save changes/ })).toBeEnabled();
  expect(screen.getByLabelText("Display name")).toHaveValue("Refused name");
  expect(screen.getByLabelText("Display name")).toBeEnabled();
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(screen.getByLabelText("Display name")).toHaveValue("Original");
});

it("a refused cover shows the server message; other changes saved", async () => {
  const saveBrand = vi.spyOn(api, "put").mockResolvedValue(undefined);
  vi.mocked(inspectImageDimensions).mockResolvedValue({ width: 1600, height: 500, aspectRatio: 3.2 });
  vi.mocked(fitToDataUrl).mockResolvedValue("data:image/png;base64,AA==");
  vi.mocked(postCover).mockRejectedValue(new ApiError([{
    code: "IMAGE_NOT_ALLOWED", field: "file",
    message: "This image can't be used. Please choose another.",
  }], 400));
  openSettings();
  fireEvent.change(screen.getByLabelText("Display name"), { target: { value: "Saved name" } });
  fireEvent.change(screen.getByLabelText(/Upload cover/), {
    target: { files: [new File(["cover"], "cover.png", { type: "image/png" })] },
  });
  await waitFor(() => expect(usePhotographerSettingsStore.getState().cover?.dataUrl).toBe("data:image/png;base64,AA=="));
  fireEvent.click(screen.getByRole("button", { name: /Save changes/ }));

  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Couldn't save Cover. This image can't be used. Please choose another.",
  );
  expect(screen.getByRole("button", { name: /Save changes/ })).toBeEnabled();
  expect(saveBrand).toHaveBeenCalledWith("/me/photographer/brand", expect.objectContaining({ brandName: "Saved name" }));
  fireEvent.click(screen.getByRole("button", { name: /Save changes/ }));
  await waitFor(() => expect(postCover).toHaveBeenCalledTimes(2));
  await waitFor(() => expect(screen.getByRole("button", { name: "Cancel" })).toBeEnabled());
  expect(saveBrand).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(screen.getByLabelText("Display name")).toHaveValue("Saved name");
  expect(usePhotographerSettingsStore.getState().cover).toBeNull();
});
