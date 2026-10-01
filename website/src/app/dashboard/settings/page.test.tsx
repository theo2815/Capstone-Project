import { act, cleanup, fireEvent, render as rtlRender, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactElement } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { useAuthStore } from "@/store/auth-store";
import { AuthHydrator } from "@/components/layout/auth-hydrator";
import { api } from "@/lib/api";
import { resetUserScopedStores } from "@/lib/auth-reset";
import { postCover, postSocial, withdrawVerification } from "@/lib/api-photographer-settings";
import { fitToDataUrl, inspectImageDimensions } from "@/lib/image-utils";
import SettingsPage from "./page";
import { DEFAULT_WATERMARK_PLACEMENT, usePhotographerSettingsStore } from "@/store/photographer-settings-store";
import { useUserMediaStore } from "@/store/user-media-store";
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
  postCover: vi.fn(),
  putHandle: vi.fn(),
  putWatermarkSettings: vi.fn(),
  postPayoutAccount: vi.fn(), patchPayoutAccount: vi.fn(), postPayoutQr: vi.fn(),
}));
vi.mock("@/lib/image-utils", async (importActual) => ({
  ...await importActual<typeof import("@/lib/image-utils")>(),
  inspectImageDimensions: vi.fn(), fitToDataUrl: vi.fn(),
}));

const photographer = { id: "photographer", role: "PHOTOGRAPHER" as const, name: "Photo", email: "photo@example.test", createdAt: "2026-09-10" };

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => { resolve = res; });
  return { promise, resolve };
}

function render(ui: ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  client.setQueryData(["photographer", "settings", "metadata", photographer.id, photographer.role, useAuthStore.getState().sessionVersion], { metadataPublicByDefault: false });
  return rtlRender(ui, {
    wrapper: ({ children }) => (
      <QueryClientProvider client={client}>
        {children}
      </QueryClientProvider>
    ),
  });
}

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
    hydratedSessionVersion: useAuthStore.getState().sessionVersion,
  });
  vi.mocked(putWatermarkSettings).mockResolvedValue();
  vi.mocked(putHandle).mockRejectedValue(new Error("Handle already taken"));
});

const payout = { id: "account", method: "gcash" as const, accountName: "Test Recipient", accountNumber: "09171234567", isPrimary: true, qr: null };
const qr = { dataUrl: "data:image/png;base64,AA==", uploadedAt: "2026-09-09T00:00:00Z" };

it("keeps every Edit entry disabled until initial settings hydrate", async () => {
  usePhotographerSettingsStore.setState({
    verificationStatus: "incomplete", verificationMissing: ["profile picture"],
    hydratedSessionVersion: null,
  });
  const reads = {
    brand: deferred<unknown>(), socials: deferred<unknown>(), payouts: deferred<unknown>(),
  };
  vi.spyOn(api, "get").mockImplementation((path) =>
    path.endsWith("/brand") ? reads.brand.promise
      : path.endsWith("/socials") ? reads.socials.promise
        : reads.payouts.promise,
  );

  rtlRender(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><AuthHydrator /><SettingsPage /></QueryClientProvider>);

  expect(screen.getByRole("status")).toHaveTextContent("Loading settings…");
  const editButtons = await screen.findAllByRole("button", { name: /Edit settings/ });
  expect(editButtons.length).toBeGreaterThan(0);
  editButtons.forEach((button) => expect(button).toBeDisabled());
  fireEvent.click(editButtons[0]);
  expect(usePhotographerSettingsStore.getState().draftEditing).toBe(false);

  await act(async () => { reads.socials.resolve([]); reads.payouts.resolve([]); });
  expect(editButtons[0]).toBeDisabled();
  await act(async () => { reads.brand.resolve({
    brandName: "Canonical", brandColor: "indigo", bio: "Existing biography", handle: "old-handle",
    regionCode: "", provinceCode: "", avatarUrl: null, coverUrl: "", watermarkUrl: "", watermarkSettings: null,
  }); });
  await waitFor(() => editButtons.forEach((button) => expect(button).not.toBeDisabled()));
});

it("preserves untouched canonical brand fields after deferred hydration", async () => {
  usePhotographerSettingsStore.setState({ hydratedSessionVersion: null });
  const reads = {
    brand: deferred<unknown>(), socials: deferred<unknown>(), payouts: deferred<unknown>(),
  };
  vi.spyOn(api, "get").mockImplementation((path) =>
    path.endsWith("/brand") ? reads.brand.promise
      : path.endsWith("/socials") ? reads.socials.promise
        : reads.payouts.promise,
  );
  const save = vi.spyOn(api, "put").mockResolvedValue(undefined);
  rtlRender(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><AuthHydrator /><SettingsPage /></QueryClientProvider>);

  await act(async () => {
    reads.brand.resolve({
      brandName: "Canonical", brandColor: "indigo", bio: "Existing biography", handle: "old-handle",
      regionCode: "", provinceCode: "", avatarUrl: null, coverUrl: "", watermarkUrl: "", watermarkSettings: null,
    });
    reads.socials.resolve([]);
    reads.payouts.resolve([]);
  });
  const edit = await screen.findByRole("button", { name: /Edit settings/ });
  await waitFor(() => expect(edit).not.toBeDisabled());
  fireEvent.click(edit);
  fireEvent.change(screen.getByLabelText("Display name"), { target: { value: "Changed name" } });
  fireEvent.click(screen.getByRole("button", { name: /Save changes/ }));

  await waitFor(() => expect(save).toHaveBeenCalledWith("/me/photographer/brand", {
    brandName: "Changed name", brandColor: "indigo", bio: "Existing biography",
  }));
});

it.each(["brand", "socials", "payouts"] as const)("keeps editing blocked and retries a failed initial %s read", async (failedRead) => {
  usePhotographerSettingsStore.setState({ hydratedSessionVersion: null });
  let recovered = false;
  const get = vi.spyOn(api, "get").mockImplementation((path) => {
    if (path.endsWith("/metadata-settings")) return Promise.resolve({ metadataPublicByDefault: false });
    const key = path.endsWith("/brand") ? "brand" : path.endsWith("/socials") ? "socials" : "payouts";
    if (key === failedRead && !recovered) return Promise.reject(new Error("Offline"));
    return Promise.resolve(key === "brand" ? {
      brandName: "Canonical", brandColor: "indigo", bio: "Existing biography", handle: "old-handle",
      regionCode: "", provinceCode: "", avatarUrl: null, coverUrl: "", watermarkUrl: "", watermarkSettings: null,
    } : []);
  });
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
  rtlRender(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><AuthHydrator /><SettingsPage /></QueryClientProvider>);

  // Advance the query's two retry delays without spending wall time on them.
  await act(async () => { await vi.advanceTimersByTimeAsync(4_001); });
  vi.useRealTimers();
  expect(screen.getByRole("alert")).toHaveTextContent("Couldn't load settings");
  screen.getAllByRole("button", { name: /Edit settings/ }).forEach((button) => expect(button).toBeDisabled());
  recovered = true;
  fireEvent.click(screen.getByRole("button", { name: "Retry" }));

  await waitFor(() => screen.getAllByRole("button", { name: /Edit settings/ }).forEach((button) => expect(button).not.toBeDisabled()));
  expect(usePhotographerSettingsStore.getState()).toMatchObject({
    brandName: "Canonical", bio: "Existing biography",
    hydratedSessionVersion: useAuthStore.getState().sessionVersion,
  });
  expect(get.mock.calls.some(([path]) => path.endsWith("/metadata-settings"))).toBe(true);
});

it("waits for the metadata default before allowing Edit and shows the saved value", async () => {
  const metadata = deferred<{ metadataPublicByDefault: boolean }>();
  vi.spyOn(api, "get").mockImplementation((path) => path.endsWith("/metadata-settings")
    ? metadata.promise : Promise.resolve([]));
  rtlRender(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><SettingsPage /></QueryClientProvider>);

  const toggle = screen.getByRole("checkbox", { name: "Show metadata publicly for new events" });
  expect(toggle).not.toBeChecked();
  expect(toggle).toBeDisabled();
  expect(screen.getByText("Applies to events you create or start covering next. Existing events keep their settings.")).toBeInTheDocument();
  expect(screen.getByText("This controls the metadata popup. Original downloads keep their embedded metadata.")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /Edit settings/ })).toBeDisabled();
  await act(async () => metadata.resolve({ metadataPublicByDefault: true }));
  await waitFor(() => expect(screen.getByRole("button", { name: /Edit settings/ })).not.toBeDisabled());
  expect(toggle).toBeChecked();
  expect(toggle).toBeDisabled();
});

it("retries a failed metadata read before making the default editable", async () => {
  let recovered = false;
  vi.spyOn(api, "get").mockImplementation((path) => path.endsWith("/metadata-settings")
    ? recovered ? Promise.resolve({ metadataPublicByDefault: true }) : Promise.reject(new Error("Offline"))
    : Promise.resolve([]));
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
  rtlRender(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}><SettingsPage /></QueryClientProvider>);
  await act(async () => { await vi.advanceTimersByTimeAsync(4_001); });
  vi.useRealTimers();
  expect(screen.getByRole("alert")).toHaveTextContent("Couldn't load settings");
  expect(screen.getByRole("button", { name: /Edit settings/ })).toBeDisabled();
  expect(screen.getByRole("checkbox", { name: "Show metadata publicly for new events" })).not.toBeChecked();
  recovered = true;
  fireEvent.click(screen.getByRole("button", { name: "Retry" }));
  await waitFor(() => expect(screen.getByRole("button", { name: /Edit settings/ })).not.toBeDisabled());
  expect(screen.getByRole("checkbox", { name: "Show metadata publicly for new events" })).toBeChecked();
});

it("cancels a draft and saves the default without touching event visibility", async () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const sessionKey = [photographer.id, photographer.role, useAuthStore.getState().sessionVersion];
  const metadataKey = ["photographer", "settings", "metadata", ...sessionKey];
  const visibilityKey = ["photographer", "events", "event-a", "metadata-visibility", ...sessionKey];
  client.setQueryData(metadataKey, { metadataPublicByDefault: false });
  client.setQueryData(visibilityKey, { metadataPublic: false });
  let saved = false;
  vi.spyOn(api, "get").mockImplementation((path) => Promise.resolve(path.endsWith("/metadata-settings")
    ? { metadataPublicByDefault: saved } : []));
  const put = vi.spyOn(api, "put").mockImplementation(async () => {
    saved = true;
    return { metadataPublicByDefault: true };
  });
  const invalidate = vi.spyOn(client, "invalidateQueries");
  rtlRender(<QueryClientProvider client={client}><SettingsPage /></QueryClientProvider>);
  const toggle = screen.getByRole("checkbox", { name: "Show metadata publicly for new events" });
  fireEvent.click(screen.getByRole("button", { name: /Edit settings/ }));
  fireEvent.click(toggle);
  expect(toggle).toBeChecked();
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(toggle).not.toBeChecked();
  fireEvent.click(screen.getByRole("button", { name: /Edit settings/ }));
  fireEvent.click(toggle);
  fireEvent.click(screen.getByRole("button", { name: /Save changes/ }));
  await waitFor(() => expect(put).toHaveBeenCalledWith("/me/photographer/metadata-settings", { metadataPublicByDefault: true }));
  await waitFor(() => expect(screen.getByRole("button", { name: /Edit settings/ })).toBeInTheDocument());
  expect(toggle).toBeChecked();
  expect(client.getQueryData(visibilityKey)).toEqual({ metadataPublic: false });
  expect(invalidate.mock.calls.every(([options]) => !String(options?.queryKey?.join("/")).includes("events"))).toBe(true);
});

it("retains the metadata draft after a failed save and retries it", async () => {
  const put = vi.spyOn(api, "put")
    .mockRejectedValueOnce(new Error("Offline"))
    .mockResolvedValueOnce({ metadataPublicByDefault: true });
  render(<SettingsPage />);
  const toggle = screen.getByRole("checkbox", { name: "Show metadata publicly for new events" });
  fireEvent.click(screen.getByRole("button", { name: /Edit settings/ }));
  fireEvent.click(toggle);
  fireEvent.click(screen.getByRole("button", { name: /Save changes/ }));
  await waitFor(() => expect(screen.getByRole("button", { name: /Save changes/ })).not.toBeDisabled());
  expect(toggle).toBeChecked();
  expect(mocks.toast).toHaveBeenCalledWith(expect.objectContaining({ kind: "error" }));
  fireEvent.click(screen.getByRole("button", { name: /Save changes/ }));
  await waitFor(() => expect(screen.getByRole("button", { name: /Edit settings/ })).toBeInTheDocument());
  expect(put).toHaveBeenCalledTimes(2);
  expect(toggle).toBeChecked();
});

it("does not save an untouched metadata default after a background refresh", async () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  client.setQueryData(["photographer", "settings", "metadata", photographer.id, photographer.role, useAuthStore.getState().sessionVersion], { metadataPublicByDefault: false });
  let refreshed = false;
  vi.spyOn(api, "get").mockImplementation((path) => Promise.resolve(path.endsWith("/metadata-settings")
    ? { metadataPublicByDefault: refreshed } : []));
  const put = vi.spyOn(api, "put").mockResolvedValue(undefined);
  rtlRender(<QueryClientProvider client={client}><SettingsPage /></QueryClientProvider>);
  fireEvent.click(screen.getByRole("button", { name: /Edit settings/ }));
  refreshed = true;
  await act(async () => { await client.invalidateQueries({ queryKey: ["photographer", "settings", "metadata"] }); });
  await waitFor(() => expect(client.getQueryData(["photographer", "settings", "metadata", photographer.id, photographer.role, useAuthStore.getState().sessionVersion])).toEqual({ metadataPublicByDefault: true }));
  expect(screen.getByRole("checkbox", { name: "Show metadata publicly for new events" })).not.toBeChecked();
  fireEvent.change(screen.getByLabelText("Display name"), { target: { value: "Changed name" } });
  fireEvent.click(screen.getByRole("button", { name: /Save changes/ }));
  await waitFor(() => expect(put).toHaveBeenCalledWith("/me/photographer/brand", expect.anything()));
  expect(put.mock.calls.some(([path]) => path.endsWith("/metadata-settings"))).toBe(false);
});

it("ignores a metadata save that finishes after logout and same-account login", async () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const firstVersion = useAuthStore.getState().sessionVersion;
  client.setQueryData(["photographer", "settings", "metadata", photographer.id, photographer.role, firstVersion], { metadataPublicByDefault: false });
  vi.spyOn(api, "get").mockImplementation((path) => Promise.resolve(path.endsWith("/metadata-settings")
    ? { metadataPublicByDefault: false } : []));
  const pending = deferred<{ metadataPublicByDefault: boolean }>();
  vi.spyOn(api, "put").mockReturnValue(pending.promise);
  rtlRender(<QueryClientProvider client={client}><SettingsPage /></QueryClientProvider>);
  fireEvent.click(screen.getByRole("button", { name: /Edit settings/ }));
  fireEvent.click(screen.getByRole("checkbox", { name: "Show metadata publicly for new events" }));
  fireEvent.click(screen.getByRole("button", { name: /Save changes/ }));
  await waitFor(() => expect(api.put).toHaveBeenCalledTimes(1));
  act(() => {
    useAuthStore.getState().logout(); resetUserScopedStores();
    useAuthStore.getState().setUser(photographer);
  });
  await waitFor(() => expect(screen.getByRole("checkbox", { name: "Show metadata publicly for new events" })).not.toBeChecked());
  await act(async () => pending.resolve({ metadataPublicByDefault: true }));
  const currentKey = ["photographer", "settings", "metadata", photographer.id, photographer.role, useAuthStore.getState().sessionVersion];
  expect(client.getQueryData(currentKey)).toEqual({ metadataPublicByDefault: false });
  expect(mocks.toast).not.toHaveBeenCalled();
});

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

it("keeps a second edit when the first save's settings refetch finishes late", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  let finishRefetch!: (value: unknown) => void;
  let brandReads = 0;
  const brand = (brandName: string) => ({
    brandName, brandColor: "none", bio: "", handle: "old-handle",
    regionCode: "", provinceCode: "", avatarUrl: null, coverUrl: "", watermarkUrl: "", watermarkSettings: null,
  });
  vi.spyOn(api, "get").mockImplementation((path) => {
    if (!path.endsWith("/brand")) return Promise.resolve([]);
    brandReads += 1;
    return brandReads === 1
      ? Promise.resolve(brand("Original"))
      : new Promise((resolve) => { finishRefetch = resolve; });
  });
  vi.spyOn(api, "put").mockResolvedValue(undefined);
  rtlRender(
    <QueryClientProvider client={queryClient}>
      <AuthHydrator />
      <SettingsPage />
    </QueryClientProvider>,
  );
  await waitFor(() => expect(screen.getByLabelText("Display name")).toHaveValue("Original"));
  fireEvent.click(screen.getByRole("button", { name: /Edit settings/ }));
  fireEvent.change(screen.getByLabelText("Display name"), { target: { value: "Saved A" } });
  fireEvent.click(screen.getByRole("button", { name: /Save changes/ }));
  await waitFor(() => expect(brandReads).toBe(2));
  fireEvent.click(screen.getByRole("button", { name: /Edit settings/ }));
  fireEvent.change(screen.getByLabelText("Display name"), { target: { value: "Unsaved B" } });
  await act(async () => { finishRefetch(brand("Saved A")); });
  await waitFor(() => expect(queryClient.getQueriesData({ queryKey: ["photographer", "settings", "brand"] })[0]?.[1]).toMatchObject({ brandName: "Saved A" }));
  expect(screen.getByLabelText("Display name")).toHaveValue("Unsaved B");
  expect(screen.getByLabelText("Display name")).not.toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  await waitFor(() => expect(screen.getByLabelText("Display name")).toHaveValue("Saved A"));
});

it("keeps a confirmed partial save when Cancel follows a stale refetch during editing", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  let reads = 0;
  vi.spyOn(api, "get").mockImplementation((path) => Promise.resolve(path.endsWith("/brand")
    ? { brandName: ++reads === 1 ? "Original" : "Saved A", brandColor: "none", bio: "", handle: "old-handle", regionCode: "", provinceCode: "", avatarUrl: "", coverUrl: "", watermarkUrl: "", watermarkSettings: null }
    : []));
  vi.spyOn(api, "put").mockResolvedValue(undefined);
  rtlRender(<QueryClientProvider client={queryClient}><AuthHydrator /><SettingsPage /></QueryClientProvider>);
  await waitFor(() => expect(screen.getByLabelText("Display name")).toHaveValue("Original"));
  fireEvent.click(screen.getByRole("button", { name: /Edit settings/ }));
  fireEvent.change(screen.getByLabelText("Display name"), { target: { value: "Confirmed B" } });
  await act(async () => { await queryClient.invalidateQueries({ queryKey: ["photographer", "settings"] }); });
  expect(screen.getByLabelText("Display name")).toHaveValue("Confirmed B");
  act(() => usePhotographerSettingsStore.setState({ handle: "taken-handle" }));
  fireEvent.click(screen.getByRole("button", { name: /Save changes/ }));
  await waitFor(() => expect(screen.getByRole("button", { name: "Cancel" })).not.toBeDisabled());
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  await waitFor(() => expect(screen.getByLabelText("Display name")).toHaveValue("Confirmed B"));
});

it("does not let a detached save release a new page's draft", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  let finishSave!: () => void;
  let reads = 0;
  vi.spyOn(api, "get").mockImplementation((path) => Promise.resolve(path.endsWith("/brand")
    ? { brandName: ++reads === 1 ? "Original" : "Saved A", brandColor: "none", bio: "", handle: "old-handle", regionCode: "", provinceCode: "", avatarUrl: "", coverUrl: "", watermarkUrl: "", watermarkSettings: null }
    : []));
  vi.spyOn(api, "put").mockImplementation(() => new Promise((resolve) => { finishSave = () => resolve(undefined); }));
  const page = (key: string) => <QueryClientProvider client={queryClient}><AuthHydrator /><SettingsPage key={key} /></QueryClientProvider>;
  const view = rtlRender(page("first"));
  await waitFor(() => expect(screen.getByLabelText("Display name")).toHaveValue("Original"));
  fireEvent.click(screen.getByRole("button", { name: /Edit settings/ }));
  fireEvent.change(screen.getByLabelText("Display name"), { target: { value: "Saved A" } });
  fireEvent.click(screen.getByRole("button", { name: /Save changes/ }));
  await waitFor(() => expect(api.put).toHaveBeenCalledTimes(1));
  view.rerender(page("second"));
  fireEvent.click(screen.getByRole("button", { name: /Edit settings/ }));
  fireEvent.change(screen.getByLabelText("Display name"), { target: { value: "Unsaved B" } });
  await act(async () => { finishSave(); });
  await act(async () => { await queryClient.invalidateQueries({ queryKey: ["photographer", "settings"] }); });
  expect(screen.getByLabelText("Display name")).toHaveValue("Unsaved B");
});

it("keeps text and media drafts through a settings refresh, then uses refreshed data on Cancel and next Edit", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const first = { brandName: "Old", brandColor: "none", bio: "", handle: "old-handle", regionCode: "", provinceCode: "", avatarUrl: "/avatar-old", coverUrl: "/cover-old", watermarkUrl: "/mark-old", watermarkSettings: null };
  const refreshed = { ...first, brandName: "Server new", avatarUrl: "/avatar-new", coverUrl: "/cover-new", watermarkUrl: "/mark-new" };
  const oldSocial = { id: "old-social", platform: "website" as const, url: "https://old.test" };
  const newSocial = { ...oldSocial, url: "https://new.test" };
  const oldPayout = { ...payout, qr };
  const newPayout = { ...oldPayout, accountName: "Server new" };
  let refresh = false;
  vi.spyOn(api, "get").mockImplementation((path) => Promise.resolve(
    path.endsWith("/brand") ? (refresh ? refreshed : first)
      : path.endsWith("/socials") ? [refresh ? newSocial : oldSocial]
        : [refresh ? newPayout : oldPayout],
  ));
  rtlRender(
    <QueryClientProvider client={queryClient}>
      <AuthHydrator />
      <SettingsPage />
    </QueryClientProvider>,
  );
  await waitFor(() => expect(screen.getByLabelText("Display name")).toHaveValue("Old"));
  fireEvent.click(screen.getByRole("button", { name: /Edit settings/ }));
  act(() => {
    usePhotographerSettingsStore.setState({
      brandName: "Unsaved", cover: { dataUrl: "data:cover", uploadedAt: "" },
      watermark: { dataUrl: "data:mark", uploadedAt: "" },
      socials: [{ ...oldSocial, url: "https://draft.test" }],
      payouts: [{ ...oldPayout, accountName: "Draft" }],
    });
    useUserMediaStore.getState().setAvatar({ dataUrl: "data:avatar", uploadedAt: "" });
  });
  refresh = true;
  await act(async () => { await queryClient.invalidateQueries({ queryKey: ["photographer", "settings"] }); });
  expect(screen.getByLabelText("Display name")).toHaveValue("Unsaved");
  expect(usePhotographerSettingsStore.getState()).toMatchObject({
    cover: { dataUrl: "data:cover" }, watermark: { dataUrl: "data:mark" },
    socials: [{ url: "https://draft.test" }], payouts: [{ accountName: "Draft" }],
  });
  expect(useUserMediaStore.getState().avatar?.dataUrl).toBe("data:avatar");
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  await waitFor(() => expect(screen.getByLabelText("Display name")).toHaveValue("Server new"));
  expect(usePhotographerSettingsStore.getState()).toMatchObject({
    cover: { dataUrl: "/cover-new" }, watermark: { dataUrl: "/mark-new" },
    socials: [{ url: "https://new.test" }], payouts: [{ accountName: "Server new" }],
  });
  expect(useUserMediaStore.getState().avatar?.dataUrl).toBe("/avatar-new");
  fireEvent.click(screen.getByRole("button", { name: /Edit settings/ }));
  fireEvent.change(screen.getByLabelText("Display name"), { target: { value: "Another draft" } });
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(screen.getByLabelText("Display name")).toHaveValue("Server new");
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
  vi.useRealTimers();
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

it("keeps a confirmed brand change when a sibling save fails and the edit is cancelled", async () => {
  usePhotographerSettingsStore.setState({ brandName: "Original" });
  const saveBrand = vi.spyOn(api, "put").mockResolvedValue(undefined);
  render(<SettingsPage />);
  fireEvent.click(screen.getByRole("button", { name: /Edit settings/ }));
  fireEvent.change(screen.getByLabelText("Display name"), { target: { value: "Saved A" } });
  act(() => usePhotographerSettingsStore.setState({ handle: "taken-handle" }));
  fireEvent.click(screen.getByRole("button", { name: /Save changes/ }));
  await waitFor(() => expect(screen.getByRole("button", { name: "Cancel" })).not.toBeDisabled());
  expect(saveBrand).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(screen.getByLabelText("Display name")).toHaveValue("Saved A");
  expect(usePhotographerSettingsStore.getState().handle).toBe("old-handle");
});

it("does not upload a confirmed cover twice when another save stage fails", async () => {
  vi.mocked(inspectImageDimensions).mockResolvedValue({ width: 1600, height: 500, aspectRatio: 3.2 });
  vi.mocked(fitToDataUrl).mockResolvedValue("data:image/png;base64,AA==");
  vi.mocked(postCover).mockResolvedValue({ url: "/saved-cover", uploadedAt: "2026-09-20T00:00:00Z" });
  render(<SettingsPage />);
  fireEvent.click(screen.getByRole("button", { name: /Edit settings/ }));
  fireEvent.change(screen.getByLabelText(/Upload cover/), { target: { files: [new File(["cover"], "cover.png", { type: "image/png" })] } });
  await waitFor(() => expect(usePhotographerSettingsStore.getState().cover?.dataUrl).toBe("data:image/png;base64,AA=="));
  act(() => usePhotographerSettingsStore.setState({ handle: "taken-handle" }));
  fireEvent.click(screen.getByRole("button", { name: /Save changes/ }));
  await waitFor(() => expect(screen.getByRole("button", { name: "Cancel" })).not.toBeDisabled());
  fireEvent.click(screen.getByRole("button", { name: /Save changes/ }));
  await waitFor(() => expect(putHandle).toHaveBeenCalledTimes(2));
  await waitFor(() => expect(screen.getByRole("button", { name: "Cancel" })).not.toBeDisabled());
  expect(postCover).toHaveBeenCalledTimes(1);
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(usePhotographerSettingsStore.getState().cover?.dataUrl).toBe("/saved-cover");
});

it("keeps a replacement cover selected while the previous upload is saving", async () => {
  vi.mocked(inspectImageDimensions).mockResolvedValue({ width: 1600, height: 500, aspectRatio: 3.2 });
  vi.mocked(fitToDataUrl).mockImplementation((file) => Promise.resolve(`data:${file.name}`));
  let finishCover!: () => void;
  vi.mocked(postCover)
    .mockImplementationOnce(() => new Promise((resolve) => { finishCover = () => resolve({ url: "/cover-a", uploadedAt: "2026-09-20T00:00:00Z" }); }))
    .mockResolvedValueOnce({ url: "/cover-b", uploadedAt: "2026-09-20T00:00:01Z" });
  render(<SettingsPage />);
  fireEvent.click(screen.getByRole("button", { name: /Edit settings/ }));
  fireEvent.change(screen.getByLabelText(/Upload cover/), { target: { files: [new File(["a"], "a.png", { type: "image/png" })] } });
  await waitFor(() => expect(usePhotographerSettingsStore.getState().cover?.dataUrl).toBe("data:a.png"));
  act(() => usePhotographerSettingsStore.setState({ handle: "taken-handle" }));
  fireEvent.click(screen.getByRole("button", { name: /Save changes/ }));
  await waitFor(() => expect(postCover).toHaveBeenCalledTimes(1));
  fireEvent.change(screen.getByLabelText(/Replace cover/), { target: { files: [new File(["b"], "b.png", { type: "image/png" })] } });
  await waitFor(() => expect(usePhotographerSettingsStore.getState().cover?.dataUrl).toBe("data:b.png"));
  await act(async () => { finishCover(); });
  expect(usePhotographerSettingsStore.getState().cover?.dataUrl).toBe("data:b.png");
  fireEvent.click(screen.getByRole("button", { name: /Save changes/ }));
  await waitFor(() => expect(postCover).toHaveBeenCalledTimes(2));
  await waitFor(() => expect(screen.getByRole("button", { name: "Cancel" })).not.toBeDisabled());
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
  expect(usePhotographerSettingsStore.getState().cover?.dataUrl).toBe("/cover-b");
});

it("keeps editing to save a newer cover even when the earlier batch succeeds", async () => {
  vi.mocked(inspectImageDimensions).mockResolvedValue({ width: 1600, height: 500, aspectRatio: 3.2 });
  vi.mocked(fitToDataUrl).mockImplementation((file) => Promise.resolve(`data:${file.name}`));
  let finishCover!: () => void;
  vi.mocked(postCover)
    .mockImplementationOnce(() => new Promise((resolve) => { finishCover = () => resolve({ url: "/cover-a", uploadedAt: "2026-09-20T00:00:00Z" }); }))
    .mockResolvedValueOnce({ url: "/cover-b", uploadedAt: "2026-09-20T00:00:01Z" });
  render(<SettingsPage />);
  fireEvent.click(screen.getByRole("button", { name: /Edit settings/ }));
  fireEvent.change(screen.getByLabelText(/Upload cover/), { target: { files: [new File(["a"], "a.png", { type: "image/png" })] } });
  await waitFor(() => expect(usePhotographerSettingsStore.getState().cover?.dataUrl).toBe("data:a.png"));
  fireEvent.click(screen.getByRole("button", { name: /Save changes/ }));
  await waitFor(() => expect(postCover).toHaveBeenCalledTimes(1));
  fireEvent.change(screen.getByLabelText(/Replace cover/), { target: { files: [new File(["b"], "b.png", { type: "image/png" })] } });
  await waitFor(() => expect(usePhotographerSettingsStore.getState().cover?.dataUrl).toBe("data:b.png"));
  await act(async () => { finishCover(); });
  expect(screen.getByRole("button", { name: /Save changes/ })).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: /Save changes/ }));
  await waitFor(() => expect(postCover).toHaveBeenCalledTimes(2));
  await waitFor(() => expect(screen.getByRole("button", { name: /Edit settings/ })).toBeInTheDocument());
  expect(usePhotographerSettingsStore.getState().cover?.dataUrl).toBe("/cover-b");
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
