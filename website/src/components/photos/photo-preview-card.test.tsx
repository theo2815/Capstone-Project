import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, onTestFinished, vi } from "vitest";
import { api, ApiError } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";
import type { PhotoMetadataDto } from "@/types/photo-metadata";
import { useState } from "react";
import { ToastContainer } from "@/components/ui/toast";
import { PhotoPreviewCard, type PhotoPreviewItem } from "./photo-preview-card";

describe("PhotoPreviewCard coupon offer", () => {
  it("credits a verified photographer by public name", async () => {
    const photo: PhotoPreviewItem = {
      id: "photo-credit",
      bib: null,
      time: "",
      tone: 0,
      price: 0,
      photographerName: "QuickPitik Photographer",
      photographerHandle: "quickpitikfree",
    };

    render(
      <PhotoPreviewCard
        photo={photo}
        eventName="NRW Fun Run 2026"
        index={1}
        total={1}
        inCart={false}
        onClose={vi.fn()}
        onToggleCart={vi.fn()}
        onBuyNow={vi.fn()}
      />,
    );

    expect(await screen.findAllByText("Photo by")).toHaveLength(2);
    expect(screen.getAllByRole("link", { name: /QuickPitik Photographer/ })).toHaveLength(2);
    expect(screen.queryByText("@quickpitikfree")).not.toBeInTheDocument();
  });

  it("shows only the coupon fields supplied by the photo API", async () => {
    const photo: PhotoPreviewItem = {
      id: "photo-1",
      bib: "101",
      time: "",
      tone: 0,
      price: 150,
      photographerName: "Aira",
      couponCode: "RACE20",
      couponPercentOff: 20,
      couponPrice: 127.5,
    };
    const props = {
      eventName: "Current Event",
      index: 1,
      total: 1,
      inCart: false,
      onClose: vi.fn(),
      onToggleCart: vi.fn(),
      onBuyNow: vi.fn(),
    };
    const { rerender } = render(<PhotoPreviewCard {...props} photo={photo} />);

    // Auto-apply (2026-09-05): the offer row says the discount is applied at
    // checkout; the code itself is no longer shown or copied.
    // The rail mounts twice (lg column + mobile strip), as before.
    expect(await screen.findAllByText(/20% off .* applied at checkout/)).toHaveLength(2);
    expect(screen.getByText(/−20% · list ₱150/)).toBeInTheDocument();
    expect(screen.queryByText(/RACE20/)).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Copy" })).not.toBeInTheDocument();

    rerender(
      <PhotoPreviewCard
        {...props}
        photo={{ ...photo, id: "photo-2", couponCode: null, couponPercentOff: null, couponPrice: null }}
      />,
    );

    await waitFor(() => expect(screen.queryAllByText(/applied at checkout/)).toHaveLength(0));
  });
});

const photo: PhotoPreviewItem = { id: "photo-a", bib: "101", time: "", tone: 0, price: 100 };
const metadata: PhotoMetadataDto = {
  photoId: "photo-a", status: "ready", originalFilename: "photo-a.jpg", fileSizeBytes: 1024,
  fields: { cameraMake: "Canon", cameraModel: "EOS R6", lens: null, aperture: null,
    exposureTime: null, iso: null, author: null, copyright: null, takenAtLocal: null,
    takenAtOffset: null, flash: null, exposureMode: null, whiteBalance: null },
};
const callbacks = () => ({ onClose: vi.fn(), onPrev: vi.fn(), onNext: vi.fn(), onDownload: vi.fn() });
function withQuery(card: React.ReactElement) {
  return <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>{card}</QueryClientProvider>;
}
afterEach(() => { vi.restoreAllMocks(); useAuthStore.getState().logout(); });

describe("PhotoPreviewCard metadata access", () => {
  it("bounds long purchase details below lg without changing callers without details", async () => {
    const base = { photo, mode: "owned" as const, eventName: "Cebu Run", index: 1, total: 1, onClose: vi.fn(), onDownload: vi.fn() };
    const view = render(withQuery(<PhotoPreviewCard {...base} details={<p>Long purchase details</p>} />));
    const rail = (await screen.findByText("Long purchase details")).closest("aside")!;
    expect(rail).toHaveClass("max-h-[50dvh]", "lg:max-h-none", "overflow-hidden");
    expect(screen.getByText("Long purchase details").parentElement).toHaveClass("flex-1", "min-h-0", "overflow-y-auto");
    expect(screen.getByRole("button", { name: "Download photo" }).closest("aside")).toBe(rail);
    view.rerender(withQuery(<PhotoPreviewCard {...base} />));
    expect(screen.getByRole("dialog").querySelector("aside")).not.toHaveClass("max-h-[50dvh]");
  });

  it("renders owner details and disables download while ownership resolves", async () => {
    const onDownload = vi.fn();
    render(withQuery(<PhotoPreviewCard photo={photo} mode="owned" eventName="Cebu Run" index={1} total={1}
      onClose={vi.fn()} onDownload={onDownload} downloadDisabled details={<p>Paid for this photo</p>} />));
    expect(await screen.findByText("Paid for this photo")).toBeVisible();
    expect(screen.getByRole("button", { name: "Download photo" })).toBeDisabled();
    expect(screen.queryByRole("button", { name: /buy now/i })).not.toBeInTheDocument();
  });
  it("shows no metadata action for sibling callers or modes without explicit access", async () => {
    const get = vi.spyOn(api, "get");
    const base = { photo, eventName: "Cebu Run", index: 1, total: 1, onClose: vi.fn() };
    const { rerender } = render(withQuery(<PhotoPreviewCard {...base} mode="owned" onDownload={vi.fn()} />));
    expect(await screen.findByRole("button", { name: "Download photo" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Show Metadata" })).toBeNull();
    rerender(withQuery(<PhotoPreviewCard {...base} mode="review" />));
    expect(screen.queryByRole("button", { name: "Show Metadata" })).toBeNull();
    rerender(withQuery(<PhotoPreviewCard {...base} inCart={false} onToggleCart={vi.fn()} onBuyNow={vi.fn()} />));
    expect(screen.queryByRole("button", { name: "Show Metadata" })).toBeNull();
    rerender(withQuery(<PhotoPreviewCard {...base} metadataAccess="none" inCart={false} onToggleCart={vi.fn()} onBuyNow={vi.fn()} />));
    expect(screen.queryByRole("button", { name: "Show Metadata" })).toBeNull();
    expect(get).not.toHaveBeenCalled();
  });

  it.each(["owner", "public"] as const)("fetches %s only on activation and keeps download primary", async (access) => {
    if (access === "owner") useAuthStore.getState().startSession({ id: "photographer", role: "PHOTOGRAPHER", email: "a@b.test", name: "A", createdAt: "2026-09-28" });
    const get = vi.spyOn(api, "get").mockResolvedValue(metadata);
    const user = userEvent.setup();
    const cb = callbacks();
    render(withQuery(<PhotoPreviewCard photo={photo} metadataAccess={access} mode="owned" eventName="Cebu Run" index={1} total={1} {...cb} />));
    const trigger = await screen.findByRole("button", { name: "Show Metadata" });
    expect(get).not.toHaveBeenCalled();
    await user.click(trigger);
    expect(await screen.findByText("Canon EOS R6")).toBeInTheDocument();
    expect(get).toHaveBeenCalledTimes(1);
    expect(get).toHaveBeenCalledWith(access === "owner" ? "/me/photographer/photos/photo-a/metadata" : "/public/photos/photo-a/metadata", expect.anything());
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog", { name: "Photo metadata" })).toBeNull();
    expect(screen.getByRole("dialog", { name: "Preview photo 101" })).toBeInTheDocument();
    expect(cb.onClose).not.toHaveBeenCalled();
    expect(trigger).toHaveFocus();
    await user.keyboard("{Escape}");
    expect(cb.onClose).toHaveBeenCalledTimes(1);
    await user.click(screen.getByRole("button", { name: "Download photo" }));
    expect(cb.onDownload).toHaveBeenCalledTimes(1);
  });

  it("closes metadata on photo navigation and discards a late prior-photo result", async () => {
    let resolveA!: (value: PhotoMetadataDto) => void;
    vi.spyOn(api, "get").mockImplementationOnce(() => new Promise((resolve) => { resolveA = resolve as typeof resolveA; })).mockResolvedValueOnce({ ...metadata, photoId: "photo-b", fields: { ...metadata.fields!, cameraModel: "EOS R8" } });
    const cb = callbacks();
    const user = userEvent.setup();
    const renderCard = (selected: PhotoPreviewItem) => withQuery(<PhotoPreviewCard photo={selected} metadataAccess="public" mode="owned" eventName="Cebu Run" index={1} total={2} {...cb} />);
    const view = render(renderCard(photo));
    await user.click(await screen.findByRole("button", { name: "Show Metadata" }));
    expect(screen.getByRole("status", { name: "Loading metadata" })).toBeInTheDocument();
    view.rerender(renderCard({ ...photo, id: "photo-b" }));
    expect(screen.queryByRole("dialog", { name: "Photo metadata" })).toBeNull();
    await act(async () => resolveA(metadata));
    expect(screen.queryByText("Canon EOS R6")).toBeNull();
    await user.click(screen.getByRole("button", { name: "Show Metadata" }));
    expect(await screen.findByText("Canon EOS R8")).toBeInTheDocument();
  });

  it("keeps browse cart actions and suspends parent arrows during metadata retry", async () => {
    const get = vi.spyOn(api, "get").mockRejectedValueOnce(new Error("offline")).mockResolvedValueOnce(metadata);
    const user = userEvent.setup();
    const onPrev = vi.fn(); const onNext = vi.fn(); const onClose = vi.fn();
    const onToggleCart = vi.fn(); const onBuyNow = vi.fn();
    render(withQuery(<PhotoPreviewCard photo={photo} metadataAccess="public" eventName="Cebu Run" index={1} total={2}
      inCart={false} onClose={onClose} onPrev={onPrev} onNext={onNext} onToggleCart={onToggleCart} onBuyNow={onBuyNow} />));
    expect(await screen.findByRole("button", { name: "Buy now" })).toBeInTheDocument();
    expect(get).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "Show Metadata" }));
    expect(await screen.findByText("Metadata could not be loaded.")).toBeInTheDocument();
    await user.keyboard("{ArrowLeft}{ArrowRight}");
    expect(onPrev).not.toHaveBeenCalled();
    expect(onNext).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByText("Canon EOS R6")).toBeInTheDocument();
    expect(get).toHaveBeenCalledTimes(2);
    await user.keyboard("{Escape}{ArrowRight}");
    expect(onClose).not.toHaveBeenCalled();
    expect(onNext).toHaveBeenCalledTimes(1);
    await user.click(screen.getByRole("button", { name: "+ Add to cart" }));
    await user.click(screen.getByRole("button", { name: "Buy now" }));
    expect(onToggleCart).toHaveBeenCalledTimes(1);
    expect(onBuyNow).toHaveBeenCalledTimes(1);
  });

  it("ignores a metadata response arriving after logout", async () => {
    let resolve!: (value: PhotoMetadataDto) => void;
    vi.spyOn(api, "get").mockImplementation(() => new Promise((r) => { resolve = r as typeof resolve; }));
    const user = userEvent.setup();
    render(withQuery(<PhotoPreviewCard photo={photo} metadataAccess="public" mode="owned" eventName="Cebu Run" index={1} total={1} {...callbacks()} />));
    await user.click(await screen.findByRole("button", { name: "Show Metadata" }));
    expect(screen.getByRole("status", { name: "Loading metadata" })).toBeInTheDocument();
    act(() => useAuthStore.getState().logout());
    await act(async () => resolve(metadata));
    expect(screen.queryByRole("dialog", { name: "Photo metadata" })).toBeNull();
    expect(screen.queryByText("Canon EOS R6")).toBeNull();
  });

  it("closes owner metadata when the viewer role changes under the same user ID", async () => {
    const viewer = { id: "viewer", role: "PHOTOGRAPHER" as const, email: "a@b.test", name: "A", createdAt: "2026-09-28" };
    useAuthStore.getState().setUser(viewer);
    vi.spyOn(api, "get").mockResolvedValue(metadata);
    const user = userEvent.setup();
    render(withQuery(<PhotoPreviewCard photo={photo} metadataAccess="owner" mode="owned" eventName="Cebu Run" index={1} total={1} {...callbacks()} />));
    await user.click(await screen.findByRole("button", { name: "Show Metadata" }));
    expect(await screen.findByText("Canon EOS R6")).toBeInTheDocument();
    act(() => useAuthStore.getState().setUser({ ...viewer, role: "RUNNER" }));
    expect(screen.queryByRole("dialog", { name: "Photo metadata" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Show Metadata" })).toBeNull();
  });

  it("drops metadata on session change and suppresses a denied public action until the photo changes", async () => {
    const get = vi.spyOn(api, "get").mockResolvedValueOnce(metadata).mockRejectedValueOnce(new ApiError([{ code: "NOT_FOUND", message: "Gone" }], 404));
    const user = userEvent.setup();
    const view = render(withQuery(<PhotoPreviewCard photo={photo} metadataAccess="public" mode="owned" eventName="Cebu Run" index={1} total={2} {...callbacks()} />));
    await user.click(await screen.findByRole("button", { name: "Show Metadata" }));
    expect(await screen.findByText("Canon EOS R6")).toBeInTheDocument();
    act(() => useAuthStore.getState().logout());
    expect(screen.queryByRole("dialog", { name: "Photo metadata" })).toBeNull();
    expect(screen.queryByText("Canon EOS R6")).toBeNull();
    await user.click(screen.getByRole("button", { name: "Show Metadata" }));
    expect(await screen.findByText("Metadata is no longer available")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Close metadata" }));
    expect(screen.queryByRole("button", { name: "Show Metadata" })).toBeNull();
    expect(screen.getAllByRole("button", { name: "Close preview" })[1]).toHaveFocus();
    view.rerender(withQuery(<PhotoPreviewCard photo={{ ...photo, id: "photo-b" }} metadataAccess="public" mode="owned" eventName="Cebu Run" index={2} total={2} {...callbacks()} />));
    await waitFor(() => expect(screen.getByRole("button", { name: "Show Metadata" })).toBeInTheDocument());
    expect(get).toHaveBeenCalledTimes(2);
  });

  it("starts a new image attempt after a successful same-URL refresh and keeps details", async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const props = { photo: { ...photo, imageUrl: "https://cdn.example.test/signed.jpg" }, mode: "owned" as const,
      eventName: "Cebu Run", index: 1, total: 1, onClose: vi.fn(), onDownload: vi.fn(),
      details: <p>Purchase facts stay visible</p>, onRefreshImage: vi.fn() };
    const card = (imageReloadKey: number) => <QueryClientProvider client={client}><PhotoPreviewCard {...props} imageReloadKey={imageReloadKey} /></QueryClientProvider>;
    const view = render(card(0));
    const failedImage = await screen.findByRole("img", { name: "Race photo of bib 101" });
    fireEvent.error(failedImage);
    expect(screen.getByText("Preview unavailable")).toBeVisible();

    view.rerender(card(1));

    const retriedImage = await screen.findByRole("img", { name: "Race photo of bib 101" });
    expect(retriedImage).not.toBe(failedImage);
    expect(retriedImage).toHaveAttribute("src", props.photo.imageUrl);
    expect(screen.getByText("Purchase facts stay visible")).toBeVisible();
  });

  it("bounds refresh clicks while pending and exposes a retryable failure", async () => {
    const onRefreshImage = vi.fn();
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const props = { photo: { ...photo, imageUrl: "https://cdn.example.test/signed.jpg" }, mode: "owned" as const,
      eventName: "Cebu Run", index: 1, total: 1, onClose: vi.fn(), onDownload: vi.fn(),
      details: <p>Purchase facts</p>, onRefreshImage, imageReloadKey: 0 };
    const card = (pending: boolean, error: string | null) => <QueryClientProvider client={client}><PhotoPreviewCard {...props} isRefreshingImage={pending} imageRefreshError={error} /></QueryClientProvider>;
    const view = render(card(true, null));
    fireEvent.error(await screen.findByRole("img", { name: "Race photo of bib 101" }));
    const pendingButton = screen.getByRole("button", { name: "Refreshing preview…" });
    expect(pendingButton).toBeDisabled();
    fireEvent.click(pendingButton);
    expect(onRefreshImage).not.toHaveBeenCalled();

    view.rerender(card(false, "Could not refresh preview. Try again."));
    expect(screen.getByRole("alert")).toHaveTextContent("Could not refresh preview. Try again.");
    fireEvent.click(screen.getByRole("button", { name: "Refresh preview" }));
    expect(onRefreshImage).toHaveBeenCalledOnce();
  });
});

describe("PhotoPreviewCard keyboard focus", () => {
  // jsdom has no layout; model the below-lg viewport, where `hidden lg:*` controls are display:none.
  const belowLg = () => Object.defineProperty(HTMLElement.prototype, "checkVisibility", {
    configurable: true, value(this: HTMLElement) { return !this.closest(".hidden"); },
  });
  afterEach(() => { delete (HTMLElement.prototype as Partial<HTMLElement>).checkVisibility; });
  const page = (open: boolean, cb: ReturnType<typeof callbacks>, withOpener = true) => withQuery(<>
    {withOpener && <button type="button">Open preview</button>}
    {open && <PhotoPreviewCard photo={photo} metadataAccess="public" mode="owned" eventName="Cebu Run" index={1} total={2} {...cb} />}
  </>);

  it("moves focus inside and wraps Tab and Shift+Tab across the visible controls only", async () => {
    belowLg();
    const user = userEvent.setup();
    const cb = callbacks();
    const view = render(page(false, cb));
    screen.getByRole("button", { name: "Open preview" }).focus();
    view.rerender(page(true, cb));
    const dialog = await screen.findByRole("dialog", { name: "Preview photo 101" });
    const [backdrop, mobileClose] = screen.getAllByRole("button", { name: "Close preview" });
    const metadataButton = screen.getByRole("button", { name: "Show Metadata" });
    expect(dialog).toHaveFocus();
    expect(backdrop).toHaveAttribute("tabindex", "-1");
    await user.tab({ shift: true });
    expect(metadataButton).toHaveFocus();
    await user.tab();
    expect(mobileClose).toHaveFocus();
    await user.tab({ shift: true });
    expect(metadataButton).toHaveFocus();
  });

  it("closes only itself on Escape, not a drawer or modal listening underneath", async () => {
    const underneath = vi.fn();
    document.addEventListener("keydown", underneath);
    onTestFinished(() => document.removeEventListener("keydown", underneath));
    const user = userEvent.setup();
    const cb = callbacks();
    render(page(true, cb));
    await screen.findByRole("dialog", { name: "Preview photo 101" });
    await user.keyboard("{Escape}");
    expect(cb.onClose).toHaveBeenCalledTimes(1);
    expect(underneath).not.toHaveBeenCalled();
  });

  it("does not land on Close after Next remounts the preview, so a repeated Enter cannot close it", async () => {
    belowLg();
    const user = userEvent.setup();
    const onClose = vi.fn();
    function Keyed() {
      const [current, setCurrent] = useState(photo);
      return withQuery(<PhotoPreviewCard key={current.id} photo={current} mode="owned" eventName="Cebu Run" index={1} total={2}
        onClose={onClose} onDownload={vi.fn()} onNext={() => setCurrent({ ...photo, id: "photo-b" })} />);
    }
    render(<Keyed />);
    await screen.findByRole("dialog", { name: "Preview photo 101" });
    screen.getAllByRole("button", { name: "Next photo" })[1].focus();
    await user.keyboard("{Enter}{Enter}");
    expect(onClose).not.toHaveBeenCalled();
  });

  it("keeps the page inert while open, then restores the trigger and prior inert state", async () => {
    const alreadyInert = document.body.appendChild(document.createElement("div"));
    alreadyInert.setAttribute("inert", "");
    // Providers mounts the toast region beside the page; preview actions raise toasts into it.
    render(<ToastContainer />);
    const toasts = document.body.appendChild(document.createElement("div"));
    toasts.setAttribute("role", "region");
    toasts.setAttribute("aria-label", "Notifications");
    expect(screen.getAllByRole("region", { name: "Notifications" })).toHaveLength(2);
    onTestFinished(() => { alreadyInert.remove(); toasts.remove(); });
    const cb = callbacks();
    const view = render(page(false, cb));
    const opener = screen.getByRole("button", { name: "Open preview" });
    opener.focus();
    view.rerender(page(true, cb));
    await screen.findByRole("dialog", { name: "Preview photo 101" });
    expect(opener.closest("[inert]")).not.toBeNull();
    expect(screen.getByRole("dialog", { name: "Preview photo 101" }).closest("[inert]")).toBeNull();
    expect(toasts).not.toHaveAttribute("inert");
    view.rerender(page(false, cb));
    expect(opener.closest("[inert]")).toBeNull();
    expect(alreadyInert).toHaveAttribute("inert");
    expect(opener).toHaveFocus();
  });

  it("lets metadata own Tab and Escape, then returns through the preview to the page", async () => {
    belowLg();
    vi.spyOn(api, "get").mockResolvedValue(metadata);
    const user = userEvent.setup();
    const cb = callbacks();
    const view = render(page(false, cb));
    const opener = screen.getByRole("button", { name: "Open preview" });
    opener.focus();
    view.rerender(page(true, cb));
    const metadataButton = await screen.findByRole("button", { name: "Show Metadata" });
    await user.click(metadataButton);
    await screen.findByText("Canon EOS R6");
    const metadataDialog = screen.getByRole("dialog", { name: "Photo metadata" });
    expect(metadataDialog.closest("[inert]")).toBeNull();
    await user.tab();
    expect(metadataDialog).toContainElement(document.activeElement as HTMLElement);
    await user.tab({ shift: true });
    expect(metadataDialog).toContainElement(document.activeElement as HTMLElement);
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog", { name: "Photo metadata" })).toBeNull();
    expect(metadataButton).toHaveFocus();
    expect(cb.onClose).not.toHaveBeenCalled();
    await user.keyboard("{Escape}");
    expect(cb.onClose).toHaveBeenCalledTimes(1);
    view.rerender(page(false, cb));
    expect(opener).toHaveFocus();
  });

  it("cleans up on unmount with metadata open and never focuses a removed trigger", async () => {
    vi.spyOn(api, "get").mockResolvedValue(metadata);
    const user = userEvent.setup();
    const cb = callbacks();
    const view = render(page(false, cb));
    screen.getByRole("button", { name: "Open preview" }).focus();
    view.rerender(page(true, cb));
    await user.click(await screen.findByRole("button", { name: "Show Metadata" }));
    await screen.findByText("Canon EOS R6");
    view.rerender(page(true, cb, false));
    act(() => useAuthStore.getState().logout());
    view.rerender(page(false, cb, false));
    expect(document.querySelector("[inert]")).toBeNull();
    expect(document.activeElement).toBe(document.body);
    await user.keyboard("{Escape}");
    expect(cb.onClose).not.toHaveBeenCalled();
  });
});

describe("PhotoPreviewCard byline target", () => {
  it("gives the photographer link a target at least 24px tall", async () => {
    render(
      <PhotoPreviewCard
        photo={{ id: "photo-target", bib: null, time: "", tone: 0, price: 0, photographerName: "Studio", photographerHandle: "studio" }}
        eventName="Run"
        index={1}
        total={1}
        inCart={false}
        onClose={vi.fn()}
        onToggleCart={vi.fn()}
        onBuyNow={vi.fn()}
      />,
    );
    // jsdom has no layout: 18px line + 3px padding each side = 24px.
    for (const link of await screen.findAllByRole("link", { name: /Studio/ })) expect(link).toHaveClass("py-[3px]");
  });
});
