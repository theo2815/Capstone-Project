import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import { EventCockpit } from "@/app/events/[slug]/event-cockpit";
import HandleEventPage from "@/app/[handle]/events/[slug]/page";

const fixtures = vi.hoisted(() => {
  const event = {
    id: "event-free", slug: "cebu-run", name: "Cebu Run", date: "2026-09-28", location: "Cebu",
    status: "ACTIVE" as const, photoCount: 1, participantCount: 100, description: "Race",
    organizerName: "Cebu", categories: [], pricePerPhoto: 0, pricingMode: "free" as const,
    photographerHandle: "aira",
  };
  const photo = {
    id: "free-photo", metadataAccess: "public" as const, bib: "101", km: null, time: "06:30", tone: 0,
    span: "default" as const, price: 0, free: true, downloadUrl: "https://photos.test/free-photo",
  };
  const profile = {
    handle: "aira", displayName: "Aira", avatarUrl: null, brandColor: "none" as const,
    bio: "", city: "Cebu", memberSince: "2026-01-01", cover: null, watermarkLabel: null,
    events: [{ eventSlug: event.slug, state: "open" as const, photoCount: 1, salesCount: 0, event }],
  };
  return { event, photo, profile };
});

vi.mock("next/navigation", () => ({ useParams: () => ({ handle: "aira", slug: "cebu-run" }), notFound: () => { throw Error("Not found"); } }));
vi.mock("@/hooks/use-url-state", () => ({ useUrlState: (key: string) => [key === "browse" ? "1" : "", vi.fn()], useUrlStateBatch: () => vi.fn() }));
vi.mock("@/hooks/use-event-photos", () => ({
  useEventPhotos: () => ({ photos: [fixtures.photo], total: 1, isLoading: false, isFetching: false, hasNextPage: false, isFetchingNextPage: false, fetchNextPage: vi.fn(), error: null }),
  usePublicPhotographerEventPhotos: () => ({ items: [fixtures.photo], total: 1, isLoading: false, isFetching: false, hasNextPage: false, isFetchingNextPage: false, fetchNextPage: vi.fn(), refetch: vi.fn(), error: null }),
}));
vi.mock("@/hooks/use-event-live-photos", () => ({ useEventLivePhotos: () => ({ newCount: 0, reconnectFailed: false, refresh: vi.fn() }) }));
vi.mock("@/hooks/use-selfies", () => ({ useSelfiesList: () => ({ selfies: [], isLoading: false }) }));
vi.mock("@/components/events/photo-mosaic-tile", () => ({ PhotoMosaicTile: ({ onOpen }: { onOpen: () => void }) => <button onClick={onOpen}>Open gallery photo</button> }));
vi.mock("@/components/layout/site-header", () => ({ SiteHeader: () => null }));
vi.mock("@/components/profile-shell", () => ({ ProfileShellFooter: () => null }));
vi.mock("@/lib/api-events", () => ({ fetchEventDetail: vi.fn(async () => fixtures.event) }));
vi.mock("@/lib/api-photographer-public", () => ({ fetchPublicPhotographer: vi.fn(async () => fixtures.profile) }));

afterEach(() => vi.restoreAllMocks());

it.each(["event gallery", "photographer gallery"])("opens free %s metadata from the backend public hint", async (surface) => {
  const get = vi.spyOn(api, "get").mockResolvedValue({ photoId: "free-photo", status: "empty", originalFilename: "free.jpg", fileSizeBytes: 1024, fields: null });
  const user = userEvent.setup();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}>
    {surface === "event gallery" ?
      <EventCockpit event={fixtures.event} initialPhotos={{ items: [fixtures.photo], total: 1, offset: 0, limit: 60 }} /> :
      <HandleEventPage />}
  </QueryClientProvider>);
  await user.click(await screen.findByRole("button", { name: "Open gallery photo" }));
  expect(screen.getByRole("button", { name: "Download photo" })).toBeInTheDocument();
  expect(get).not.toHaveBeenCalled();
  await user.click(screen.getByRole("button", { name: "Show Metadata" }));
  expect(await screen.findByText("No camera metadata was found.")).toBeInTheDocument();
  expect(get).toHaveBeenCalledTimes(1);
  expect(get).toHaveBeenCalledWith("/public/photos/free-photo/metadata", expect.anything());
  client.clear();
});
