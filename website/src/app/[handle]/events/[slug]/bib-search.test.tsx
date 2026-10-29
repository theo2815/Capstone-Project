import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, expect, it, vi, type MockInstance } from "vitest";
import { ApiError } from "@/lib/api";
import * as eventsApi from "@/lib/api-events";
import * as photographerApi from "@/lib/api-photographer-public";
import type { PhotographerProfile } from "@/lib/photographer-registry";
import type { EventDetail } from "@/types/event";
import type { MockPhoto } from "@/types/photo";
import HandleEventPage from "./page";

// INC-10 (STD v3): the photographer's event page searched a bib only among
// the 60 photos loaded in the browser. The bib must go to the server.

const search = { value: "" };
vi.mock("next/navigation", () => ({
  useParams: () => ({ handle: "shooter", slug: "nrw" }),
  useRouter: () => ({ replace: vi.fn(), push: vi.fn() }),
  usePathname: () => "/shooter/events/nrw",
  useSearchParams: () => new URLSearchParams(search.value),
  notFound: vi.fn(),
}));
vi.mock("@/components/layout/site-header", () => ({ SiteHeader: () => null }));
vi.mock("@/components/profile-shell", () => ({ ProfileShellFooter: () => null }));

const photo = (id: string, bib: string | null, handle = "shooter") =>
  ({ id, bib, km: null, time: "", tone: 0, span: "default", price: 0, photographerHandle: handle }) as MockPhoto;
const page = (items: MockPhoto[], total = items.length) => ({ items, total, offset: 0, limit: 60 });

// The gallery's first page: 60 photos, none with the searched bib, of 3,026.
const galleryPage = page(Array.from({ length: 60 }, (_, i) => photo(`g${i}`, "11111")), 3026);
let listSpy: MockInstance<typeof photographerApi.fetchPublicPhotographerEventPhotos>;

beforeEach(() => {
  vi.restoreAllMocks();
  vi.spyOn(eventsApi, "fetchEventDetail").mockResolvedValue({
    slug: "nrw", name: "NRW Fun Run", date: "2026-10-19", location: "Cebu", status: "completed",
    pricingMode: "free", pricePerPhoto: 0, visibility: "public", photoCount: 3026,
  } as unknown as EventDetail);
  vi.spyOn(photographerApi, "fetchPublicPhotographer").mockResolvedValue({
    handle: "shooter", displayName: "Shooter", brandColor: "none",
    events: [{ eventSlug: "nrw", photoCount: 3026, state: "open" }],
  } as unknown as PhotographerProfile);
  listSpy = vi.spyOn(photographerApi, "fetchPublicPhotographerEventPhotos");
});

function renderPage(query: string) {
  search.value = query;
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><HandleEventPage /></QueryClientProvider>);
}

it("a bib on no photo ends in the no-match panel, not an endless search", async () => {
  listSpy.mockImplementation(async (_h, _s, args) => (args?.bib ? page([]) : galleryPage));
  renderPage("bib=99999");

  expect(await screen.findByRole("heading", { name: "No matches yet." })).toBeInTheDocument();
  expect(screen.getByText(/BIB 99999/)).toBeInTheDocument();
});

it("a bib finds every match of this photographer on the server, not only loaded photos", async () => {
  const matches = [photo("m1", "10342"), photo("m2", "10342"), photo("m3", "10342")];
  listSpy.mockImplementation(async (_h, _s, args) => (args?.bib ? page(matches) : galleryPage));
  renderPage("bib=10342");

  expect(await screen.findByRole("heading", { name: /We found\s*3\s*photos/ })).toBeInTheDocument();
  expect(listSpy).toHaveBeenCalledWith("shooter", "nrw", expect.objectContaining({ bib: "10342" }), expect.anything());
  // The counter follows the bib list (3 of 3, so the "all loaded" line), not the gallery's 60 of 3,026.
  expect(screen.getByText("All 3 loaded")).toBeInTheDocument();
  expect(screen.queryByText(/Showing 60 of 3,026/)).not.toBeInTheDocument();
});

it("a bib search refused by the rate limit says so, not 'no matches'", async () => {
  listSpy.mockImplementation(async (_h, _s, args) => {
    if (args?.bib) throw new ApiError([], 429, 60);
    return galleryPage;
  });
  renderPage("bib=10342");

  expect(await screen.findByRole("heading", { name: "Search didn't finish." })).toBeInTheDocument();
  expect(screen.getByText(/Too many searches/)).toBeInTheDocument();
});

it("a bib typed as B-10342 asks the server for 10342", async () => {
  listSpy.mockImplementation(async (_h, _s, args) => (args?.bib ? page([photo("m1", "10342")]) : galleryPage));
  renderPage("bib=B-10342");

  await waitFor(() =>
    expect(listSpy).toHaveBeenCalledWith("shooter", "nrw", expect.objectContaining({ bib: "10342" }), expect.anything()),
  );
});
