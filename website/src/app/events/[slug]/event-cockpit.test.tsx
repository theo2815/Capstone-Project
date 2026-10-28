import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { EventCockpit } from "@/app/events/[slug]/event-cockpit";
import { ApiError } from "@/lib/api";
import { useAuthStore } from "@/store/auth-store";

// FS-7 (2026-10-27): a face search can return up to 200 matches. They render a
// page at a time like the bib gallery, instead of every tile and image at once.
const fixtures = vi.hoisted(() => {
  const event = {
    id: "event-1", slug: "cebu-run", name: "Cebu Run", date: "2026-09-28", location: "Cebu",
    status: "ACTIVE" as const, photoCount: 130, participantCount: 100, description: "Race",
    organizerName: "Cebu", categories: [], pricePerPhoto: 100, pricingMode: "paid" as const,
  };
  const matches = Array.from({ length: 130 }, (_, i) => ({
    id: `match-${i}`, bib: null, km: null, time: "06:30", tone: 0, span: "default" as const, price: 100,
  }));
  // Per-test url flags and bib-search error (the default is the ?mine=1 landing).
  const state = { url: { browse: "1", face: "1", mine: "1" } as Record<string, string>, bibError: null as unknown };
  return { event, matches, state };
});

vi.mock("@/hooks/use-url-state", () => ({
  // Arrive from the "your photos are ready" email: face mode, auto-run search.
  useUrlState: (key: string) => [fixtures.state.url[key] ?? "", vi.fn()],
  useUrlStateBatch: () => vi.fn(),
}));
vi.mock("@/hooks/use-event-photos", () => ({
  useEventPhotos: () => ({ photos: [], total: 0, isLoading: false, isFetching: false, hasNextPage: false, isFetchingNextPage: false, fetchNextPage: vi.fn(), error: fixtures.state.bibError }),
}));
vi.mock("@/hooks/use-event-live-photos", () => ({ useEventLivePhotos: () => ({ newCount: 0, reconnectFailed: false, refresh: vi.fn() }) }));
vi.mock("@/hooks/use-selfies", () => ({ useSelfiesList: () => ({ selfies: [{ id: "s1", isPrimary: true }], isLoading: false }) }));
vi.mock("@/lib/api-photo-alert", () => ({ fetchPhotoAlertStatus: vi.fn(async () => ({ registered: true, selfieId: "s1" })) }));
vi.mock("@/lib/api-photos", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api-photos")>()),
  searchEventByFace: vi.fn(async () => ({ items: fixtures.matches, total: 130, offset: 0, limit: 200 })),
}));
vi.mock("@/components/events/photo-mosaic-tile", () => ({ PhotoMosaicTile: () => <div data-testid="tile" /> }));
vi.mock("@/components/layout/site-header", () => ({ SiteHeader: () => null }));

beforeEach(() => {
  useAuthStore.setState({
    user: { id: "u1", email: "runner@example.com", name: "R", role: "RUNNER" } as never,
    isAuthenticated: true,
    isLoading: false,
  });
});

afterEach(() => {
  fixtures.state.url = { browse: "1", face: "1", mine: "1" };
  fixtures.state.bibError = null;
  useAuthStore.setState({ user: null, isAuthenticated: false, isLoading: false });
  vi.restoreAllMocks();
});

it("shows face matches a page at a time, while Buy all still covers every match", async () => {
  const user = userEvent.setup();
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <EventCockpit event={fixtures.event} initialPhotos={{ items: [], total: 0, offset: 0, limit: 60 }} />
    </QueryClientProvider>,
  );

  expect(await screen.findByText("Showing 60 of 130")).toBeInTheDocument();
  expect(screen.getAllByTestId("tile")).toHaveLength(60);
  expect(screen.getByRole("button", { name: /Buy all 130/ })).toBeInTheDocument();

  await user.click(screen.getByRole("button", { name: "Load 60 more" }));
  expect(screen.getAllByTestId("tile")).toHaveLength(120);

  await user.click(screen.getByRole("button", { name: "Load 10 more" }));
  expect(screen.getAllByTestId("tile")).toHaveLength(130);
  expect(screen.getByText("All 130 loaded")).toBeInTheDocument();
});

it("says how long to wait when a bib search is refused with a 429, not 'No matches yet'", async () => {
  fixtures.state.url = { bib: "123" };
  fixtures.state.bibError = new ApiError([{ code: "RATE_LIMITED", message: "Too many requests. Please slow down and try again." }], 429, 19);
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <EventCockpit event={fixtures.event} initialPhotos={{ items: [], total: 0, offset: 0, limit: 60 }} />
    </QueryClientProvider>,
  );

  expect(await screen.findByText("Too many searches. Try again in about 19 seconds.")).toBeInTheDocument();
  expect(screen.queryByText(/No matches yet|Still uploading/)).not.toBeInTheDocument();
});
