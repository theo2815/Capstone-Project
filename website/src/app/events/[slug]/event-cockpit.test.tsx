import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { EventCockpit } from "@/app/events/[slug]/event-cockpit";
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
  return { event, matches };
});

vi.mock("@/hooks/use-url-state", () => ({
  // Arrive from the "your photos are ready" email: face mode, auto-run search.
  useUrlState: (key: string) => [["browse", "face", "mine"].includes(key) ? "1" : "", vi.fn()],
  useUrlStateBatch: () => vi.fn(),
}));
vi.mock("@/hooks/use-event-photos", () => ({
  useEventPhotos: () => ({ photos: [], total: 0, isLoading: false, isFetching: false, hasNextPage: false, isFetchingNextPage: false, fetchNextPage: vi.fn(), error: null }),
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
