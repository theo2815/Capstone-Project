import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import EventPage from '@/app/dashboard/events/[id]/page';
import EventList from '@/app/dashboard/events/page';

const mocks = vi.hoisted(() => ({
  post: vi.fn(async (path: string, body: { photoIds: string[] }) => path.endsWith('permanent-delete') ? { deleted: body.photoIds, blockedSold: [] } : { affected: body.photoIds.length }),
  toast: vi.fn(),
  rows: 120,
  purging: false,
  covered: [] as unknown[],
}));
vi.mock('next/navigation', () => ({ useParams: () => ({ id: 'event-1' }), notFound: () => { throw Error('404'); } }));
vi.mock('@tanstack/react-query', () => ({ useQueryClient: () => ({ invalidateQueries: async () => {} }) }));
vi.mock('@/hooks/use-toast', () => ({ useToast: () => ({ showToast: mocks.toast }) }));
vi.mock('@/hooks/use-confirmation', () => ({ useConfirmation: () => ({ confirm: async () => true }) }));
vi.mock('@/components/layout/site-header', () => ({ SiteHeader: () => null }));
vi.mock('@/components/dashboard/event-coupon-modal', () => ({ EventCouponModal: () => null }));
vi.mock('@/components/dashboard/my-event-form', () => ({ MyEventForm: () => null }));
vi.mock('@/components/photos/photo-preview-card', () => ({ PhotoPreviewCard: () => null }));
vi.mock('@/components/profile-shell', () => ({ Slab: ({ children }: { children: React.ReactNode }) => <div>{children}</div> }));
vi.mock('@/components/events/event-tile', () => ({ EventTile: ({ event }: { event: { name: string } }) => <div>{event.name}</div> }));
vi.mock('@/lib/api', () => ({ api: { post: mocks.post }, refreshAccessToken: vi.fn() }));
vi.mock('@/hooks/use-photographer-data', () => ({
  COVERED_EVENTS_MAX: 200,
  usePhotographerEvents: () => mocks.covered,
  usePhotographerEventDetail: () => ({ isMissing: false, detail: {
    id: 'event-1', slug: 'test-event', name: 'Review Event', date: '2026-09-07',
    location: 'Cebu', state: 'past', photoCount: 1, salesCount: 0, revenueKept: 0,
    pricingMode: 'free', handle: 'reviewer',
  } }),
  usePhotographerEventPhotos: (_id: string, { deleted }: { deleted: boolean }) => ({
    items: Array.from({ length: deleted ? mocks.rows : 1 }, (_, i) => ({
      id: `photo-${i}`, eventId: 'event-1', tone: 0, span: 'default', salesCount: 0,
      uploadedAt: '2026-09-07T00:00:00Z', deletedAt: deleted ? '2026-09-07T01:00:00Z' : null,
      deletionStartedAt: deleted && mocks.purging ? '2026-09-07T02:00:00Z' : null,
    })), total: deleted ? mocks.rows : 1, isLoading: false, isFetchingNextPage: false,
    fetchNextPage: vi.fn(),
  }),
}));
afterEach(() => { cleanup(); vi.clearAllMocks(); });
describe('photographer photo management', () => {
  it('permanent deletion must respect the backend maximum of 100 IDs per request', async () => {
    render(<EventPage />);
    fireEvent.click(screen.getByRole('tab', { name: 'Deleted' }));
    fireEvent.click(screen.getByRole('button', { name: 'Select' }));
    fireEvent.click(screen.getByRole('button', { name: 'Select all' }));
    fireEvent.click(screen.getByRole('button', { name: 'Delete forever' }));
    await waitFor(() => expect(mocks.post).toHaveBeenCalledTimes(2));
    expect(mocks.post.mock.calls.map(([, body]) => body.photoIds.length)).toEqual([100, 20]);
  });
  it('a covered event with only trashed photos must remain reachable from My events', () => {
    mocks.covered = [{
      id: 'event-1', slug: 'test-event', name: 'Trash-only Event', date: '2026-09-07',
      location: 'Cebu', state: 'past', photoCount: 0, salesCount: 0, revenueKept: 0,
      ownedByMe: false,
    }];
    render(<EventList />);
    expect(screen.queryByText('Trash-only Event')).not.toBeNull();
  });

  it('does not offer restore after permanent deletion has started', () => {
    mocks.purging = true;
    render(<EventPage />);
    fireEvent.click(screen.getByRole('tab', { name: 'Deleted' }));
    fireEvent.click(screen.getByRole('button', { name: 'Select' }));
    fireEvent.click(screen.getByRole('button', { name: 'Select all' }));
    expect(screen.getByRole('button', { name: 'Restore 0' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Delete forever' })).toBeEnabled();
    mocks.purging = false;
  });
});
