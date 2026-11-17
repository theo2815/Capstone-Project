import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { afterEach, expect, it, vi } from 'vitest';
import EventPage from '@/app/dashboard/events/[id]/page';
import { api } from '@/lib/api';

const mocks = vi.hoisted(() => ({ race: {} as Record<string, unknown>, toast: vi.fn() }));
vi.mock('next/navigation', () => ({ useParams: () => ({ id: 'race-1' }), useSearchParams: () => new URLSearchParams(''), useRouter: () => ({ replace: vi.fn() }), notFound: () => { throw Error('404'); } }));
vi.mock('@/hooks/use-toast', () => ({ useToast: () => ({ showToast: mocks.toast }) }));
vi.mock('@/hooks/use-confirmation', () => ({ useConfirmation: () => ({ confirm: async () => true }) }));
vi.mock('@/components/layout/site-header', () => ({ SiteHeader: () => null }));
vi.mock('@/hooks/use-photographer-data', async (original) => ({
  ...await original<typeof import('@/hooks/use-photographer-data')>(),
  usePhotographerEventDetail: () => ({ isMissing: false, detail: {
    id: 'race-1', slug: 'cebu-marathon-2027', name: 'Cebu Marathon 2027', date: '2027-01-10', location: 'Cebu',
    state: 'upcoming', photoCount: 0, salesCount: 0, revenueKept: 0, pricingMode: 'paid', pricePerPhoto: 125,
    handle: 'me', ownedByMe: false, deletedAt: null, eventType: 'ROAD_RACE', submittedByMe: true,
    reviewStatus: 'approved', checkReasons: [], hidden: false, canEditRace: false, ...mocks.race,
  } }),
  usePhotographerEventPhotos: () => ({ items: [], total: 0, isLoading: false, isFetchingNextPage: false, fetchNextPage: vi.fn() }),
}));
vi.mock('@/hooks/use-photographer-purchases', () => ({ usePhotographerPurchasedPhotos: () => ({ items: [], total: 0, isLoading: false, error: null, appendError: null, hasNextPage: false, refetch: vi.fn(), fetchNextPage: vi.fn() }), usePhotographerPhotoPurchases: vi.fn() }));

afterEach(() => { cleanup(); vi.restoreAllMocks(); mocks.race = {}; });

function mount() {
  vi.spyOn(api, 'get').mockResolvedValue({ items: [], total: 0, offset: 0, limit: 60 });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><EventPage /></QueryClientProvider>);
}

it('edit shows only when canEditRace', () => {
  mount();
  expect(screen.getByText('Your race')).toBeInTheDocument();
  expect(screen.queryByRole('button', { name: 'Edit race' })).toBeNull();
  cleanup();
  mocks.race = { canEditRace: true };
  mount();
  expect(screen.getByRole('button', { name: 'Edit race' })).toBeInTheDocument();
});

it('a race edit sends only the changed field to the race route and shows the held line', async () => {
  mocks.race = { canEditRace: true, description: '', organizerName: '' };
  const patch = vi.spyOn(api, 'fetch').mockResolvedValue({ id: 'race-1', reviewStatus: 'change_pending' });
  mount();
  fireEvent.click(screen.getByRole('button', { name: 'Edit race' }));
  fireEvent.change(screen.getByLabelText('Title'), { target: { value: 'Cebu Marathon 2027 Edition' } });
  fireEvent.click(screen.getByRole('button', { name: 'Save changes' }));
  await waitFor(() => expect(mocks.toast).toHaveBeenCalledWith({ kind: 'success', message: 'Your change is waiting for an admin; the race keeps its current details.' }));
  expect(patch.mock.calls[0][0]).toBe('/me/photographer/races/race-1');
  const body = (patch.mock.calls[0][1] as { body: FormData }).body;
  expect([...body.keys()]).toEqual(['title']);
});

it('a held race shows the change pending line and its check reasons', () => {
  mocks.race = { reviewStatus: 'change_pending', checkReasons: ['COVER_FLAGGED'] };
  mount();
  expect(screen.getByText('Your change is waiting for an admin; the race keeps its current details.')).toBeInTheDocument();
  expect(screen.getByText(/cover image needs an admin to check it/)).toBeInTheDocument();
});
