import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { afterEach, expect, it, vi } from 'vitest';
import EventPage from '@/app/dashboard/events/[id]/page';
import { api } from '@/lib/api';
import { useAuthStore } from '@/store/auth-store';

const mocks = vi.hoisted(() => ({ toast: vi.fn() }));
vi.mock('next/navigation', () => ({ useParams: () => ({ id: 'event-1' }), notFound: () => { throw Error('404'); } }));
vi.mock('@/hooks/use-toast', () => ({ useToast: () => ({ showToast: mocks.toast }) }));
vi.mock('@/hooks/use-confirmation', () => ({ useConfirmation: () => ({ confirm: async () => true }) }));
vi.mock('@/components/layout/site-header', () => ({ SiteHeader: () => null }));
vi.mock('@/hooks/use-photographer-data', async (original) => ({
  ...await original<typeof import('@/hooks/use-photographer-data')>(),
  usePhotographerEventDetail: () => ({ isMissing: false, detail: {
    id: 'event-1', slug: 'test-event', name: 'Test Event', date: '2026-09-07', location: 'Cebu',
    state: 'past', photoCount: 0, salesCount: 0, revenueKept: 0, pricingMode: 'free',
    handle: 'reviewer', ownedByMe: false,
  } }),
  usePhotographerEventPhotos: () => ({ items: [], total: 0, isLoading: false, isFetchingNextPage: false, fetchNextPage: vi.fn() }),
}));

afterEach(() => { cleanup(); vi.restoreAllMocks(); useAuthStore.getState().logout(); });

it('keeps a confirmed event choice when an older visibility GET completes', async () => {
  useAuthStore.getState().setUser({ id: 'photographer-a', role: 'PHOTOGRAPHER', name: 'A', email: 'a@example.test', createdAt: '2026-09-28' });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const key = ['photographer', 'events', 'event-1', 'metadata-visibility', 'photographer-a', 'PHOTOGRAPHER', useAuthStore.getState().sessionVersion];
  client.setQueryData(key, { metadataPublic: false }, { updatedAt: Date.now() - 120_000 });
  let resolveRead!: (value: { metadataPublic: boolean }) => void;
  const get = vi.spyOn(api, 'get').mockReturnValue(new Promise((resolve) => { resolveRead = resolve; }));
  vi.spyOn(api, 'put').mockResolvedValue({ metadataPublic: true });

  render(<QueryClientProvider client={client}><EventPage /></QueryClientProvider>);
  await waitFor(() => expect(get).toHaveBeenCalledWith('/me/photographer/events/event-1/metadata-visibility', expect.objectContaining({ signal: expect.any(AbortSignal) })));
  fireEvent.click(screen.getByRole('checkbox', { name: 'Show metadata publicly for my photos' }));
  fireEvent.click(screen.getByRole('button', { name: 'Save metadata visibility' }));
  await waitFor(() => expect(mocks.toast).toHaveBeenCalledWith({ kind: 'success', message: 'Metadata visibility saved.' }));
  expect(client.getQueryData(key)).toEqual({ metadataPublic: true });

  await act(async () => resolveRead({ metadataPublic: false }));
  await waitFor(() => expect(client.isFetching()).toBe(0));
  expect(client.getQueryData(key)).toEqual({ metadataPublic: true });
  expect(screen.getByRole('checkbox', { name: 'Show metadata publicly for my photos' })).toBeChecked();
  client.clear();
});
