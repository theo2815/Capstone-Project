import { afterEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import EventPage from '@/app/dashboard/events/[id]/page';
import EventList from '@/app/dashboard/events/page';
import { useAuthStore } from '@/store/auth-store';

const mocks = vi.hoisted(() => ({
  post: vi.fn(async (path: string, body: { photoIds: string[] }) => path.endsWith('permanent-delete') ? { deleted: body.photoIds, blockedSold: [] } : { affected: body.photoIds.length }),
  toast: vi.fn(),
  rows: 120,
  purging: false,
  covered: [] as unknown[],
  activeRows: 1,
  photoStatus: "live",
  mediaAvailable: undefined as boolean | undefined,
  previewDownload: undefined as (() => void) | undefined,
  get: vi.fn(),
  save: vi.fn(async () => 'downloaded'),
  download: vi.fn(),
  put: vi.fn(),
  cancelQueries: vi.fn(async () => {}),
  invalidate: vi.fn(async () => {}),
  setQueryData: vi.fn(),
  removeQueries: vi.fn(),
  search: "",
  replace: vi.fn(),
  listeners: new Set<() => void>(),
  visibility: { isSuccess: true, isPending: false, isError: false, data: { metadataPublic: false } as { metadataPublic: boolean } | undefined, refetch: vi.fn() },
}));
vi.mock('next/navigation', async () => {
  const { useSyncExternalStore } = await import('react');
  return { useParams: () => ({ id: 'event-1' }),
    useSearchParams: () => new URLSearchParams(useSyncExternalStore((onChange) => { mocks.listeners.add(onChange); return () => { mocks.listeners.delete(onChange); }; }, () => mocks.search)),
    useRouter: () => ({ replace: (url: string, options: { scroll: boolean }) => { mocks.replace(url, options); mocks.search = new URL(url, 'http://test').search; mocks.listeners.forEach((notify) => notify()); } }),
    notFound: () => { throw Error('404'); } };
});
vi.mock('@tanstack/react-query', () => ({ useQueryClient: () => ({ cancelQueries: mocks.cancelQueries, invalidateQueries: mocks.invalidate, setQueryData: mocks.setQueryData, removeQueries: mocks.removeQueries }) }));
vi.mock('@/hooks/use-toast', () => ({ useToast: () => ({ showToast: mocks.toast }) }));
vi.mock('@/hooks/use-confirmation', () => ({ useConfirmation: () => ({ confirm: async () => true }) }));
vi.mock('@/components/layout/site-header', () => ({ SiteHeader: () => null }));
vi.mock('@/components/dashboard/event-coupon-modal', () => ({ EventCouponModal: () => null }));
vi.mock('@/components/dashboard/my-event-form', () => ({ MyEventForm: () => null }));
vi.mock('@/components/photos/photo-preview-card', () => ({ PhotoPreviewCard: ({ onDownload, metadataAccess, downloadDisabled }: { onDownload: () => void; metadataAccess?: string; downloadDisabled?: boolean }) => {
  mocks.previewDownload = onDownload;
  return <button data-metadata-access={metadataAccess} disabled={downloadDisabled} onClick={onDownload}>Download photo</button>;
} }));
vi.mock('@/lib/save-photo', () => ({ savePhoto: mocks.save, showInAppSaveHint: vi.fn() }));
vi.mock('@/lib/utils', async (original) => ({ ...await original<typeof import('@/lib/utils')>(), triggerDownload: mocks.download }));
vi.mock('@/components/profile-shell', () => ({ Slab: ({ children }: { children: React.ReactNode }) => <div>{children}</div> }));
vi.mock('@/components/events/event-tile', () => ({ EventTile: ({ event }: { event: { name: string } }) => <div>{event.name}</div> }));
vi.mock('@/lib/api', () => ({ api: { post: mocks.post, get: mocks.get, put: mocks.put }, refreshAccessToken: vi.fn() }));
vi.mock('@/hooks/use-photographer-data', () => ({
  COVERED_EVENTS_MAX: 200,
  usePhotographerEvents: () => mocks.covered,
  usePhotographerEventDetail: () => ({ isMissing: false, detail: {
    id: 'event-1', slug: 'test-event', name: 'Review Event', date: '2026-09-07',
    location: 'Cebu', state: 'past', photoCount: 1, salesCount: 0, revenueKept: 0,
    pricingMode: 'free', handle: 'reviewer', ownedByMe: false,
  } }),
  usePhotographerEventMetadataVisibility: () => mocks.visibility,
  usePhotographerEventPhotos: (_id: string, { deleted }: { deleted: boolean }) => ({
    items: Array.from({ length: deleted ? mocks.rows : mocks.activeRows }, (_, i) => ({
      status: mocks.photoStatus, id: `photo-${i}`, eventId: 'event-1', tone: 0, span: 'default', salesCount: 0,
      uploadedAt: '2026-09-07T00:00:00Z', deletedAt: deleted ? '2026-09-07T01:00:00Z' : null,
      deletionStartedAt: deleted && mocks.purging ? '2026-09-07T02:00:00Z' : null,
      mediaAvailable: mocks.mediaAvailable,
    })), total: deleted ? mocks.rows : 3028, isLoading: false, isFetchingNextPage: false,
    fetchNextPage: vi.fn(),
  }),
}));
afterEach(() => { cleanup(); vi.clearAllMocks(); useAuthStore.getState().logout(); mocks.mediaAvailable = undefined; mocks.activeRows = 1; mocks.photoStatus = "live"; mocks.search = ""; mocks.visibility = { isSuccess: true, isPending: false, isError: false, data: { metadataPublic: false }, refetch: vi.fn() }; });
describe('photographer photo management', () => {
  it('blocks ordinary owner preview media actions when the server reports unavailable', async () => {
    mocks.mediaAvailable = false;
    render(<EventPage />);
    fireEvent.click(screen.getByRole('button', { name: 'Open photo 1' }));
    const download = screen.getByRole('button', { name: 'Download photo' });
    expect(download).toBeDisabled();
    expect(download).not.toHaveAttribute('data-metadata-access');
    await act(async () => mocks.previewDownload?.());
    expect(mocks.get).not.toHaveBeenCalled();
    expect(mocks.save).not.toHaveBeenCalled();
  });
  it('uses the URL tab while keeping active and deleted selection controls', () => {
    mocks.search = '?tab=deleted';
    render(<EventPage />);
    expect(screen.getByRole('tab', { name: 'Deleted' })).toHaveAttribute('aria-selected', 'true');
    fireEvent.click(screen.getByRole('button', { name: 'Select' }));
    expect(screen.getByRole('button', { name: 'Restore 0' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('tab', { name: 'Active' }));
    expect(mocks.replace).toHaveBeenCalledWith('/dashboard/events/event-1?tab=active', { scroll: false });
  });
  it('clears selection and preview when an external URL changes the effective tab', () => {
    const { rerender } = render(<EventPage />);
    fireEvent.click(screen.getByRole('button', { name: 'Select' }));
    fireEvent.click(screen.getByRole('button', { name: 'Select all' }));
    expect(screen.getByRole('button', { name: 'Delete 1' })).toBeInTheDocument();
    mocks.search = '?tab=deleted';
    rerender(<EventPage />);
    expect(screen.getByRole('tab', { name: 'Deleted' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('button', { name: 'Select' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Delete 1' })).toBeNull();

    mocks.search = '?tab=active';
    rerender(<EventPage />);
    fireEvent.click(screen.getByRole('button', { name: 'Open photo 1' }));
    expect(screen.getByRole('button', { name: 'Download photo' })).toBeInTheDocument();
    mocks.search = '?tab=deleted';
    rerender(<EventPage />);
    expect(screen.queryByRole('button', { name: 'Download photo' })).toBeNull();
  });
  it('shows the server-false event choice on a shared event and saves only an explicit change', async () => {
    useAuthStore.getState().setUser({ id: 'photographer-a', role: 'PHOTOGRAPHER', name: 'A', email: 'a@example.test', createdAt: '2026-09-28' });
    mocks.put.mockResolvedValue({ metadataPublic: true });
    render(<EventPage />);
    const choice = screen.getByRole('checkbox', { name: 'Show metadata publicly for my photos' });
    expect(choice).not.toBeChecked();
    expect(screen.getByText('Only your photos in this event. You can always view your own metadata.')).toBeInTheDocument();
    expect(screen.getByText('This controls the metadata popup. Original downloads keep their embedded metadata.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Save metadata visibility' })).toBeDisabled();
    fireEvent.click(choice);
    fireEvent.click(screen.getByRole('button', { name: 'Cancel metadata change' }));
    expect(choice).not.toBeChecked();
    fireEvent.click(choice);
    fireEvent.click(screen.getByRole('button', { name: 'Save metadata visibility' }));
    await waitFor(() => expect(mocks.put).toHaveBeenCalledWith('/me/photographer/events/event-1/metadata-visibility', { metadataPublic: true }));
    await waitFor(() => expect(mocks.setQueryData).toHaveBeenCalled());
    expect(mocks.invalidate).toHaveBeenCalledWith({ queryKey: ['events', 'test-event', 'photos'] });
    expect(mocks.invalidate).toHaveBeenCalledWith({ queryKey: ['photographer', 'public', 'reviewer', 'test-event', 'photos'] });
    expect(mocks.removeQueries).toHaveBeenCalledWith({ queryKey: ['photo', 'metadata', 'public'] });
    expect(mocks.toast).toHaveBeenCalledWith({ kind: 'success', message: 'Metadata visibility saved.' });
  });

  it('blocks an unconfirmed event value, offers retry, and keeps a failed save draft', async () => {
    useAuthStore.getState().setUser({ id: 'photographer-a', role: 'PHOTOGRAPHER', name: 'A', email: 'a@example.test', createdAt: '2026-09-28' });
    mocks.visibility = { isSuccess: false, isPending: false, isError: true, data: undefined, refetch: vi.fn() };
    const { rerender } = render(<EventPage />);
    expect(screen.queryByRole('checkbox', { name: 'Show metadata publicly for my photos' })).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Retry metadata visibility' }));
    expect(mocks.visibility.refetch).toHaveBeenCalledOnce();
    mocks.visibility = { isSuccess: true, isPending: false, isError: false, data: { metadataPublic: false }, refetch: vi.fn() };
    mocks.put.mockRejectedValueOnce(new Error('offline'));
    rerender(<EventPage />);
    fireEvent.click(screen.getByRole('checkbox', { name: 'Show metadata publicly for my photos' }));
    fireEvent.click(screen.getByRole('button', { name: 'Save metadata visibility' }));
    await waitFor(() => expect(mocks.toast).toHaveBeenCalledWith({ kind: 'error', message: "Couldn't save metadata visibility." }));
    expect(screen.getByRole('checkbox', { name: 'Show metadata publicly for my photos' })).toBeChecked();
    expect(mocks.setQueryData).not.toHaveBeenCalled();
    expect(mocks.removeQueries).not.toHaveBeenCalled();
  });

  it('keeps an unsaved event choice across a visibility refetch error and recovery', () => {
    useAuthStore.getState().setUser({ id: 'photographer-a', role: 'PHOTOGRAPHER', name: 'A', email: 'a@example.test', createdAt: '2026-09-28' });
    const { rerender } = render(<EventPage />);
    fireEvent.click(screen.getByRole('checkbox', { name: 'Show metadata publicly for my photos' }));
    expect(screen.getByRole('checkbox', { name: 'Show metadata publicly for my photos' })).toBeChecked();

    mocks.visibility = { isSuccess: false, isPending: false, isError: true, data: { metadataPublic: false }, refetch: vi.fn() };
    rerender(<EventPage />);
    expect(screen.getByText("Couldn't load metadata visibility.")).toBeInTheDocument();
    mocks.visibility = { isSuccess: true, isPending: false, isError: false, data: { metadataPublic: false }, refetch: vi.fn() };
    rerender(<EventPage />);

    const choice = screen.getByRole('checkbox', { name: 'Show metadata publicly for my photos' });
    expect(choice).toBeChecked();
    expect(screen.getByRole('button', { name: 'Save metadata visibility' })).toBeEnabled();
    fireEvent.click(screen.getByRole('button', { name: 'Cancel metadata change' }));
    expect(choice).not.toBeChecked();
  });

  it('clears an unsaved event choice when the account changes role', () => {
    const user = { id: 'photographer-a', name: 'A', email: 'a@example.test', createdAt: '2026-09-28' };
    useAuthStore.getState().setUser({ ...user, role: 'PHOTOGRAPHER' });
    render(<EventPage />);
    fireEvent.click(screen.getByRole('checkbox', { name: 'Show metadata publicly for my photos' }));
    act(() => useAuthStore.getState().setUser({ ...user, role: 'RUNNER' }));
    expect(screen.queryByRole('checkbox', { name: 'Show metadata publicly for my photos' })).toBeNull();
    act(() => useAuthStore.getState().setUser({ ...user, role: 'PHOTOGRAPHER' }));
    expect(screen.getByRole('checkbox', { name: 'Show metadata publicly for my photos' })).not.toBeChecked();
  });

  it('ignores a save completed after the photographer session changes', async () => {
    useAuthStore.getState().setUser({ id: 'photographer-a', role: 'PHOTOGRAPHER', name: 'A', email: 'a@example.test', createdAt: '2026-09-28' });
    let resolve!: (value: { metadataPublic: boolean }) => void;
    mocks.put.mockReturnValueOnce(new Promise((yes) => { resolve = yes; }));
    render(<EventPage />);
    fireEvent.click(screen.getByRole('checkbox', { name: 'Show metadata publicly for my photos' }));
    fireEvent.click(screen.getByRole('button', { name: 'Save metadata visibility' }));
    act(() => { useAuthStore.getState().logout(); useAuthStore.getState().setUser({ id: 'photographer-b', role: 'PHOTOGRAPHER', name: 'B', email: 'b@example.test', createdAt: '2026-09-28' }); });
    await act(async () => resolve({ metadataPublic: true }));
    expect(mocks.setQueryData).not.toHaveBeenCalled();
    expect(mocks.invalidate).not.toHaveBeenCalled();
    expect(mocks.removeQueries).not.toHaveBeenCalled();
    expect(mocks.toast).not.toHaveBeenCalled();
  });

  it('labels processing photos until the refreshed library reports them live', () => {
    mocks.photoStatus = 'processing';
    const { rerender } = render(<EventPage />);
    expect(screen.getByText('Processing')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Open photo 1/ })).toHaveAccessibleName(/Processing/);
    mocks.photoStatus = 'live';
    rerender(<EventPage />);
    expect(screen.queryByText('Processing')).toBeNull();
    mocks.photoStatus = 'hidden';
    rerender(<EventPage />);
    expect(screen.queryByText('Processing')).toBeNull();
  });

  it('sends single-photo and single-selection saves through the same existing save helper', async () => {
    mocks.get.mockResolvedValue({ url: 'https://storage.test/photo?attachment=signed' });
    render(<EventPage />);
    fireEvent.click(screen.getByRole('button', { name: 'Open photo 1' }));
    expect(screen.getByRole('button', { name: 'Download photo' })).toHaveAttribute('data-metadata-access', 'owner');
    fireEvent.click(screen.getByRole('button', { name: 'Download photo' }));
    await waitFor(() => expect(mocks.save).toHaveBeenCalledOnce());
    fireEvent.click(screen.getByRole('button', { name: 'Select' }));
    fireEvent.click(screen.getByRole('button', { name: 'Select all' }));
    fireEvent.click(screen.getByRole('button', { name: 'Download 1' }));
    await waitFor(() => expect(mocks.save).toHaveBeenCalledTimes(2));
    expect(mocks.get).toHaveBeenLastCalledWith('/me/photographer/photos/photo-0/download');
    expect(mocks.save).toHaveBeenLastCalledWith('https://storage.test/photo?attachment=signed');
  });

  it('passes owner access for hidden and recoverably trashed previews', () => {
    mocks.photoStatus = 'hidden';
    render(<EventPage />);
    fireEvent.click(screen.getByRole('button', { name: 'Open photo 1' }));
    expect(screen.getByRole('button', { name: 'Download photo' })).toHaveAttribute('data-metadata-access', 'owner');
    fireEvent.click(screen.getByRole('tab', { name: 'Deleted' }));
    fireEvent.click(screen.getByRole('button', { name: 'Open photo 1' }));
    expect(screen.getByRole('button', { name: 'Download photo' })).toHaveAttribute('data-metadata-access', 'owner');
  });

  it('downloads all independently of 60 loaded tiles, then posts only selected IDs for a ZIP', async () => {
    mocks.activeRows = 60;
    mocks.post.mockResolvedValueOnce({ url: 'https://api.test/bundle?token=signed' } as never);
    render(<EventPage />);
    fireEvent.click(screen.getByRole('button', { name: 'Download all' }));
    await waitFor(() => expect(mocks.download).toHaveBeenCalledWith('https://api.test/bundle?token=signed', undefined));
    expect(mocks.post).toHaveBeenCalledWith('/me/photographer/events/event-1/download-bundle', {});
    mocks.post.mockResolvedValueOnce({ url: 'https://api.test/bundle?token=signed' } as never);
    fireEvent.click(screen.getByRole('button', { name: 'Select' }));
    fireEvent.click(screen.getByRole('button', { name: 'Select photo 1' }));
    fireEvent.click(screen.getByRole('button', { name: 'Select photo 2' }));
    fireEvent.click(screen.getByRole('button', { name: 'Download 2' }));
    await waitFor(() => expect(mocks.download).toHaveBeenLastCalledWith('https://api.test/bundle?token=signed', { photoIds: 'photo-0,photo-1' }));
    expect(screen.getByRole('button', { name: 'Delete 2' })).toBeEnabled();
    fireEvent.click(screen.getByRole('button', { name: 'Delete 2' }));
    await waitFor(() => expect(mocks.post).toHaveBeenLastCalledWith('/me/photographer/events/event-1/photos/trash', { photoIds: ['photo-0', 'photo-1'] }));
  });
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
