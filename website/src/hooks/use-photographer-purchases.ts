"use client";

import { useInfiniteQuery } from "@tanstack/react-query";
import { ApiError } from "@/lib/api";
import { dedupeItems } from "@/hooks/use-infinite-list";
import { usePrivateQuerySession } from "@/hooks/use-private-query-session";
import { PAGE_SIZE } from "@/lib/pagination-config";
import {
  fetchPhotographerPurchasedPhotos,
  fetchPhotographerPhotoPurchases,
  type PhotographerPurchasedPhoto,
  type PhotographerPurchase,
} from "@/lib/api-photographer-earnings";

const retry = (failureCount: number, err: Error) =>
  !(err instanceof ApiError && (err.status === 404 || err.status === 429)) && failureCount < 3;

export function usePhotographerPurchasedPhotos(eventId: string | null, enabled: boolean) {
  const session = usePrivateQuerySession("PHOTOGRAPHER");
  const active = session.enabled && enabled && !!eventId;
  const query = useInfiniteQuery({
    queryKey: ["photographer", "events", eventId, "purchased-photos", ...session.key],
    queryFn: ({ pageParam, signal }) => fetchPhotographerPurchasedPhotos(eventId!, { offset: pageParam, limit: PAGE_SIZE.PHOTO_INCREMENT, signal }),
    initialPageParam: 0,
    getNextPageParam: (last) => last.offset + last.items.length < last.total ? last.offset + last.items.length : undefined,
    enabled: active,
    retry,
  });
  const pages = query.data?.pages ?? [];
  const items = dedupeItems(pages.map((page) => ({ ...page, items: page.items.map((item: PhotographerPurchasedPhoto) => ({ id: item.photo.id, value: item })) }))).map(({ value }) => value);
  return {
    items,
    total: pages.at(-1)?.total ?? 0,
    isLoading: active && query.isPending,
    error: query.isFetchNextPageError ? null : query.error,
    appendError: query.isFetchNextPageError ? query.error : null,
    refetch: query.refetch,
    fetchNextPage: query.fetchNextPage,
    hasNextPage: query.hasNextPage,
    isFetchingNextPage: query.isFetchingNextPage,
  };
}

export function usePhotographerPhotoPurchases(photoId: string | null, orderId?: string, enabled = true) {
  const session = usePrivateQuerySession("PHOTOGRAPHER");
  const active = session.enabled && enabled && !!photoId;
  const query = useInfiniteQuery({
    queryKey: ["photographer", "photos", photoId, "purchases", orderId ?? null, ...session.key],
    queryFn: ({ pageParam, signal }) => fetchPhotographerPhotoPurchases(photoId!, { orderId, offset: pageParam, limit: PAGE_SIZE.TRANSACTION_INCREMENT, signal }),
    initialPageParam: 0,
    getNextPageParam: (last) => last.offset + last.items.length < last.total ? last.offset + last.items.length : undefined,
    enabled: active,
    retry,
  });
  const pages = query.data?.pages ?? [];
  const items = dedupeItems(pages.map((page) => ({ ...page, items: page.items.map((item: PhotographerPurchase) => ({ id: `${item.orderId}:${item.photoId}`, value: item })) }))).map(({ value }) => value);
  return {
    items,
    total: pages.at(-1)?.total ?? 0,
    photo: pages[0]?.photo ?? null,
    event: pages[0]?.event ?? null,
    selectedPurchase: pages[0]?.selectedPurchase ?? null,
    isLoading: active && query.isPending,
    error: query.isFetchNextPageError ? null : query.error,
    appendError: query.isFetchNextPageError ? query.error : null,
    refetch: query.refetch,
    fetchNextPage: query.fetchNextPage,
    hasNextPage: query.hasNextPage,
    isFetchingNextPage: query.isFetchingNextPage,
  };
}
