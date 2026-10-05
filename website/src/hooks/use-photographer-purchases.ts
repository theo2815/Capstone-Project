"use client";

import { useInfiniteQuery } from "@tanstack/react-query";
import { useLayoutEffect, useRef } from "react";
import { ApiError } from "@/lib/api";
import { dedupeItems } from "@/hooks/use-infinite-list";
import { usePrivateQuerySession } from "@/hooks/use-private-query-session";
import { PAGE_SIZE } from "@/lib/pagination-config";
import { useAuthStore } from "@/store/auth-store";
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
  const activeRef = useRef(active);
  useLayoutEffect(() => { activeRef.current = active; return () => { activeRef.current = false; }; }, [active]);
  const canRead = () => {
    const auth = useAuthStore.getState();
    return activeRef.current && auth.isAuthenticated && auth.user?.id === session.key[0]
      && auth.user?.role === "PHOTOGRAPHER" && auth.sessionVersion === session.key[2];
  };
  const query = useInfiniteQuery({
    queryKey: ["photographer", "events", eventId, "purchased-photos", ...session.key],
    queryFn: ({ pageParam, signal }) => {
      if (!canRead()) throw new DOMException("Seller session ended", "AbortError");
      return fetchPhotographerPurchasedPhotos(eventId!, { offset: pageParam, limit: PAGE_SIZE.PHOTO_INCREMENT, signal });
    },
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
    refetch: () => canRead() ? query.refetch() : Promise.resolve(),
    fetchNextPage: () => canRead() ? query.fetchNextPage() : Promise.resolve(),
    hasNextPage: query.hasNextPage,
    isFetchingNextPage: query.isFetchingNextPage,
  };
}

export function usePhotographerPhotoPurchases(photoId: string | null, orderId?: string, enabled = true) {
  const session = usePrivateQuerySession("PHOTOGRAPHER");
  const active = session.enabled && enabled && !!photoId;
  const activeRef = useRef(active);
  useLayoutEffect(() => { activeRef.current = active; return () => { activeRef.current = false; }; }, [active]);
  const canRead = () => {
    const auth = useAuthStore.getState();
    return activeRef.current && auth.isAuthenticated && auth.user?.id === session.key[0]
      && auth.user?.role === "PHOTOGRAPHER" && auth.sessionVersion === session.key[2];
  };
  const query = useInfiniteQuery({
    queryKey: ["photographer", "photos", photoId, "purchases", orderId ?? null, ...session.key],
    queryFn: ({ pageParam, signal }) => {
      if (!canRead()) throw new DOMException("Seller session ended", "AbortError");
      return fetchPhotographerPhotoPurchases(photoId!, { orderId, offset: pageParam, limit: PAGE_SIZE.TRANSACTION_INCREMENT, signal });
    },
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
    refetch: () => canRead() ? query.refetch() : Promise.resolve(),
    fetchNextPage: () => canRead() ? query.fetchNextPage() : Promise.resolve(),
    hasNextPage: query.hasNextPage,
    isFetchingNextPage: query.isFetchingNextPage,
  };
}
