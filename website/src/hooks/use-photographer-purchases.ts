"use client";

import { useInfiniteQuery, useQueryClient, type InfiniteData } from "@tanstack/react-query";
import { useLayoutEffect, useRef, useState } from "react";
import { ApiError } from "@/lib/api";
import { dedupeItems } from "@/hooks/use-infinite-list";
import { usePrivateQuerySession } from "@/hooks/use-private-query-session";
import { PAGE_SIZE } from "@/lib/pagination-config";
import { useAuthStore } from "@/store/auth-store";
import {
  fetchPhotographerPurchasedPhotos,
  fetchPhotographerPhotoPurchases,
  type PhotographerPhotoPurchasesResponse,
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
    getNextPageParam: (last) => last.items.length && last.offset + last.limit < last.total ? last.offset + last.limit : undefined,
    enabled: active,
    retry,
  });
  const pages = query.data?.pages ?? [];
  const items = dedupeItems(pages.map((page) => ({ ...page, items: page.items.map((item: PhotographerPurchasedPhoto) => ({ id: item.photo.id, value: item })) }))).map(({ value }) => value);
  return {
    items,
    total: pages.at(-1)?.total ?? 0,
    isLoading: active && query.isPending,
    isPaused: active && query.isPaused,
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
  const [sessionUserId, sessionRole, sessionVersion] = session.key;
  const queryClient = useQueryClient();
  const active = session.enabled && enabled && !!photoId;
  const activeRef = useRef(active);
  const refreshController = useRef<AbortController | null>(null);
  const refreshPromise = useRef<Promise<void> | null>(null);
  const [isRefreshingPhoto, setIsRefreshingPhoto] = useState(false);
  const [photoRefreshError, setPhotoRefreshError] = useState<Error | null>(null);
  useLayoutEffect(() => {
    activeRef.current = active;
    refreshController.current?.abort();
    refreshController.current = null;
    refreshPromise.current = null;
    setIsRefreshingPhoto(false);
    setPhotoRefreshError(null);
    return () => {
      activeRef.current = false;
      refreshController.current?.abort();
    };
  }, [active, photoId, orderId, sessionUserId, sessionRole, sessionVersion]);
  const canRead = () => {
    const auth = useAuthStore.getState();
    return activeRef.current && auth.isAuthenticated && auth.user?.id === sessionUserId
      && auth.user?.role === "PHOTOGRAPHER" && auth.sessionVersion === sessionVersion;
  };
  const queryKey = ["photographer", "photos", photoId, "purchases", orderId ?? null, ...session.key] as const;
  const query = useInfiniteQuery({
    queryKey,
    queryFn: ({ pageParam, signal }) => {
      if (!canRead()) throw new DOMException("Seller session ended", "AbortError");
      return fetchPhotographerPhotoPurchases(photoId!, { orderId, offset: pageParam, limit: PAGE_SIZE.TRANSACTION_INCREMENT, signal });
    },
    initialPageParam: 0,
    getNextPageParam: (last) => last.items.length && last.offset + last.limit < last.total ? last.offset + last.limit : undefined,
    enabled: active,
    retry,
  });
  const pages = query.data?.pages ?? [];
  const items = dedupeItems(pages.map((page) => ({ ...page, items: page.items.map((item: PhotographerPurchase) => ({ id: `${item.orderId}:${item.photoId}`, value: item })) }))).map(({ value }) => value);
  const refreshPhoto = (selectedOrderId = orderId): Promise<void> => {
    if (refreshPromise.current) return refreshPromise.current;
    if (!canRead() || !photoId || !queryClient.getQueryData<InfiniteData<PhotographerPhotoPurchasesResponse>>(queryKey)?.pages.length) return Promise.resolve();

    const controller = new AbortController();
    const targetKey = queryKey;
    refreshController.current = controller;
    setIsRefreshingPhoto(true);
    setPhotoRefreshError(null);
    const pending = (async () => {
      try {
        const fresh = await fetchPhotographerPhotoPurchases(photoId, {
          orderId: selectedOrderId,
          offset: 0,
          limit: 1,
          signal: controller.signal,
        });
        if (controller.signal.aborted || refreshController.current !== controller || !canRead()) return;
        await queryClient.cancelQueries({ queryKey: targetKey, exact: true });
        if (controller.signal.aborted || refreshController.current !== controller || !canRead()) return;
        queryClient.setQueryData<InfiniteData<PhotographerPhotoPurchasesResponse>>(targetKey, (current) => current ? {
          ...current,
          pages: current.pages.map((page, index) => index === 0 ? { ...page, photo: fresh.photo, event: fresh.event } : page),
        } : current);
      } catch (error) {
        if (controller.signal.aborted || refreshController.current !== controller || !canRead()) return;
        const refreshError = error instanceof Error ? error : new Error("Could not refresh preview");
        setPhotoRefreshError(refreshError);
        throw refreshError;
      } finally {
        if (refreshController.current === controller) {
          refreshController.current = null;
          refreshPromise.current = null;
          setIsRefreshingPhoto(false);
        }
      }
    })();
    refreshPromise.current = pending;
    return pending;
  };
  return {
    items,
    total: pages.at(-1)?.total ?? 0,
    photo: pages[0]?.photo ?? null,
    event: pages[0]?.event ?? null,
    selectedPurchase: pages[0]?.selectedPurchase ?? null,
    isLoading: active && query.isPending,
    isPaused: active && query.isPaused,
    error: query.isFetchNextPageError ? null : query.error,
    appendError: query.isFetchNextPageError ? query.error : null,
    refreshPhoto,
    isRefreshingPhoto,
    photoRefreshError,
    refetch: () => canRead() ? query.refetch() : Promise.resolve(),
    fetchNextPage: () => canRead() ? query.fetchNextPage() : Promise.resolve(),
    hasNextPage: query.hasNextPage,
    isFetchingNextPage: query.isFetchingNextPage,
  };
}
