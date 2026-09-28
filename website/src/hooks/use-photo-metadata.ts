"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchPhotoMetadata } from "@/lib/api-photos";
import { usePrivateQuerySession } from "@/hooks/use-private-query-session";
import type { PhotoMetadataAccess } from "@/types/photo-metadata";

export function usePhotoMetadata(photoId: string, access: PhotoMetadataAccess | undefined, isOpen: boolean) {
  const viewer = usePrivateQuerySession();
  return useQuery({
    queryKey: ["photo", "metadata", access, photoId, ...viewer.key, isOpen],
    queryFn: ({ signal }) => fetchPhotoMetadata(photoId, access === "owner" ? "owner" : "public", signal),
    enabled: isOpen && !!photoId && (
      access === "public" ||
      (access === "owner" && viewer.enabled && viewer.key[1] === "PHOTOGRAPHER")
    ),
    staleTime: 0,
    refetchOnMount: "always",
    gcTime: 0,
  });
}
