"use client";

import { PhotoCard } from "./photo-card";
import { Skeleton } from "@/components/ui/skeleton";
import type { Photo } from "@/types/photo";

interface PhotoGridProps {
  photos: Photo[];
  pricePerPhoto: number;
  isLoading?: boolean;
  onPhotoClick?: (photo: Photo) => void;
}

export function PhotoGrid({
  photos,
  pricePerPhoto,
  isLoading,
  onPhotoClick,
}: PhotoGridProps) {
  if (isLoading) {
    return (
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
        {Array.from({ length: 12 }).map((_, i) => (
          <div key={i} className="overflow-hidden rounded-lg">
            <Skeleton className="aspect-[3/2] w-full" />
            <div className="p-3">
              <Skeleton className="h-4 w-20" />
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (photos.length === 0) {
    return (
      <div className="py-12 text-center">
        <p className="text-gray-500">No photos found.</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
      {photos.map((photo) => (
        <PhotoCard
          key={photo.id}
          photo={photo}
          price={pricePerPhoto}
          onClick={() => onPhotoClick?.(photo)}
        />
      ))}
    </div>
  );
}
