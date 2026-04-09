"use client";

import { PhotoGrid } from "@/components/photos/photo-grid";
import { Badge } from "@/components/ui/badge";
import type { SearchResult } from "@/types/photo";

interface SearchResultsProps {
  results: SearchResult[];
  pricePerPhoto: number;
  isLoading?: boolean;
}

export function SearchResults({
  results,
  pricePerPhoto,
  isLoading,
}: SearchResultsProps) {
  const photos = results.map((r) => r.photo);

  return (
    <div className="space-y-4">
      {results.length > 0 && (
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-600">
            Found {results.length} {results.length === 1 ? "photo" : "photos"}
          </span>
          {results[0] && (
            <Badge variant="info">
              {results[0].matchType === "FACE" ? "Face match" : "Bib match"}
            </Badge>
          )}
        </div>
      )}

      <PhotoGrid
        photos={photos}
        pricePerPhoto={pricePerPhoto}
        isLoading={isLoading}
      />
    </div>
  );
}
