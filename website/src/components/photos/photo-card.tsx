"use client";

import Image from "next/image";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useCart } from "@/hooks/use-cart";
import { ShoppingCart, Check } from "lucide-react";
import type { Photo } from "@/types/photo";

interface PhotoCardProps {
  photo: Photo;
  price: number;
  onClick?: () => void;
}

export function PhotoCard({ photo, price, onClick }: PhotoCardProps) {
  const { addItem, hasItem } = useCart();
  const inCart = hasItem(photo.id);

  return (
    <div className="group overflow-hidden rounded-lg border border-gray-200 bg-white">
      <div
        className="relative aspect-[3/2] cursor-pointer bg-gray-100"
        onClick={onClick}
      >
        <Image
          src={photo.watermarkedUrl}
          alt={`Photo ${photo.id}`}
          fill
          className="object-cover transition-transform group-hover:scale-105"
        />
        <div className="absolute bottom-2 left-2 flex gap-1">
          {photo.tags.map((tag, i) => (
            <Badge key={i} variant="info">
              {tag.type === "BIB" ? `#${tag.value}` : tag.value}
            </Badge>
          ))}
        </div>
      </div>

      <div className="flex items-center justify-between p-3">
        <span className="text-sm font-medium text-gray-900">
          PHP {price.toFixed(2)}
        </span>
        <Button
          size="sm"
          variant={inCart ? "secondary" : "primary"}
          onClick={() =>
            !inCart &&
            addItem({
              photoId: photo.id,
              eventId: photo.eventId,
              thumbnailUrl: photo.thumbnailUrl,
              price,
            })
          }
          disabled={inCart}
        >
          {inCart ? (
            <>
              <Check className="mr-1 h-3.5 w-3.5" /> In Cart
            </>
          ) : (
            <>
              <ShoppingCart className="mr-1 h-3.5 w-3.5" /> Add
            </>
          )}
        </Button>
      </div>
    </div>
  );
}
