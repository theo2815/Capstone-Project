"use client";

import Image from "next/image";
import { Modal } from "@/components/ui/modal";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useCart } from "@/hooks/use-cart";
import { ShoppingCart, Check, ChevronLeft, ChevronRight } from "lucide-react";
import type { Photo } from "@/types/photo";

interface PhotoLightboxProps {
  photo: Photo | null;
  price: number;
  onClose: () => void;
  onPrev?: () => void;
  onNext?: () => void;
}

export function PhotoLightbox({
  photo,
  price,
  onClose,
  onPrev,
  onNext,
}: PhotoLightboxProps) {
  const { addItem, hasItem } = useCart();

  if (!photo) return null;

  const inCart = hasItem(photo.id);

  return (
    <Modal isOpen={!!photo} onClose={onClose} className="max-w-4xl">
      <div className="relative aspect-[3/2] bg-gray-100">
        <Image
          src={photo.watermarkedUrl}
          alt={`Photo ${photo.id}`}
          fill
          className="object-contain"
        />
        {onPrev && (
          <button
            onClick={onPrev}
            className="absolute left-2 top-1/2 -translate-y-1/2 rounded-full bg-white/80 p-2 shadow hover:bg-white"
          >
            <ChevronLeft className="h-5 w-5" />
          </button>
        )}
        {onNext && (
          <button
            onClick={onNext}
            className="absolute right-2 top-1/2 -translate-y-1/2 rounded-full bg-white/80 p-2 shadow hover:bg-white"
          >
            <ChevronRight className="h-5 w-5" />
          </button>
        )}
      </div>

      <div className="mt-4 flex items-center justify-between">
        <div className="flex gap-2">
          {photo.tags.map((tag, i) => (
            <Badge key={i} variant="info">
              {tag.type === "BIB" ? `#${tag.value}` : tag.value}
            </Badge>
          ))}
        </div>
        <div className="flex items-center gap-4">
          <span className="text-lg font-semibold">
            PHP {price.toFixed(2)}
          </span>
          <Button
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
                <Check className="mr-1 h-4 w-4" /> In Cart
              </>
            ) : (
              <>
                <ShoppingCart className="mr-1 h-4 w-4" /> Add to Cart
              </>
            )}
          </Button>
        </div>
      </div>
    </Modal>
  );
}
