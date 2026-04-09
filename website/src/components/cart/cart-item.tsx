"use client";

import Image from "next/image";
import { useCart } from "@/hooks/use-cart";
import { formatPrice } from "@/lib/utils";
import { Trash2 } from "lucide-react";

interface CartItemProps {
  photoId: string;
  thumbnailUrl: string;
  price: number;
}

export function CartItem({ photoId, thumbnailUrl, price }: CartItemProps) {
  const { removeItem } = useCart();

  return (
    <div className="flex items-center gap-4 rounded-lg border border-gray-200 p-3">
      <div className="relative h-16 w-24 shrink-0 overflow-hidden rounded bg-gray-100">
        <Image
          src={thumbnailUrl}
          alt="Photo"
          fill
          className="object-cover"
        />
      </div>

      <div className="flex flex-1 items-center justify-between">
        <span className="text-sm font-medium text-gray-900">
          {formatPrice(price)}
        </span>
        <button
          onClick={() => removeItem(photoId)}
          className="text-gray-400 hover:text-red-500"
        >
          <Trash2 className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}
