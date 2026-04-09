"use client";

import { useCart } from "@/hooks/use-cart";
import { Button } from "@/components/ui/button";
import { formatPrice } from "@/lib/utils";
import Link from "next/link";
import { ROUTES } from "@/lib/constants";

export function CartSummary() {
  const { items, total } = useCart();

  if (items.length === 0) return null;

  return (
    <div className="rounded-lg border border-gray-200 bg-gray-50 p-6">
      <h3 className="text-lg font-semibold text-gray-900">Order Summary</h3>

      <div className="mt-4 space-y-2">
        <div className="flex justify-between text-sm text-gray-600">
          <span>
            {items.length} {items.length === 1 ? "photo" : "photos"}
          </span>
          <span>{formatPrice(total)}</span>
        </div>
        <div className="border-t border-gray-200 pt-2">
          <div className="flex justify-between text-base font-semibold text-gray-900">
            <span>Total</span>
            <span>{formatPrice(total)}</span>
          </div>
        </div>
      </div>

      <Link href={ROUTES.CHECKOUT} className="mt-4 block">
        <Button className="w-full">Proceed to Checkout</Button>
      </Link>
    </div>
  );
}
