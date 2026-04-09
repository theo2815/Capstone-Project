"use client";

import { useCart } from "@/hooks/use-cart";
import { CartItem } from "@/components/cart/cart-item";
import { CartSummary } from "@/components/cart/cart-summary";
import { Button } from "@/components/ui/button";
import { ShoppingCart } from "lucide-react";
import Link from "next/link";
import { ROUTES } from "@/lib/constants";

export default function CartPage() {
  const { items, clear } = useCart();

  if (items.length === 0) {
    return (
      <div className="mx-auto max-w-7xl px-4 py-16 text-center sm:px-6 lg:px-8">
        <ShoppingCart className="mx-auto h-12 w-12 text-gray-300" />
        <h1 className="mt-4 text-2xl font-bold text-gray-900">
          Your cart is empty
        </h1>
        <p className="mt-2 text-gray-600">
          Find your race photos and add them to your cart.
        </p>
        <Link href={ROUTES.EVENTS} className="mt-6 inline-block">
          <Button>Browse Events</Button>
        </Link>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900">Shopping Cart</h1>
        <Button variant="ghost" size="sm" onClick={clear}>
          Clear All
        </Button>
      </div>

      <div className="mt-8 grid gap-8 lg:grid-cols-3">
        <div className="space-y-3 lg:col-span-2">
          {items.map((item) => (
            <CartItem
              key={item.photoId}
              photoId={item.photoId}
              thumbnailUrl={item.thumbnailUrl}
              price={item.price}
            />
          ))}
        </div>
        <CartSummary />
      </div>
    </div>
  );
}
