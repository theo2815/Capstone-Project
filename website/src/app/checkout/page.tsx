"use client";

import { useCart } from "@/hooks/use-cart";
import { CheckoutForm } from "@/components/cart/checkout-form";
import { CartSummary } from "@/components/cart/cart-summary";
import { ProtectedRoute } from "@/components/auth/protected-route";

export default function CheckoutPage() {
  const { total } = useCart();

  return (
    <ProtectedRoute>
      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <h1 className="text-2xl font-bold text-gray-900">Checkout</h1>

        <div className="mt-8 grid gap-8 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <CheckoutForm
              total={total}
              onSubmit={(method) => {
                // Payment integration will be connected here
                console.log("Payment method:", method);
              }}
            />
          </div>
          <CartSummary />
        </div>
      </div>
    </ProtectedRoute>
  );
}
