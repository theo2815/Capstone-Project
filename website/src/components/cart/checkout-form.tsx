"use client";

import { useState, type FormEvent } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface CheckoutFormProps {
  onSubmit: (paymentMethod: string) => void;
  isProcessing?: boolean;
  total: number;
}

export function CheckoutForm({
  onSubmit,
  isProcessing,
  total,
}: CheckoutFormProps) {
  const [paymentMethod, setPaymentMethod] = useState("gcash");

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    onSubmit(paymentMethod);
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div>
        <h3 className="mb-3 text-lg font-semibold text-gray-900">
          Payment Method
        </h3>
        <div className="space-y-2">
          {[
            { value: "gcash", label: "GCash" },
            { value: "maya", label: "Maya" },
            { value: "card", label: "Credit / Debit Card" },
          ].map((method) => (
            <label
              key={method.value}
              className="flex cursor-pointer items-center gap-3 rounded-lg border border-gray-200 p-3 hover:bg-gray-50"
            >
              <input
                type="radio"
                name="payment"
                value={method.value}
                checked={paymentMethod === method.value}
                onChange={(e) => setPaymentMethod(e.target.value)}
                className="text-blue-600"
              />
              <span className="text-sm font-medium text-gray-700">
                {method.label}
              </span>
            </label>
          ))}
        </div>
      </div>

      {paymentMethod === "card" && (
        <div className="space-y-3">
          <Input id="cardNumber" label="Card Number" placeholder="4242 4242 4242 4242" />
          <div className="grid grid-cols-2 gap-3">
            <Input id="expiry" label="Expiry" placeholder="MM/YY" />
            <Input id="cvc" label="CVC" placeholder="123" />
          </div>
        </div>
      )}

      <Button type="submit" isLoading={isProcessing} className="w-full" size="lg">
        Pay PHP {total.toFixed(2)}
      </Button>
    </form>
  );
}
