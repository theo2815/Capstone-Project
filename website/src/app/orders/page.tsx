"use client";

import { ProtectedRoute } from "@/components/auth/protected-route";

export default function OrdersPage() {
  return (
    <ProtectedRoute>
      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <h1 className="text-2xl font-bold text-gray-900">My Orders</h1>
        <p className="mt-1 text-gray-600">Your purchase history.</p>
        {/* Orders table with status badges will be implemented here */}
      </div>
    </ProtectedRoute>
  );
}
