"use client";

import { useParams } from "next/navigation";
import { ProtectedRoute } from "@/components/auth/protected-route";

export default function OrderDetailPage() {
  const { id } = useParams<{ id: string }>();

  return (
    <ProtectedRoute>
      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <h1 className="text-2xl font-bold text-gray-900">Order Details</h1>
        <p className="mt-1 text-sm text-gray-500">Order ID: {id}</p>
        {/* Order items with download links will be implemented here */}
      </div>
    </ProtectedRoute>
  );
}
