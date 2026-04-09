"use client";

import { ProtectedRoute } from "@/components/auth/protected-route";

export default function ProfilePage() {
  return (
    <ProtectedRoute>
      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        <h1 className="text-2xl font-bold text-gray-900">My Profile</h1>
        <p className="mt-1 text-gray-600">Manage your account settings.</p>
        {/* Profile form (name, email, password change) will go here */}
      </div>
    </ProtectedRoute>
  );
}
