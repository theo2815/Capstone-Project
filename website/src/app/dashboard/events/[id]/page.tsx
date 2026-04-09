"use client";

import { useParams } from "next/navigation";

export default function DashboardEventDetailPage() {
  const { id } = useParams<{ id: string }>();

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900">Event Management</h1>
      <p className="mt-1 text-sm text-gray-500">Event ID: {id}</p>
      {/* Event stats, photo overview, quick actions will go here */}
    </div>
  );
}
