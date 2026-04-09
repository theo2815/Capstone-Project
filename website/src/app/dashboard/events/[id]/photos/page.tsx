"use client";

import { useParams } from "next/navigation";

export default function DashboardPhotosPage() {
  const { id } = useParams<{ id: string }>();

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900">Manage Photos</h1>
      <p className="mt-1 text-sm text-gray-500">Event ID: {id}</p>
      {/* Photo management grid with delete/tag actions will go here */}
    </div>
  );
}
