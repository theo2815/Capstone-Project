"use client";

import { useParams } from "next/navigation";

export default function EditEventPage() {
  const { id } = useParams<{ id: string }>();

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900">Edit Event</h1>
      <p className="mt-1 text-sm text-gray-500">Event ID: {id}</p>
      {/* Event edit form will be implemented here */}
    </div>
  );
}
