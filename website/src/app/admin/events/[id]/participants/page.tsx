"use client";

import { useParams } from "next/navigation";

export default function ParticipantsPage() {
  const { id } = useParams<{ id: string }>();

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900">Participants</h1>
      <p className="mt-1 text-sm text-gray-500">Event ID: {id}</p>
      {/* Participant table, CSV import, face enrollment will go here */}
    </div>
  );
}
