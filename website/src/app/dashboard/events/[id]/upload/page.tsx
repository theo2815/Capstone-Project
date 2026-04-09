"use client";

import { useParams } from "next/navigation";

export default function UploadPage() {
  const { id } = useParams<{ id: string }>();

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900">Upload Photos</h1>
      <p className="mt-1 text-sm text-gray-500">Event ID: {id}</p>
      {/* PhotoUploader component with upload progress will be connected here */}
    </div>
  );
}
