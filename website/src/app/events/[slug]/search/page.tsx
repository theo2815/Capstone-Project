"use client";

import { useParams } from "next/navigation";

export default function SearchPage() {
  const params = useParams<{ slug: string }>();

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      <h1 className="text-2xl font-bold text-gray-900">Find My Photos</h1>
      <p className="mt-1 text-sm text-gray-500">Event: {params.slug}</p>

      <div className="mt-8 grid gap-8 lg:grid-cols-2">
        {/* FaceSearchUpload and BibSearchInput will be connected here */}
        <div>
          <p className="text-gray-500">Face search coming soon...</p>
        </div>
        <div>
          <p className="text-gray-500">Bib search coming soon...</p>
        </div>
      </div>

      {/* SearchResults will be rendered below */}
    </div>
  );
}
