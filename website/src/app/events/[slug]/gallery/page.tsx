import type { Metadata } from "next";

interface GalleryPageProps {
  params: Promise<{ slug: string }>;
}

export async function generateMetadata({
  params,
}: GalleryPageProps): Promise<Metadata> {
  const { slug } = await params;
  return {
    title: `Gallery - ${slug} | EventAI`,
  };
}

export default async function GalleryPage({ params }: GalleryPageProps) {
  const { slug } = await params;

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      <h1 className="text-2xl font-bold text-gray-900">Photo Gallery</h1>
      <p className="mt-1 text-sm text-gray-500">Event: {slug}</p>
      {/* PhotoGrid with infinite scroll will be implemented here */}
    </div>
  );
}
