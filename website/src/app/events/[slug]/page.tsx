import type { Metadata } from "next";

interface EventDetailPageProps {
  params: Promise<{ slug: string }>;
}

export async function generateMetadata({
  params,
}: EventDetailPageProps): Promise<Metadata> {
  const { slug } = await params;
  return {
    title: `${slug} | EventAI`,
  };
}

export default async function EventDetailPage({
  params,
}: EventDetailPageProps) {
  const { slug } = await params;

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      <p className="text-sm text-gray-500">Event: {slug}</p>
      {/* EventHero + gallery preview + search CTA will be implemented here */}
    </div>
  );
}
