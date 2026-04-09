import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Events | EventAI",
  description: "Browse marathon and running events with AI-powered photo search.",
};

export default function EventsPage() {
  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      <h1 className="text-3xl font-bold text-gray-900">Events</h1>
      <p className="mt-2 text-gray-600">
        Find your race photos from recent events.
      </p>
      {/* EventGrid with data fetching will be implemented here */}
    </div>
  );
}
