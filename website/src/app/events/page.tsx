import type { Metadata } from "next";
import { SiteHeader } from "@/components/layout/site-header";
import { EventsBrowser, type ListEvent } from "./events-browser";

export const metadata: Metadata = {
  title: "Events | QuickPitik",
  description:
    "Browse marathon and running events in Cebu. Pick your race, find your photos.",
};

const MOCK_EVENTS: ReadonlyArray<ListEvent> = [
  {
    id: "u1",
    slug: "cebu-bay-run-2026",
    name: "Cebu Bay Run",
    date: "2026-05-09",
    location: "Mactan Channel Bridge",
    city: "Mactan",
    photoCount: 0,
    participantCount: 1500,
    status: "ACTIVE",
    state: "upcoming",
  },
  {
    id: "u2",
    slug: "it-park-sunrise-5k-2026",
    name: "IT Park Sunrise 5K",
    date: "2026-05-17",
    location: "IT Park, Cebu City",
    city: "Cebu City",
    photoCount: 0,
    participantCount: 800,
    status: "ACTIVE",
    state: "upcoming",
  },
  {
    id: "u3",
    slug: "cordova-foundation-run-2026",
    name: "Cordova Foundation Run",
    date: "2026-06-07",
    location: "Cordova, Cebu",
    city: "Cordova",
    photoCount: 0,
    participantCount: 600,
    status: "ACTIVE",
    state: "upcoming",
  },
  {
    id: "1",
    slug: "cebu-marathon-2026",
    name: "Cebu Marathon 2026",
    date: "2026-04-28",
    location: "SRP Boulevard, Cebu City",
    city: "Cebu City",
    photoCount: 1240,
    participantCount: 4800,
    status: "ACTIVE",
    state: "live",
  },
  {
    id: "2",
    slug: "mactan-sunset-run-2026",
    name: "Mactan Sunset Run",
    date: "2026-04-26",
    location: "Mactan Channel Bridge",
    city: "Mactan",
    photoCount: 612,
    participantCount: 1800,
    status: "ACTIVE",
    state: "live",
  },
  {
    id: "3",
    slug: "srp-half-marathon-2026",
    name: "SRP Half-Marathon",
    date: "2026-04-12",
    location: "South Road Properties, Cebu",
    city: "Cebu City",
    photoCount: 3850,
    participantCount: 2400,
    status: "COMPLETED",
    state: "open",
  },
  {
    id: "4",
    slug: "sun-run-cebu-2026",
    name: "Sun Run Cebu",
    date: "2026-04-05",
    location: "IT Park, Cebu City",
    city: "Cebu City",
    photoCount: 2120,
    participantCount: 3100,
    status: "COMPLETED",
    state: "open",
  },
  {
    id: "5",
    slug: "mactan-coastal-5k-2026",
    name: "Mactan Coastal 5K",
    date: "2026-03-29",
    location: "Mactan, Lapu-Lapu City",
    city: "Lapu-Lapu",
    photoCount: 980,
    participantCount: 1200,
    status: "COMPLETED",
    state: "open",
  },
  {
    id: "6",
    slug: "cebu-night-run-2025",
    name: "Cebu City Night Run 2025",
    date: "2025-12-14",
    location: "Cebu Business Park",
    city: "Cebu City",
    photoCount: 4200,
    participantCount: 5600,
    status: "ARCHIVED",
    state: "past",
  },
  {
    id: "7",
    slug: "talisay-10k-2025",
    name: "Talisay 10K",
    date: "2025-11-09",
    location: "Talisay City, Cebu",
    city: "Talisay",
    photoCount: 1640,
    participantCount: 1900,
    status: "ARCHIVED",
    state: "past",
  },
  {
    id: "8",
    slug: "cordova-bayrun-2025",
    name: "Cordova Bay Run",
    date: "2025-10-19",
    location: "Cordova, Cebu",
    city: "Cordova",
    photoCount: 720,
    participantCount: 950,
    status: "ARCHIVED",
    state: "past",
  },
];

export default function EventsPage() {
  return (
    <main className="bg-bone text-ink relative">
      <SiteHeader />

      <EventsBrowser events={MOCK_EVENTS} />

      <Footer />
    </main>
  );
}

function Footer() {
  return (
    <footer className="px-6 md:px-10 py-8 flex flex-col md:flex-row items-center justify-between gap-4 border-t border-line bg-bone">
      <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
        QuickPitik &middot; Cebu, Philippines
      </p>
      <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
        © 2026
      </p>
    </footer>
  );
}
