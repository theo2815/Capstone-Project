import type { Metadata } from "next";
import { Suspense } from "react";
import { SiteHeader } from "@/components/layout/site-header";
import { EVENT_CATALOG } from "@/lib/event-catalog";
import { EventsBrowser } from "./events-browser";

export const metadata: Metadata = {
  title: "Events | QuickPitik",
  description:
    "Browse marathon and running events in Cebu. Pick your race, find your photos.",
};

export default function EventsPage() {
  return (
    <main className="bg-bone text-ink relative">
      <SiteHeader />

      <Suspense fallback={null}>
        <EventsBrowser events={EVENT_CATALOG} />
      </Suspense>

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
