"use client";

import { Kicker } from "@/components/ui/kicker";
import { PhotographerRemovalRequestList } from "@/components/dashboard/photo-removal-request";

export default function PhotoRemovalRequestsPage() {
  return (
    <>
      <header className="pb-8 md:pb-12 border-b border-line">
        <Kicker as="p">Photo removals</Kicker>
        <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-[1.05] text-ink mt-3">Removal requests.</h1>
        <p className="font-sans text-sm md:text-base text-ink-soft mt-3 max-w-xl">
          Runners who asked QuickPitik to take down photos of themselves. Review the photos of yours that are involved and respond.
        </p>
      </header>
      <PhotographerRemovalRequestList />
    </>
  );
}
