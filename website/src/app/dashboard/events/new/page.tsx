"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useQueryClient } from "@tanstack/react-query";
import { MyEventForm } from "@/components/dashboard/my-event-form";
import { SiteHeader } from "@/components/layout/site-header";
import { Kicker } from "@/components/ui/kicker";
import { useToast } from "@/hooks/use-toast";
import { ROUTES } from "@/lib/constants";

// Create a photographer-owned event (V46). Focused page (no dashboard rail —
// the /dashboard/events/[^/]+ pattern in app/dashboard/layout.tsx escapes
// it, same as the share page). Submit → pending admin review → back to the
// events list, where the card carries a "Pending review" chip.

export default function NewEventPage() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const { showToast } = useToast();

  return (
    <main className="bg-bone text-ink min-h-screen flex flex-col">
      <SiteHeader />
      <div className="flex-1 max-w-3xl w-full mx-auto px-6 md:px-10 pt-8 md:pt-12 pb-16 md:pb-24">
        <Kicker
          as={Link}
          href={ROUTES.DASHBOARD_EVENTS}
          className="inline-flex items-center gap-2 hover:text-ink transition-colors mb-8 md:mb-10"
        >
          <span aria-hidden="true">←</span>
          <span>Back to events</span>
        </Kicker>
        <section className="mb-10">
          <Kicker as="p">Your event</Kicker>
          <h1 className="font-display text-4xl md:text-6xl font-extrabold tracking-tight text-ink mt-4 leading-[1.05]">
            Create an event.
          </h1>
          <p className="font-sans text-base md:text-lg text-ink-soft mt-4 max-w-md">
            A race you organise or shoot on your own. Choose paid or free,
            public or link-only; an admin approves it before it goes live.
          </p>
        </section>
        <MyEventForm
          onDone={() => {
            void queryClient.invalidateQueries({
              queryKey: ["photographer", "events"],
            });
            showToast({
              kind: "success",
              message: "Sent for review. Uploads open once an admin approves it.",
            });
            router.push(ROUTES.DASHBOARD_EVENTS);
          }}
          onCancel={() => router.push(ROUTES.DASHBOARD_EVENTS)}
        />
      </div>
    </main>
  );
}
