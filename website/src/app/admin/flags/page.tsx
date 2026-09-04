"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  FlagsQueue,
  useOpenFlagsCount,
  useHiddenFlagsCount,
} from "@/components/admin/flags-queue";
import { Kicker } from "@/components/ui/kicker";
import { ADMIN_FLAGS_ENABLED, ROUTES } from "@/lib/constants";

// Phase 2 admin redesign — /admin/flags is the focus-mode route for the
// flags queue, sharing its body (search + four slabs) with /admin/inbox
// via the lifted <FlagsQueue> component.
//
// Hidden in v1 — when ADMIN_FLAGS_ENABLED is false this page redirects
// stale bookmarks to the inbox instead of 404-ing. The queue body and
// store stay intact so flipping the constant fully revives the surface.

export default function AdminFlagsPage() {
  const router = useRouter();
  const openCount = useOpenFlagsCount();
  const hiddenCount = useHiddenFlagsCount();

  useEffect(() => {
    if (!ADMIN_FLAGS_ENABLED) {
      router.replace(ROUTES.ADMIN_INBOX);
    }
  }, [router]);

  if (!ADMIN_FLAGS_ENABLED) {
    return null;
  }

  return (
    <>
      <Header openCount={openCount} hiddenCount={hiddenCount} />
      <FlagsQueue />
    </>
  );
}

function Header({
  openCount,
  hiddenCount,
}: {
  openCount: number;
  hiddenCount: number;
}) {
  return (
    <header className="pb-8 md:pb-12 border-b border-line">
      <Kicker as="p">
        Flags · <span className="tnum">{openCount}</span> open ·{" "}
        <span className="tnum">{hiddenCount}</span> hidden
      </Kicker>
      <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-[1.05] text-ink mt-3">
        Flags.
      </h1>
      <p className="font-sans text-sm md:text-base text-ink-soft mt-3 max-w-xl">
        Photos reported by runners or surfaced during admin review. Hide
        what shouldn&apos;t be live.
      </p>
    </header>
  );
}
