"use client";

import Link from "next/link";
import { Kicker } from "@/components/ui/kicker";
import { BTN_PRIMARY, BTN_SIZE } from "@/components/ui/button-styles";
import { RemovalRequestList } from "@/components/admin/photo-removal/request-list";
import { ROUTES } from "@/lib/constants";
import { cn } from "@/lib/utils";

export default function AdminPhotoRemovalRequestsPage() {
  return (
    <>
      <header className="pb-8 md:pb-12 border-b border-line">
        <Kicker as="p">Photo removals</Kicker>
        <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-[1.05] text-ink mt-3">Removal requests.</h1>
        <p className="font-sans text-sm md:text-base text-ink-soft mt-3 max-w-xl">
          Runners who asked support to take their photos down. Verify which photos are theirs, then each photographer reviews their own.
        </p>
        <Link href={`${ROUTES.ADMIN_PHOTO_REMOVAL_REQUESTS}/new`} className={cn(BTN_PRIMARY, BTN_SIZE.sm, "mt-6")}>New request</Link>
      </header>
      <RemovalRequestList />
    </>
  );
}
