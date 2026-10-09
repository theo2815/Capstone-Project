"use client";

import Link from "next/link";
import { Kicker } from "@/components/ui/kicker";
import { RemovalRequestIntake } from "@/components/admin/photo-removal/request-intake";
import { ROUTES } from "@/lib/constants";

export default function AdminNewPhotoRemovalRequestPage() {
  return (
    <>
      <Link href={ROUTES.ADMIN_PHOTO_REMOVAL_REQUESTS} className="text-sm text-slate hover:text-ink">← Removal requests</Link>
      <header className="mt-6 pb-8 border-b border-line">
        <Kicker as="p">Photo removals · new</Kicker>
        <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-[1.05] text-ink mt-3">New removal request.</h1>
      </header>
      <RemovalRequestIntake />
    </>
  );
}
