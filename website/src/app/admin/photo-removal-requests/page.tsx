"use client";

import Link from "next/link";
import { Kicker } from "@/components/ui/kicker";
import { BTN_PRIMARY, BTN_SIZE } from "@/components/ui/button-styles";
import { RemovalRequestList } from "@/components/admin/photo-removal/request-list";
import { ROUTES } from "@/lib/constants";
import { cn } from "@/lib/utils";
import { AdminSectionIntro } from "@/components/admin/admin-section-intro";
import { useAdminQueueCounts } from "@/lib/admin-queues";

export default function AdminPhotoRemovalRequestsPage() {
  const waiting = useAdminQueueCounts().byKey.removals;
  return (
    <>
      <header className="pb-8 md:pb-12 border-b border-line">
        <Kicker as="p">Photo removals{waiting !== null && <> · <span className="tnum">{waiting}</span> waiting</>}</Kicker>
        <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-[1.05] text-ink mt-3">Photo removals.</h1>
        <AdminSectionIntro href={ROUTES.ADMIN_PHOTO_REMOVAL_REQUESTS} />
        <Link href={`${ROUTES.ADMIN_PHOTO_REMOVAL_REQUESTS}/new`} className={cn(BTN_PRIMARY, BTN_SIZE.sm, "mt-6")}>New request</Link>
      </header>
      <RemovalRequestList />
    </>
  );
}
