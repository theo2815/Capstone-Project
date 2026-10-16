"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { PhotographerPhotoRemovalRequest } from "@/components/dashboard/photo-removal-request";
import { ROUTES } from "@/lib/constants";

export default function PhotoRemovalRequestPage() {
  const { id } = useParams<{ id: string }>();
  return (
    <>
      <Link href={ROUTES.DASHBOARD_PHOTO_REMOVAL_REQUESTS} className="text-sm text-slate hover:text-ink">← Removal requests</Link>
      <div className="mt-6">
        <PhotographerPhotoRemovalRequest requestId={id} />
      </div>
    </>
  );
}
