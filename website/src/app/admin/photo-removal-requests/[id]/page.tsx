"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { RemovalRequestReview } from "@/components/admin/photo-removal/request-review";
import { ROUTES } from "@/lib/constants";

export default function AdminPhotoRemovalRequestPage() {
  const { id } = useParams<{ id: string }>();
  return (
    <>
      <Link href={ROUTES.ADMIN_PHOTO_REMOVAL_REQUESTS} className="text-sm text-slate hover:text-ink">← Removal requests</Link>
      <div className="mt-6"><RemovalRequestReview requestId={id} /></div>
    </>
  );
}
