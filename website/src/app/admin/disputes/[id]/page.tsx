import { redirect } from "next/navigation";
import { ROUTES } from "@/lib/constants";

// One detail view: the queue drawer (?row=) fetches any dispute by id and holds the admin actions.
export default async function AdminDisputeDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  redirect(`${ROUTES.ADMIN_DISPUTES}?row=${encodeURIComponent(id)}`);
}
