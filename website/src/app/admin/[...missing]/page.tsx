import { notFound } from "next/navigation";

// FT-4: an admin URL that matches no route shows app/admin/not-found.tsx inside the rail, not the public 404.
export default function AdminMissingPage() {
  notFound();
}
