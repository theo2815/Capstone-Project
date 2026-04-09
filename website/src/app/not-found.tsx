import Link from "next/link";
import { Button } from "@/components/ui/button";
import { ROUTES } from "@/lib/constants";

export default function NotFound() {
  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center px-4">
      <h1 className="text-6xl font-bold text-gray-200">404</h1>
      <h2 className="mt-4 text-xl font-semibold text-gray-900">
        Page Not Found
      </h2>
      <p className="mt-2 text-gray-600">
        The page you&apos;re looking for doesn&apos;t exist.
      </p>
      <Link href={ROUTES.HOME} className="mt-6">
        <Button>Back to Home</Button>
      </Link>
    </div>
  );
}
