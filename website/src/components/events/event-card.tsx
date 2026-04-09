import Link from "next/link";
import Image from "next/image";
import { Badge } from "@/components/ui/badge";
import { formatDate } from "@/lib/utils";
import { Calendar, MapPin, Camera } from "lucide-react";
import type { Event } from "@/types/event";

interface EventCardProps {
  event: Event;
}

export function EventCard({ event }: EventCardProps) {
  const statusVariant = {
    DRAFT: "default" as const,
    ACTIVE: "success" as const,
    COMPLETED: "info" as const,
    ARCHIVED: "warning" as const,
  };

  return (
    <Link href={`/events/${event.slug}`} className="group block">
      <div className="overflow-hidden rounded-lg border border-gray-200 bg-white shadow-sm transition-shadow group-hover:shadow-md">
        <div className="relative aspect-[16/9] bg-gray-100">
          {event.bannerUrl ? (
            <Image
              src={event.bannerUrl}
              alt={event.name}
              fill
              className="object-cover"
            />
          ) : (
            <div className="flex h-full items-center justify-center">
              <Camera className="h-10 w-10 text-gray-300" />
            </div>
          )}
          <Badge
            variant={statusVariant[event.status]}
            className="absolute right-2 top-2"
          >
            {event.status}
          </Badge>
        </div>

        <div className="p-4">
          <h3 className="text-lg font-semibold text-gray-900 group-hover:text-blue-600">
            {event.name}
          </h3>
          <div className="mt-2 flex flex-col gap-1 text-sm text-gray-600">
            <span className="flex items-center gap-1.5">
              <Calendar className="h-3.5 w-3.5" />
              {formatDate(event.date)}
            </span>
            <span className="flex items-center gap-1.5">
              <MapPin className="h-3.5 w-3.5" />
              {event.location}
            </span>
          </div>
          <p className="mt-2 text-sm text-gray-500">
            {event.photoCount.toLocaleString()} photos
          </p>
        </div>
      </div>
    </Link>
  );
}
