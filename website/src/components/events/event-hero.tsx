import Image from "next/image";
import { Badge } from "@/components/ui/badge";
import { formatDate } from "@/lib/utils";
import { Calendar, MapPin, Users, Camera } from "lucide-react";
import type { EventDetail } from "@/types/event";

interface EventHeroProps {
  event: EventDetail;
}

export function EventHero({ event }: EventHeroProps) {
  return (
    <div className="relative overflow-hidden rounded-xl bg-gray-900">
      {event.bannerUrl && (
        <Image
          src={event.bannerUrl}
          alt={event.name}
          fill
          className="object-cover opacity-40"
          priority
        />
      )}
      <div className="relative px-6 py-12 sm:px-12 sm:py-16">
        <Badge variant="info" className="mb-4">
          {event.status}
        </Badge>
        <h1 className="text-3xl font-bold text-white sm:text-4xl">
          {event.name}
        </h1>
        <p className="mt-2 max-w-2xl text-gray-300">{event.description}</p>
        <div className="mt-6 flex flex-wrap gap-6 text-sm text-gray-300">
          <span className="flex items-center gap-1.5">
            <Calendar className="h-4 w-4" />
            {formatDate(event.date)}
          </span>
          <span className="flex items-center gap-1.5">
            <MapPin className="h-4 w-4" />
            {event.location}
          </span>
          <span className="flex items-center gap-1.5">
            <Camera className="h-4 w-4" />
            {event.photoCount.toLocaleString()} photos
          </span>
          <span className="flex items-center gap-1.5">
            <Users className="h-4 w-4" />
            {event.participantCount.toLocaleString()} participants
          </span>
        </div>
      </div>
    </div>
  );
}
