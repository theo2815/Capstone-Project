export type EventStatus = "DRAFT" | "ACTIVE" | "COMPLETED" | "ARCHIVED";

export interface Event {
  id: string;
  slug: string;
  name: string;
  date: string;
  location: string;
  bannerUrl?: string;
  photoCount: number;
  participantCount: number;
  status: EventStatus;
}

export interface EventDetail extends Event {
  description: string;
  organizerName: string;
  categories: string[];
  pricePerPhoto: number;
  bundlePrice?: number;
  bundleSize?: number;
}

export interface EventFilter {
  search?: string;
  status?: EventStatus;
  dateFrom?: string;
  dateTo?: string;
  page?: number;
  pageSize?: number;
}
