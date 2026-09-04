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
  // Per-photo price in PHP. Admin sets it at create time and overrides via
  // PATCH /admin/events/{id}; on change the BE re-prices every photo under
  // the event. Optional here because non-admin list endpoints historically
  // omitted it — keep the field opt-in so older mock data doesn't break.
  pricePerPhoto?: number;
  // Organizer name + race-day notes for the "About this race" strip. Optional
  // on base Event because only the detail + admin-list endpoints populate them
  // (the public /events list omits them). Admin can set both at create/edit.
  organizerName?: string;
  description?: string;
  // Photographer-owned events (V46). Admin events are public + paid; the
  // public list only ever carries public ones, so a missing value reads as
  // the default. Wire-lowercase.
  visibility?: EventVisibility;
  pricingMode?: EventPricingMode;
}

export type EventVisibility = "public" | "unlisted";
export type EventPricingMode = "paid" | "free";
export type WatermarkPolicy = "platform" | "own" | "none";

export interface EventDetail extends Event {
  description: string;
  organizerName: string;
  categories: string[];
  pricePerPhoto: number;
  bundlePrice?: number;
  bundleSize?: number;
  watermarkPolicy?: WatermarkPolicy;
  // Owner's public handle for a photographer-owned event ("Free · courtesy
  // of @handle"); null for admin events or an owner without a handle.
  photographerHandle?: string | null;
}

export interface EventFilter {
  search?: string;
  status?: EventStatus;
  dateFrom?: string;
  dateTo?: string;
  page?: number;
  pageSize?: number;
}
