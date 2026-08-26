import { api } from "@/lib/api";

export interface PhotoAlertStatus {
  registered: boolean;
  selfieId: string | null;
}

// Runner opt-in for the "your photos are ready" email. The backend matches the
// runner's selfie against the event during its date-based sweep and emails once
// when photos of them appear. selfieId is optional — omitting it registers with
// the runner's primary (or most recent) selfie.
export async function fetchPhotoAlertStatus(
  slug: string,
): Promise<PhotoAlertStatus> {
  return api.get<PhotoAlertStatus>(
    `/events/${encodeURIComponent(slug)}/photo-alert`,
  );
}

export async function registerPhotoAlert(
  slug: string,
  selfieId?: string,
): Promise<PhotoAlertStatus> {
  return api.post<PhotoAlertStatus>(
    `/events/${encodeURIComponent(slug)}/photo-alert`,
    selfieId ? { selfieId } : {},
  );
}

export async function unregisterPhotoAlert(
  slug: string,
): Promise<{ removed: boolean }> {
  return api.delete<{ removed: boolean }>(
    `/events/${encodeURIComponent(slug)}/photo-alert`,
  );
}
