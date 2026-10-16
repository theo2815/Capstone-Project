export type PhotoMetadataAccess = "none" | "public" | "owner";
export type PhotoMetadataStatus = "pending" | "ready" | "empty" | "failed" | "unavailable";

export interface PhotoMetadataFields {
  cameraMake: string | null;
  cameraModel: string | null;
  lens: string | null;
  aperture: number | null;
  exposureTime: string | null;
  iso: number | null;
  author: string | null;
  copyright: string | null;
  takenAtLocal: string | null;
  takenAtOffset: string | null;
  flash: string | null;
  exposureMode: string | null;
  whiteBalance: string | null;
}

export interface PhotoMetadataDto {
  photoId: string;
  status: PhotoMetadataStatus;
  originalFilename: string | null;
  fileSizeBytes: number | null;
  fields: PhotoMetadataFields | null;
}
