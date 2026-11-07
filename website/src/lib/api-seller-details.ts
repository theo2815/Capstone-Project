import { api } from "@/lib/api";

// Seller details (SC-1 backend, plan § 3.4): required before new uploads.
// E1 GET / E2 POST /api/v1/me/photographer/seller-details. No links or storage
// keys come back; only admins open the files.

export type SellerDetailsStatus = "missing" | "submitted" | "approved" | "rejected";

export const SELLER_ID_TYPES = [
  { value: "philsys", label: "PhilSys National ID (card or ePhilID)" },
  { value: "drivers_license", label: "Driver's licence" },
  { value: "passport", label: "Passport" },
  { value: "umid", label: "UMID" },
  { value: "prc", label: "PRC ID" },
  { value: "postal", label: "Postal ID" },
  { value: "voters", label: "Voter's ID" },
] as const;

export type SellerIdType = (typeof SELLER_ID_TYPES)[number]["value"];

export interface SellerDetails {
  status: SellerDetailsStatus;
  legalName?: string | null;
  idType?: SellerIdType | null;
  address?: string | null;
  phone?: string | null;
  hasIdBack?: boolean | null;
  hasBirForm?: boolean | null;
  rejectionReason?: string | null;
  submittedAt?: string | null;
  reviewedAt?: string | null;
}

export interface SellerDetailsInput {
  legalName: string;
  idType: SellerIdType;
  address: string;
  phone: string;
  idFront: File;
  idBack?: File | null;
  selfie: File;
  birForm?: File | null;
}

// Mirrors SellerDetailsService on the server; the server still decides.
export const SELLER_FILE_MAX_BYTES = 5 * 1024 * 1024;
export const SELLER_FILE_TYPES = ["image/jpeg", "image/png", "image/webp"];

export function isValidPhone(raw: string): boolean {
  const typed = raw.trim();
  const compact = typed.replace(/[\s.()-]/g, "");
  const digits = compact.replace(/\D/g, "").length;
  return typed.length <= 30 && /^\+?\d+$/.test(compact)
    && (compact.startsWith("+63") || compact.startsWith("0")) && digits >= 10 && digits <= 13;
}

export function getSellerDetails(signal?: AbortSignal): Promise<SellerDetails> {
  return api.get<SellerDetails>("/me/photographer/seller-details", { signal });
}

export function submitSellerDetails(input: SellerDetailsInput): Promise<SellerDetails> {
  const fd = new FormData();
  fd.append("legalName", input.legalName);
  fd.append("idType", input.idType);
  fd.append("address", input.address);
  fd.append("phone", input.phone);
  fd.append("idFront", input.idFront);
  if (input.idBack) fd.append("idBack", input.idBack);
  fd.append("selfie", input.selfie);
  if (input.birForm) fd.append("birForm", input.birForm);
  return api.post<SellerDetails>("/me/photographer/seller-details", fd);
}
