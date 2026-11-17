import { SUPPORT_EMAIL } from "@/lib/constants";

// A block is a paragraph, a list (numbered when `start` is set), a table whose
// first row is the header, or the refund-policy marker. Text may use
// **bold** and [label](href).
export type LegalBlock =
  | string
  | { list: readonly string[]; start?: number }
  | { table: readonly (readonly string[])[] };

export interface LegalSection {
  id: string;
  heading: string;
  blocks: readonly LegalBlock[];
}

export interface LegalDoc {
  title: string;
  sections: readonly LegalSection[];
}

// Where the terms render lib/refund-policy.ts (one copy of the refund rules).
export const REFUND_POLICY_MARKER = "[[REFUND-POLICY]]";

export const EMAIL_LINK = `[${SUPPORT_EMAIL}](mailto:${SUPPORT_EMAIL})`;
export const OPERATOR = "QuickPitik Inc., Tuyan, Naga City, Cebu, Philippines";
