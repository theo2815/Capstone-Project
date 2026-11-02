import { ROUTES, SUPPORT_EMAIL } from "@/lib/constants";

// Short UI texts from the PT-0a copy deck. Links use the same
// [label](href) and **bold** marks as the documents (rendered by LegalText).
export const DELETION_WINDOW = "7 business days";
export const DELETION_MAIL_SUBJECT = "Account deletion request";
export const FACE_NOTICE_VERSION = 1;

export const LEGAL_COPY = {
  signupTickLabel: `I am 18 or older, I agree to the [Terms of Service](${ROUTES.TERMS}) and I have read the [Privacy Policy](${ROUTES.PRIVACY}).`,
  signupTickError: "Please tick the box to agree before you continue.",
  googleLine: `By continuing with Google, you agree to the [Terms of Service](${ROUTES.TERMS}) and [Privacy Policy](${ROUTES.PRIVACY}). You'll confirm this on the next step.`,
  faceNotice: `To find your photos, we send your selfie to Amazon Rekognition to compare it with faces in this event's photos. If you only search, we don't keep your selfie; selfies you save to your profile stay until you delete them. [How face matching works](${ROUTES.PRIVACY}#face-matching)`,
  faceTickLabel: "I agree to face matching with my selfie.",
  checkoutLine: `By paying, you agree to the [Terms of Service (including refunds)](${ROUTES.TERMS}#refunds).`,
  deletionText: `To delete your account, email [${SUPPORT_EMAIL}](mailto:${SUPPORT_EMAIL}?subject=${encodeURIComponent(DELETION_MAIL_SUBJECT)}) from your account's email address. We'll do it within ${DELETION_WINDOW}. Some purchase records are kept, as the [Privacy Policy](${ROUTES.PRIVACY}#deletion) explains.`,
} as const;
