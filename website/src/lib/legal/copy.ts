import { ROUTES, SUPPORT_EMAIL } from "@/lib/constants";

// Short UI texts from the PT-0a copy deck. Links use the same
// [label](href) and **bold** marks as the documents (rendered by LegalText).
export const DELETION_WINDOW = "7 business days";
export const DELETION_MAIL_SUBJECT = "Account deletion request";
// 2 since 2026-11-05: the notice now names Singapore and covers a child's selfie (review S10).
export const FACE_NOTICE_VERSION = 2;
// The bib-search notice (task BT-1, review S10); raising it asks everyone again.
export const BIB_NOTICE_VERSION = 1;
// RD-11: the RD-15 draft wording behind a placeholder; the production build
// refuses it until a later legal version signs these four texts (AA-0.7 O1).
const RD15_PENDING = "[ASK OWNER: legal 1.3 wording from RD-15]";
const RUNNER_DELETION_MAIL_SUBJECT = `${RD15_PENDING} Photo deletion request`;

export const LEGAL_COPY = {
  signupTickLabel: `I am 18 or older, I agree to the [Terms of Service](${ROUTES.TERMS}) and I have read the [Privacy Policy](${ROUTES.PRIVACY}).`,
  signupTickError: "Please tick the box to agree before you continue.",
  googleLine: `By continuing with Google, you agree to the [Terms of Service](${ROUTES.TERMS}) and [Privacy Policy](${ROUTES.PRIVACY}). You'll confirm this on the next step.`,
  faceNotice: `To find your photos, we send this selfie to Amazon Rekognition (run by AWS in Singapore) to compare the face in it with faces in the photos of the event you search. If you only search, we don't keep the selfie; selfies you save to your profile stay until you delete them. [How face matching works](${ROUTES.PRIVACY}#face-matching)`,
  faceTickLabel: "I agree to face matching with this selfie. If it shows my child, I am their parent or guardian.",
  bibNotice: `To find your photos, we look for this bib number in this event's photos. Search only for your own photos, your child's, or those of someone who agreed. [How search works](${ROUTES.PRIVACY}#face-matching)`,
  bibTickLabel: "I agree. This is my bib number, my child's, or I have the runner's permission.",
  checkoutLine: `By paying, you agree to the [Terms of Service (including refunds)](${ROUTES.TERMS}#refunds).`,
  // The live-selfie step (AA-I14), adviser-signed legal 1.3 text (AA-I21).
  identityNotice: `We use a short live video selfie to check that you are a real person and that you match your ID. Amazon Web Services processes it in Malaysia and Singapore for us. We keep one still image from it, privately, with your ID photos. It is never used for photo search. [How seller checks work](${ROUTES.PRIVACY}#seller-checks)`,
  identityTickLabel: "I agree to the live selfie check and to QuickPitik keeping one still image from it with my ID photos.",
  identityAdminInstead: "Prefer not to? Have an admin check your ID instead. This takes longer.",
  runnerDeletionNotice: `${RD15_PENDING} To find photos of you, we run a short live check with your camera (Amazon Rekognition Face Liveness, run by AWS in Malaysia) and use one still image from it to search only this event's photos (AWS in Singapore). We keep that image with your request and delete it when the request closes. [How photo deletion works](${ROUTES.PRIVACY}#face-matching)`,
  runnerDeletionTickLabel: `${RD15_PENDING} I agree to the live check and to this face search. The photos I'm asking about are of me.`,
  runnerDeletionEmailInstead: `${RD15_PENDING} Rather not use the camera, or asking for someone else? Email [${SUPPORT_EMAIL}](mailto:${SUPPORT_EMAIL}?subject=${encodeURIComponent(RUNNER_DELETION_MAIL_SUBJECT)}) instead.`,
  runnerDeletionMailSubject: RUNNER_DELETION_MAIL_SUBJECT,
  deletionText: `To delete your account, email [${SUPPORT_EMAIL}](mailto:${SUPPORT_EMAIL}?subject=${encodeURIComponent(DELETION_MAIL_SUBJECT)}) from your account's email address. We'll do it within ${DELETION_WINDOW}. Some purchase records are kept, as the [Privacy Policy](${ROUTES.PRIVACY}#deletion) explains.`,
} as const;
