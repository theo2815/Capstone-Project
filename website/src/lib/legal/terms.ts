import { EMAIL_LINK, OPERATOR, REFUND_POLICY_MARKER, type LegalDoc } from "./types";

// Draft text from the vault's PT-0a draft (2026-11-02). Not approved: PT-0b
// puts the final wording in. Section ids are linked from other pages and apps.
export const TERMS_OF_SERVICE: LegalDoc = {
  title: "Terms of Service",
  sections: [
    {
      id: "agreement",
      heading: "Agreement",
      blocks: [
        `These Terms are an agreement between you and ${OPERATOR} ("QuickPitik", "we"). They apply when you create an account, and when you use QuickPitik without one, for example to search for photos or to buy them as a guest. Our [Privacy Policy](/privacy) explains how we handle personal data.`,
        "If you do not agree, do not use QuickPitik.",
      ],
    },
    {
      id: "accounts",
      heading: "Accounts",
      blocks: [
        {
          list: [
            "You must be 18 or older to create an account or buy photos. A parent or guardian may search for and buy photos of their child.",
            "Give your real name and an email address you use, and verify the email.",
            "Keep your password to yourself. You are responsible for what happens in your account.",
            "If you sign in with Google and an account with that email already exists, the two are linked.",
            "After 5 wrong passwords, your account is locked for a short time.",
            "To delete your account, see the [Privacy Policy](/privacy#deletion).",
          ],
        },
      ],
    },
    {
      id: "roles",
      heading: "Roles",
      blocks: [
        {
          list: [
            "**Guests** can browse public events, search by bib or selfie, and buy photos with an email address.",
            '**Runners** have an account to save selfies and events, get "photos ready" alerts, and see their orders.',
            "**Photographers** upload and sell event photos. A runner account cannot become a photographer account; photographers sign up separately.",
            "**Admins** run QuickPitik: they approve photographers and photographer events, and handle reports, refunds, disputes, removals and payouts.",
          ],
        },
      ],
    },
    {
      id: "events-and-watermarks",
      heading: "Events and watermarks",
      blocks: [
        {
          list: [
            "Events created by QuickPitik are public and listed on the Events page.",
            "A photographer makes each of their events Public or Private. A Private event can be opened only by its link. A photographer's Public event is listed on the Events page only if an admin chooses to list it.",
            "Photographer events start as drafts and need an admin's approval.",
            "Previews carry a watermark: QuickPitik's mark, the photographer's own mark, or none, as set for the event. Paid events always carry QuickPitik's mark. Photos you buy are downloaded without the watermark.",
            "Some events are free: their photos can be downloaded without paying.",
          ],
        },
      ],
    },
    {
      id: "buying-photos",
      heading: "Buying photos",
      blocks: [
        {
          list: [
            "The photographer sets the price of each photo for paid events. Prices are in Philippine pesos.",
            "The only payment method is QR Ph through PayMongo: you scan a QR code with your bank or e-wallet app. We never see your bank or e-wallet login.",
            "After payment we email a receipt with download links to the email on the order. You can download bought photos for one year after payment.",
            "You get the original file, which may include the location where it was taken (see the [Privacy Policy](/privacy#photo-metadata)).",
            "Photographers may offer coupons.",
          ],
        },
      ],
    },
    {
      id: "buyer-licence",
      heading: "Buyer licence",
      blocks: [
        "When you buy a photo, the copyright stays with its owners (see [Copyright](#copyright)). You get a licence to:",
        {
          list: [
            "keep, print and use the photo for personal, non-commercial purposes; and",
            "post it on your own social media, with the photographer credited.",
          ],
        },
        "You may not sell or resell the photo, use it commercially (for example in advertising or on products for sale), or remove the photographer's credit or any watermark.",
      ],
    },
    {
      id: "refunds",
      heading: "Refunds",
      blocks: [
        "Our full refund policy is below. QuickPitik decides refund requests and disputes.",
        REFUND_POLICY_MARKER,
      ],
    },
    {
      id: "photographers",
      heading: "Photographers",
      blocks: [
        "**Verification and approval.** An admin verifies each photographer account. A photographer's public page is hidden until the account is verified, and while it is suspended. Photographer events need an admin's approval before they go live.",
        "**Fees.** QuickPitik keeps 25% of the price of each photo sold; the photographer gets the rest. The 25% is worked out on the full price.",
        "**Coupons.** Photographers can offer coupons, up to the limit shown in the app. A coupon comes out of the photographer's share only.",
        "**Payouts.**",
        {
          list: [
            "A sale becomes available to withdraw 7 days after it is made (the same as the refund window).",
            "You can request a payout when your available balance is at least ₱500, one request at a time.",
            "Payouts go to the GCash, Maya or GoTyme account and QR code you saved in the app.",
            "Refunded sales are taken out of your balance, and money from a disputed sale is held until the dispute is decided.",
          ],
        },
        "**Your promises.** By uploading, you promise that:",
        {
          list: [
            "you took the photos or hold the rights to sell them;",
            "you have the event organiser's permission where one is needed;",
            "you will follow QuickPitik's removal decisions; and",
            "you will follow the photographer rules in [Prohibited conduct](#prohibited-conduct) (rules 9 to 15).",
          ],
        },
      ],
    },
    {
      id: "copyright",
      heading: "Copyright",
      blocks: [
        {
          list: [
            "For photos a photographer uploads after agreeing to these Terms, the photographer transfers to QuickPitik an equal, undivided half share of the copyright, so **the photographer and QuickPitik share the copyright** in equal parts. The photographer's tick at sign-up is their written agreement to this.",
            "For photos uploaded before the photographer agreed to these Terms, the photographer keeps the whole copyright and gives QuickPitik a licence to store, watermark, show, face-match and sell them on QuickPitik.",
            "Either way, the photographer lets QuickPitik store, watermark, show, face-match and sell the photos on QuickPitik.",
            "The photographer stays the author. Their moral rights (to be named as the author and to object to changes that harm their reputation) stay with them, and QuickPitik credits them.",
          ],
        },
        "**How the two owners share a photo.** Instead of the Civil Code's default rules for co-owners, we agree that:",
        {
          list: [
            "the photographer may sell the photo, give non-exclusive licences for it and show it anywhere, including outside QuickPitik, and use it in their portfolio;",
            "QuickPitik may store, watermark, show, face-match and sell it on QuickPitik. QuickPitik may use it to promote QuickPitik, with the photographer credited, only if no one in it can be recognised or the people who can be recognised agreed;",
            "neither owner may, without the other's written agreement, give anyone exclusive rights to the photo, transfer its share, or change the photo itself (making previews, thumbnails and watermarked copies is not a change);",
            "for 10 years after upload, the longest such an agreement may last under the Civil Code, neither owner will ask for the shared copyright to be divided or for the photo to be sold to end the sharing;",
            "if someone uses a shared photo without permission, either owner may act against them and must tell the other.",
          ],
        },
        {
          list: [
            "Money from sales on QuickPitik is shared as set out in [Photographers](#photographers). Neither owner owes the other a share of money it earns from the photo outside QuickPitik.",
            'Previews carry this notice: "All rights reserved. Do not remove, alter or obscure the watermark or attribution."',
          ],
        },
      ],
    },
    {
      id: "prohibited-conduct",
      heading: "Prohibited conduct",
      blocks: [
        "**Everyone must not:**",
        {
          start: 1,
          list: [
            "give false details or pretend to be someone else when signing up, or use someone else's account;",
            "use the selfie search to look for anyone but themselves (or someone who clearly agreed, or their own child), or put other people's faces in their selfie library;",
            "try to break or get around the app's protections: automated or bulk requests, scraping photos, guessing passwords, getting around rate limits or the sign-in lockout, reaching admin pages or other people's data;",
            "remove, crop, cover or edit out a watermark, or save or screenshot previews to avoid paying;",
            "share download or bundle links publicly, or use bought photos beyond what the [buyer licence](#buyer-licence) allows;",
            "make false refund claims or disputes, or pay with money they are not allowed to use;",
            "send abusive, threatening or false content in reports, disputes, refund or removal requests, or any other text field;",
            "upload anything that is not a photo or could harm the service (malware, broken or oversized files).",
          ],
        },
        "**Photographers must not:**",
        {
          start: 9,
          list: [
            "upload photos they did not take or have no right to sell;",
            "upload sexual, violent, hateful or degrading images, or photos taken where people expect privacy, such as changing areas or toilets;",
            "put photos in an event they do not belong to, or create fake or misleading events, prices or event details;",
            "re-upload a photo QuickPitik removed, or work around a removal decision;",
            "buy their own photos, use fake purchases or coupons to move money, or try to cash out sales that are refunded or disputed;",
            "enter payout details for an account that is not theirs;",
            "use buyer names or anything else seen in their dashboard for anything except QuickPitik sales.",
          ],
        },
      ],
    },
    {
      id: "what-quickpitik-may-do",
      heading: "What QuickPitik may do",
      blocks: [
        "If someone breaks these Terms, or to protect users or the service, QuickPitik may:",
        {
          list: [
            "remove or restrict photos;",
            "refuse refunds and disputes;",
            "hold money from disputed sales; and",
            "suspend an account, with a written reason. A suspended photographer cannot upload, and their public pages are hidden.",
          ],
        },
      ],
    },
    {
      id: "appeals",
      heading: "Appeals",
      blocks: [
        `To appeal a decision (for example a suspension, a photo removal or a refund decision), email ${EMAIL_LINK} within 14 days of the decision. We answer within 14 days. Our answer on the appeal is final.`,
      ],
    },
    {
      id: "photo-removal",
      heading: "Photo removal",
      blocks: [
        "Anyone can ask us to remove a photo of them by email (see the [Privacy Policy](/privacy#photos-of-you)). The photographer gets a set time to respond, then QuickPitik makes the final decision. Photographers must follow removal decisions. Copies already downloaded by buyers cannot be recalled.",
      ],
    },
    {
      id: "liability",
      heading: "Liability",
      blocks: [
        "[ASK OWNER/lawyer: review the suggested wording below (G11)]",
        "QuickPitik is provided as it is. We work to keep it running and accurate, but we do not promise that it will always be available, that face or bib search will find every photo of you, or that it will never show you a photo of someone else.",
        "QuickPitik is not responsible for indirect losses, such as lost profits or lost opportunities. Our total liability to you for any claim is limited to the amount you paid us (or, for photographers, the fees we kept from your sales) in the 12 months before the claim. Nothing in these Terms limits liability that Philippine law does not allow us to limit, such as liability for fraud or gross negligence.",
        "You are responsible for what you upload and how you use photos. If your breach of these Terms leads to a claim against QuickPitik, you will cover QuickPitik's resulting costs.",
      ],
    },
    {
      id: "governing-law",
      heading: "Governing law",
      blocks: [
        "These Terms are governed by the laws of the Philippines. Disputes go to the courts of Cebu City.",
      ],
    },
    {
      id: "changes",
      heading: "Changes to these Terms",
      blocks: [
        "When we change these Terms, we post the new version on this page with a new version number and date. New sign-ups agree to the version shown when they sign up.",
      ],
    },
    {
      id: "contact",
      heading: "Contact",
      blocks: [OPERATOR, `Email: ${EMAIL_LINK}`],
    },
  ],
};
