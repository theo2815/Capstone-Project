import { EMAIL_LINK, OPERATOR, type LegalDoc } from "./types";

// Draft text from the vault's PT-0a draft (2026-11-02). Not approved: PT-0b
// puts the final wording in. Section ids are linked from other pages and apps.
export const PRIVACY_POLICY: LegalDoc = {
  title: "Privacy Policy",
  sections: [
    {
      id: "who-we-are",
      heading: "Who we are",
      blocks: [
        "QuickPitik is a marathon and running-event photo service. Photographers upload event photos; runners find their photos by bib number or selfie and buy them on the website or the Android app.",
        `QuickPitik is run by ${OPERATOR}. QuickPitik Inc. decides how your personal data is used (it is the "personal information controller" under the Data Privacy Act of 2012, RA 10173).`,
        `Our Data Protection Officer is Theo Cedric Chan. Contact them at ${EMAIL_LINK}.`,
      ],
    },
    {
      id: "what-we-collect",
      heading: "What we collect",
      blocks: [
        "**When you create an account**",
        {
          list: [
            "Your name, email address, password (we store only a scrambled form called a hash, never the password itself) and role (runner or photographer).",
            "If you sign in with Google: your Google email, name, profile picture link and Google account number. If an account with the same email already exists, Google sign-in is linked to it.",
            "Account records: when you verified your email, failed sign-in attempts, and whether your account is suspended.",
          ],
        },
        "**Selfies and faces** (details in [Face matching](#face-matching))",
        {
          list: [
            "Selfies you save to your profile (up to 5), with a quality score.",
            "Selfies you use for a single search. These are not stored.",
            "Face data and bib numbers from event photos, for everyone who appears in them.",
          ],
        },
        "**Photos**",
        {
          list: [
            "Event photos uploaded by photographers, as the original file, a watermarked preview and a thumbnail, with the original file name. The original keeps the data the camera wrote into it (see [Photo metadata](#photo-metadata)).",
            "Your profile picture, if you add one.",
          ],
        },
        "**Photographers**",
        {
          list: [
            "Profile: handle, brand name, bio, region, province and city, social links, verification status, and whether photo details are shown publicly.",
            "Payout details: account name and number for GCash, Maya or GoTyme, and the payout QR image. A copy of these details is saved with each payout.",
          ],
        },
        "**Orders and payments**",
        {
          list: [
            "What you bought, your account or (for guests) the email you entered, any coupon, the payment method, when you paid, and PayMongo's reference numbers. We never see your bank or e-wallet login: you pay through PayMongo.",
            'A sales record that shows the photographer your name as first name and last initial (for example "Aira S."). For guests, this is made from the part of the email before the @.',
          ],
        },
        "**Other things you send us**",
        {
          list: [
            'Inbox messages, saved events, your cart, and "photos ready" alerts you turn on.',
            "Reports about photos, refund requests and disputes (with any note you write), and photo removal requests (your contact details, the reason, and any evidence file).",
          ],
        },
        "**Security records**",
        {
          list: [
            "Sign-in tokens and one-time codes (password reset, email verification, email change). We store them hashed.",
          ],
        },
        "**What we do not collect:** your phone number, date of birth, home address, government ID, location, or bank or e-wallet login. We use your internet (IP) address for a moment to limit repeated requests, but we do not store it. We have no analytics, advertising or tracking tools.",
      ],
    },
    {
      id: "how-we-use-it",
      heading: "How we use it",
      blocks: [
        {
          list: [
            "**Accounts:** to create your account and sign you in, with email and password or with Google.",
            "**Finding photos:** to match selfies with faces in event photos and to read bib numbers, and to email you when new photos of you may be ready (only if you turn this on).",
            "**Selling photos:** to take payment through PayMongo, send receipts and download links, and handle refunds.",
            "**Paying photographers:** to keep the sales record, handle payout requests and send payouts to the photographer's saved account.",
            "**Safety:** to handle reports, disputes and photo removal requests, to suspend accounts that break the [Terms](/terms#prohibited-conduct), and to keep a record of admin decisions.",
            "**Security:** to limit repeated requests, lock an account for a while after 5 wrong passwords, and require a second sign-in step for admins.",
            "**Emails:** we send only emails about your account and orders: password reset codes, email verification, email-change confirmation, \"photos ready\" alerts (if you turn them on), order receipts with download links, refund links, and an alert when a photographer's payout account changes. We send no newsletters or marketing emails.",
          ],
        },
        "We do not use your data for advertising, and we do not give it to advertisers or data brokers.",
        "[ASK OWNER/lawyer: review the suggested wording below (G8)]",
        "**Why we are allowed to use it** (Data Privacy Act, Section 12):",
        {
          list: [
            "Accounts, selling photos, paying photographers, and emails about your account and orders: to carry out our agreement with you (these Terms).",
            "Keeping orders, payments and sales records: to meet our legal duties, such as tax and accounting rules.",
            "Safety and security: our legitimate interest in protecting users and the service.",
            'Face matching with your selfie and "photos ready" alerts: your consent (the face-matching tick, and turning alerts on). You can withdraw it by deleting your saved selfies and turning alerts off.',
          ],
        },
      ],
    },
    {
      id: "face-matching",
      heading: "Face matching",
      blocks: [
        "Face matching is how QuickPitik finds your photos from a selfie. It uses Amazon Rekognition, a face-recognition service run by Amazon Web Services (AWS).",
        "**Event photos.** When a photographer uploads an event photo, we send it to Amazon Rekognition. It finds each face and stores a face template (a set of numbers that describes the face, not a picture) in a collection that belongs to that one event. It also reads bib numbers. This happens for **everyone in the photo, whether or not they have a QuickPitik account.** Faces from one event are never compared with another event's photos.",
        "**Searching with a selfie.** When you search an event with a selfie, we send the selfie to Amazon Rekognition to compare it with that event's face templates. Guests can search without an account. **If you do not save the selfie, we do not store it:** it is used for that search and then dropped.",
        '**Saved selfies.** Runners can save up to 5 selfies to their profile. We keep them in our cloud storage until you delete them (you can delete them yourself at any time in your profile). Saved selfies are not added to any event\'s face collection; we send a saved selfie to Amazon Rekognition each time you use it to search. If you turn on "photos ready" alerts for an event, we use the saved selfie you choose to look for you in that event\'s new photos.',
        "**Before your first search** the website and app show a short notice and ask you to tick that you agree. Your tick is remembered on that device.",
        "**How long.** An event's face templates are kept until the event is deleted, or until a photo is removed after a removal request (see [Photos of you](#photos-of-you)). Archived events keep their face templates.",
        "**Admins.** When someone asks us to remove photos of them, an admin may use the selfie they send to find those photos. That selfie is kept only for the request and deleted when it closes.",
        "[ASK OWNER/lawyer: review the suggested wording below (G9)]",
        "**People without an account.** For people in event photos who do not have a QuickPitik account, we rely on our legitimate interest in helping runners find their own race photos at events where photography is expected. Face templates stay inside one event and are compared only when someone searches that event. Anyone can ask us to remove their photos and face templates (see [Photos of you](#photos-of-you)).",
      ],
    },
    {
      id: "who-receives-it",
      heading: "Who receives it",
      blocks: [
        "We use these companies to run QuickPitik. Each receives only what it needs for its job.",
        {
          table: [
            ["Company", "What it receives", "Why"],
            ["Amazon Web Services (Amazon Rekognition)", "event photos and search selfies", "face matching and bib reading, in AWS's Singapore region"],
            ["Cloudflare (R2 storage)", "all images: original photos, previews, selfies, profile pictures, payout QR images, removal evidence", "storage"],
            ["PayMongo", "the buyer's email (the part before the @ is used as the billing name); if we send payouts through PayMongo, the photographer's payout account name and number", "payments, refunds, payouts"],
            ["Resend", "your email address, your name and the email's content", "sending emails"],
            ["Google", "your Google sign-in, if you use it", "sign-in"],
            ["Railway and Vercel", "everything we store or show", "hosting the server, database and website"],
          ],
        },
        "Some of these companies may store or process data outside the Philippines.",
        "**Other users see:**",
        {
          list: [
            "Photographers see buyers as first name and last initial.",
            "Anyone can see a photographer's public profile: handle, name, profile picture, bio, city and events.",
            "Anyone can see watermarked previews of photos in events they can open, and the photo details (camera, lens, author, copyright, date) if the photographer allows it.",
            "A buyer receives the original photo file, with any data the camera put in it.",
          ],
        },
        "**Admins** see every user's name and email, and the records they need to handle orders, payouts, reports and removal requests.",
        "We may also share data when Philippine law or a court requires it.",
      ],
    },
    {
      id: "how-long-we-keep-it",
      heading: "How long we keep it",
      blocks: [
        "We do not yet delete data automatically after a set time. Data is kept until you or QuickPitik delete it, as below.",
        {
          table: [
            ["Data", "How long"],
            ["Your account", "until it is deleted (see [Deleting your account](#deletion))"],
            ["Saved selfies", "until you delete them or your account is deleted"],
            ["Selfies used only for a search", "not stored"],
            ["Face templates from event photos", "until the event is deleted or the photo is removed"],
            ["Event photos", "until the photographer deletes them or they are removed after a request. Sold photos cannot be deleted by the photographer, because buyers keep their downloads"],
            ["Orders, payments and sales records", "kept for accounting for at least 5 years after the tax-return deadline for the year they belong to, as Philippine tax law requires, and longer while a tax case or claim about them is open. This applies after an account is deleted too"],
            ["Download links for bought photos", "downloads work for one year after payment"],
            ["Photo removal evidence", "deleted after the request is closed"],
            ["Sign-in tokens and codes (hashed)", "not yet cleaned up automatically"],
            ["Server logs (technical records our server writes; they can include your email address when we send you an email)", "kept by our hosting provider, Railway, for the time its settings allow"],
            ["Backups", "our hosting and storage providers may keep backup copies for the time their settings allow"],
          ],
        },
      ],
    },
    {
      id: "security",
      heading: "Security",
      blocks: [
        {
          list: [
            "Passwords are stored as bcrypt hashes, never as plain text.",
            "Sign-in tokens last 15 minutes and refresh tokens 7 days; we store them hashed.",
            "After 5 wrong passwords an account is locked for a short time, and repeated requests are limited.",
            "Admins need a second sign-in step.",
            "The website and app talk to our server over HTTPS (the Android app refuses plain HTTP connections).",
            "Original photos are reached only through links that expire: 15 minutes for buyers, 5 minutes for photographers, 1 hour for previews and profile images.",
            "Public previews carry watermarks.",
            "Payments happen on PayMongo; we never see your bank or e-wallet login.",
          ],
        },
        `No system is perfectly secure. If you think your account has been misused, email ${EMAIL_LINK}.`,
      ],
    },
    {
      id: "your-rights",
      heading: "Your rights",
      blocks: [
        "Under the Data Privacy Act you have the right to be informed, to see your data, to correct it, to object, to have it deleted or blocked, to get a copy, to claim damages, and to complain to the National Privacy Commission. Here is how to use them on QuickPitik today:",
        {
          list: [
            "**See and correct:** change your name, email (with a confirmation email) and profile picture in your account settings.",
            "**Delete selfies:** in your profile, at any time.",
            '**Stop "photos ready" alerts:** turn them off for each event.',
            `**Ask for a copy of your data:** email ${EMAIL_LINK}. There is no download button yet.`,
            "**Delete your account:** see [Deleting your account](#deletion).",
            "**Remove photos of you:** see [Photos of you](#photos-of-you).",
            `**Object, or ask us to block data:** email ${EMAIL_LINK}.`,
          ],
        },
        "If you are not happy with our answer, you can complain to the National Privacy Commission (privacy.gov.ph).",
      ],
    },
    {
      id: "deletion",
      heading: "Deleting your account",
      blocks: [
        `**How to ask.** Email ${EMAIL_LINK} from the email address on your account, with the subject "Account deletion request". You can also tap "Request account deletion" in the app's settings, which opens that email for you. We delete or close the account within 7 business days and reply when it is done.`,
        '**Runner accounts** are deleted. This removes your profile, saved selfies, profile picture, saved events, cart, "photos ready" alerts, inbox messages and sign-in records.',
        "**Photographer accounts** are closed, not deleted, because sales and payout records must be kept. Before you ask:",
        {
          start: 1,
          list: [
            "Delete your unsold photos in the app (move them to the trash, then delete them forever). This erases the files and their face templates.",
            "Make sure no payout is waiting. We pay out any unpaid balance by hand first, even below the usual minimum.",
          ],
        },
        "We then remove your name, email, profile picture, bio, social links, saved payout details and sign-in records, hide your public page, and mark the account as closed. Photos you sold stay, so buyers keep their downloads. The payout details copied into each past payout stay with that payout record.",
        '**What we keep for every account:** orders, payments and sales records (including the email on each order and the "first name, last initial" shown on sales), and the notes in reports, disputes and removal requests. We keep them for accounting and to handle refunds and disputes (see [How long we keep it](#how-long-we-keep-it)).',
        "**Your face in event photos** is not linked to your account, so deleting the account does not remove it. To remove photos of you, see [Photos of you](#photos-of-you).",
      ],
    },
    {
      id: "photos-of-you",
      heading: "Photos of you",
      blocks: [
        `If you appear in an event photo and want it removed, email ${EMAIL_LINK} with the event and the photos (a link or a description), and why. A parent or guardian can ask for a child.`,
        "The photographer gets a set time to respond, then QuickPitik decides. If we remove a photo, we delete its files and its face templates; a minimal record of the removal is kept. We cannot recall copies that buyers already downloaded, screenshots, or copies shared elsewhere.",
      ],
    },
    {
      id: "children",
      heading: "Children",
      blocks: [
        "You must be 18 or older to create an account or buy photos. A parent or guardian may search for and buy photos of their child.",
        "Children often take part in races and appear in event photos. A parent or guardian can ask us to remove a child's photos (see [Photos of you](#photos-of-you)). If we learn that someone under 18 has an account, we may close it.",
      ],
    },
    {
      id: "on-your-device",
      heading: "On your device",
      blocks: [
        "**Website.** The website sets no cookies of its own. It keeps these in your browser's storage: your sign-in tokens, cart, a payment in progress, saved events, display choices, tips you have dismissed, and whether you have agreed to the face-matching notice. Fonts are part of the website (no font service is called). The sign-in pages load Google's sign-in button from Google; Google's own policies apply to it.",
        "**Android app.**",
        {
          list: [
            "The app keeps your sign-in details, name, email, role and profile picture link in its private storage on your phone.",
            "Photographers: photos waiting to upload stay in the app's storage until you clear them, along with a list of uploads and camera transfers.",
            "Downloaded photos are saved to Pictures/QuickPitik and Downloads/QuickPitik.",
            "**Selfies you take with the in-app camera are saved to your phone's gallery and stay there after upload.** Delete them from your gallery if you do not want to keep them.",
            "Your agreement to the face-matching notice is remembered on the phone.",
            "The app asks only for internet access, and (for photographers who connect a camera by USB) USB, notification and background-service permissions. It does not ask for your camera, location, contacts or files: taking or picking a photo uses your phone's own camera and photo picker.",
          ],
        },
      ],
    },
    {
      id: "photo-metadata",
      heading: "Photo metadata",
      blocks: [
        "Cameras write data into each photo: camera and lens model, settings, the time it was taken, and sometimes the author, copyright and **location (GPS)**. QuickPitik keeps the original file as the photographer uploaded it. **A buyer who downloads the original gets all of this data, including the location if the camera recorded it.** Previews and thumbnails are separate copies.",
        "Photographers choose whether the camera, lens, author, copyright and date are shown on QuickPitik's photo pages. Photographers who do not want to share location should turn it off in the camera or remove it before uploading.",
      ],
    },
    {
      id: "changes",
      heading: "Changes to this policy",
      blocks: [
        "When we change this policy, we post the new version on this page with a new version number and date. New sign-ups agree to the version shown when they sign up.",
      ],
    },
    {
      id: "contact",
      heading: "Contact",
      blocks: [
        OPERATOR,
        `Email: ${EMAIL_LINK}`,
        "Data Protection Officer: Theo Cedric Chan, at the same email.",
      ],
    },
  ],
};
