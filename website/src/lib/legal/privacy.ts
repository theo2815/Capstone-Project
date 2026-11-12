import { EMAIL_LINK, OPERATOR, type LegalDoc } from "./types";

// Text approved by the owner on 2026-11-04 (PT-0b): the PT-0a draft with the
// 2026-11-04 legal review's wording applied. Section ids are linked from other
// pages and apps.
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
            "Seller checks, required before you can upload: your full legal name, photos of a valid government ID (both sides for a PhilSys National ID), a photo of you holding that ID, your ID number, your address and phone number; and, if you choose, your BIR Certificate of Registration (Form 2303). We check these automatically (see [Seller checks](#seller-checks)), or an admin checks them by eye if you choose. We do not show your ID, your ID number, these photos or your phone number publicly.",
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
            "Reports about photos, events and photographers, refund requests and disputes (with any note you write), and photo removal requests (your contact details, the reason, and any evidence file). If you ask on the website with a live check, the request also holds the event you chose, the reason you picked, when you agreed and to which version of this policy, the result of the live check, and one still image of your face from the check (see [Deletion requests with a live check](#face-matching)).",
          ],
        },
        "**Security records**",
        {
          list: [
            "Sign-in tokens and one-time codes (password reset, email verification, email change). We store them hashed.",
          ],
        },
        "**What we do not collect:** from runners and buyers, phone number, date of birth, home address, government ID or location; from anyone, bank or e-wallet logins. We use your internet (IP) address to limit repeated requests. We do not save it in our database, but our hosting providers' technical logs may record it (see [How long we keep it](#how-long-we-keep-it)). We have no analytics, advertising or tracking tools.",
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
            "**Safety:** to check photographers' seller details (see [Seller checks](#seller-checks)); to check photographers' brand names, bios, handles and race event names against lists of offensive words, and their profile pictures, covers, watermarks and race covers for sexual, violent or hateful images, using Amazon Rekognition; to handle reports from signed-in users about photos, events and photographers, disputes and photo removal requests; to hide race events that no one can confirm took place; to suspend accounts that break the [Terms](/terms#prohibited-conduct), and to keep a record of admin decisions.",
            "**Automatic checks on uploads:** when a photographer uploads photos, we check that they belong to the event and are the photographer's own. We work out a fingerprint of each photo (a short code computed from the image, not a face template) and compare it with other photographers' photos in the same event. We also check whether the same file was already uploaded to QuickPitik by another photographer, in any event, look for the QuickPitik photo id that our watermarked previews carry, compare the bib numbers read from the photos with other photographers' photos in the same event, and compare the date the camera recorded with the event's date. We store the results for each photographer and event and do not show them publicly. If the dates suggest photos are in the wrong event, we may send the photographer a message in their inbox. Otherwise only admins see the results, and an admin may see both photographers' names and sample photos side by side. A person reviews the results before any photo is hidden or any payout is held, and the photographer is told in their inbox when a hold starts and when it ends.",
            "**Promoting QuickPitik:** we use event photos in our own promotion only when no one in them can be recognised, or the people who can be recognised agreed, and we credit the photographer.",
            "**Security:** to limit repeated requests, lock an account for a while after 5 wrong passwords, and let admins turn on a second sign-in step.",
            "**Emails:** we send only emails about your account and orders: password reset codes, email verification, email-change confirmation, \"photos ready\" alerts (if you turn them on), order receipts with download links, refund links, an alert when a photographer's payout account changes, an alert when a photographer's payouts are put on hold, updates about a photo deletion request you made, and, for photographers, a notice when photos of theirs are hidden for review after a request and the decision on them, and one notice before the identity check becomes required to upload. We send no newsletters or marketing emails.",
          ],
        },
        "We do not use your data for advertising, and we do not give it to advertisers or data brokers.",
        "**Why we are allowed to use it** (Data Privacy Act, Sections 12 and 13):",
        {
          list: [
            "Accounts, buying and selling photos, paying photographers, and emails about your account and orders: to carry out our agreement with you (the [Terms](/terms)).",
            "Orders, payments, sales and payout records: to meet our legal duties, such as tax and bookkeeping rules, and to handle refunds, disputes and legal claims.",
            "Seller checks (a photographer's government ID, the photo of them holding it, the ID number and BIR registration): because the Internet Transactions Act and BIR rules require them, and to make sure the ID belongs to the person selling. We treat ID numbers as sensitive personal information.",
            'The live selfie and the face comparison in a seller check: your consent, given with the tick before the live selfie starts. If you do not agree, choose "Have an admin check instead"; you can still sell.',
            "Safety, security, automatic checks on uploads and photo removal requests: our legitimate interest in protecting users, photographers and the service. A legitimate interest is a real need of ours that does not override your rights.",
            'Face matching with your selfie, saved selfies and "photos ready" alerts: your consent, given with the tick before you search or save a selfie, and by turning alerts on. A parent or guardian gives it for their child. You can withdraw it at any time by deleting your saved selfies and turning alerts off; searches already made are not undone.',
            "The live check and the face search for a deletion request: your consent, given with the tick before the check starts. If you do not agree, you can ask by email instead (see [Photos of you](#photos-of-you)).",
            "Faces and bib numbers of people in event photos who have not searched: our legitimate interest in helping runners find their own race photos (see [People without an account](#face-matching)).",
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
        "**Before you search** by selfie or by bib number, the website and app show a short notice and ask you to tick that you agree. Your tick is remembered on that device.",
        "**How long.** An event's face templates are deleted 6 months after the event date, or earlier if the event is deleted or a photo is removed after a request (see [Photos of you](#photos-of-you)). After that, the event's photos can still be found by bib number.",
        "**Deletion requests with a live check.** A runner account can ask us to delete photos of them from one event, on the website. The website first runs a live check: Amazon Rekognition Face Liveness, run by AWS in Malaysia, watches a few seconds of video from your camera to confirm a real person is there, and gives us one still image of your face. We use that image once, to search only the event you chose (Amazon Rekognition in Singapore). It is not added to any face collection, not used for any other search, and not compared with other accounts. We keep it with your request as evidence and delete it when the request closes. AWS keeps no extra images from the check for us. Very close matches may be hidden from the gallery at once; every other match waits for an admin, and only an admin deletes a photo. The photographers are told how many of their photos are affected, but not who asked, why, or what you look like.",
        "**Admins.** When someone asks us by email to remove photos of them, an admin may use the selfie they send to find those photos. That selfie is kept only for the request and deleted when it closes.",
        "**People without an account.** Most people in event photos have never used QuickPitik. For them we rely on our legitimate interest in helping runners find their own race photos at events where photography is expected. To protect them, face templates stay inside one event's collection, are never shown to anyone, are compared only when someone searches that event after ticking the notice, and are deleted 6 months after the event date. You can object, or ask us to remove your photos and face templates, at any time (see [Photos of you](#photos-of-you)). A parent or guardian can do this for a child.",
      ],
    },
    {
      id: "seller-checks",
      heading: "Seller checks",
      blocks: [
        "When a photographer sends their seller details, we check them automatically before an admin needs to look.",
        "**Live selfie.** In your browser, you take a short live video selfie. Amazon Rekognition Face Liveness, run by AWS in Malaysia, checks that a real person is in front of the camera and gives us one still image of your face. AWS keeps no extra images from the check for us. We then compare that still image with the photo on the front of your ID, using Amazon Rekognition in Singapore. We never compare it with the photo of you holding your ID.",
        "**ID number.** You type your ID number. Amazon Rekognition in Singapore reads the text on the front of your ID, and we check that the number you typed is on it. We do not keep the number itself: we keep only a fingerprint of it (a code made from the number with a secret key, which cannot be turned back into the number). We use the fingerprint, your payout account number and your phone number to see whether the same person already has another account, and we compare your legal name with the name on your payout account.",
        "**What happens next.** If every check passes, your seller details are approved automatically. If any check does not pass, or another account matches, an admin reviews your details. Only an admin can reject them, and an admin may review automatically approved details later. If an admin rejected your details before, an admin always reviews them again.",
        '**Rather not take the live selfie?** Choose "Have an admin check instead". We then run no live check, no ID text reading and no face comparison: an admin compares your photos by eye. We still keep the fingerprint of your ID number for the repeat-account check.',
        "**What the still image is used for.** The still image from a seller check is used only to check your identity. It is never added to a face collection, never used to search event photos, and never compared with other accounts. Admins can see it, like your ID photos.",
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
            ["Amazon Web Services (Amazon Rekognition)", "event photos, search selfies, photographers' ID front and live-selfie video and still image, photographers' profile pictures, covers, watermarks and race covers, and the live-check video and image for a deletion request", "face matching, bib reading, ID text reading, seller face comparison and image checks in AWS's Singapore region; the live checks in AWS's Malaysia region"],
            ["Cloudflare (R2 storage)", "all images: original photos, previews, selfies, profile pictures, payout QR images, seller-check photos and the still image from the seller live selfie, removal evidence and the still image from a deletion live check", "storage"],
            ["PayMongo", "the buyer's email (the part before the @ is used as the billing name); if we send payouts through PayMongo, the photographer's payout account name and number", "payments, refunds, payouts"],
            ["Resend", "your email address, your name and the email's content", "sending emails"],
            ["Google", "your Google sign-in, if you use it", "sign-in"],
            ["Railway and Vercel", "everything we store or show", "hosting the server, database and website"],
          ],
        },
        "Amazon Rekognition processes event photos, selfies, ID photos and profile images in Singapore, and the live checks (for seller checks and deletion requests) in Malaysia. The other companies above may also store or process data outside the Philippines. Wherever your data is, QuickPitik stays responsible for it, and we use these companies' data-protection terms to keep it protected to a level comparable with Philippine law.",
        "**Other users see:**",
        {
          list: [
            "Photographers see buyers as first name and last initial.",
            "Anyone can see a photographer's public profile: handle, name, profile picture, bio, city and events.",
            "Anyone can see watermarked previews of photos in events they can open, and the photo details (camera, lens, author, copyright, date) if the photographer allows it.",
            "A buyer receives the original photo file, with any data the camera put in it.",
          ],
        },
        "**Admins** see every user's name and email, and the records they need to handle orders, payouts, reports, upload checks and removal requests.",
        "We may also share data when Philippine law or a court requires it.",
      ],
    },
    {
      id: "how-long-we-keep-it",
      heading: "How long we keep it",
      blocks: [
        "Apart from face templates, which are deleted 6 months after the event, we do not yet delete data automatically after a set time. Data is kept until you or QuickPitik delete it, as below.",
        {
          table: [
            ["Data", "How long"],
            ["Your account", "until it is deleted (see [Deleting your account](#deletion))"],
            ["Saved selfies", "until you delete them or your account is deleted"],
            ["Selfies used only for a search", "not stored"],
            ["Face templates from event photos", "6 months after the event date, or earlier if the event is deleted or the photo is removed"],
            ["Event photos", "until the photographer deletes them or they are removed after a request. Sold photos cannot be deleted by the photographer, because buyers keep their downloads"],
            ["Orders, payments and sales records", "kept for at least 5 years after the tax-return deadline for the last year recorded in our books, as Philippine tax law requires (Tax Code Section 235, as amended by RA 11976), and longer while a tax audit, case or claim about them is open. This applies after an account is deleted too"],
            ["Photographer seller-check details and photos, the still image from the live selfie, and the fingerprint of the ID number", "while you sell on QuickPitik, then for at least 5 years like other money records. AWS keeps no extra images from the live selfie for us"],
            ["Download links for bought photos", "downloads work for one year after payment"],
            ["Results of automatic checks on uploads, and admin reviews of them", "until the photo, the event or the photographer's account is deleted"],
            ["Photo removal evidence, including the live-check image", "deleted after the request is closed"],
            ["The rest of a deletion request (event, reason, contact, consent, live-check result)", "kept with the request record, like other removal requests"],
            ["Sign-in tokens and codes (hashed)", "not yet cleaned up automatically"],
            ["Server logs (technical records our server and hosting providers write; they can include your IP address, and your email address when we send you an email)", "kept by our hosting provider, Railway, for the time its settings allow"],
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
            "Admins can turn on a second sign-in step (a code from an authenticator app).",
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
        "We reply within 15 days. Before we act, we may ask you to show the request is yours, for example by writing from your account's email address. A copy of your data comes in a common electronic format. After a person's death, their lawful heirs can use these rights.",
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
        "We then remove your name, email, profile picture, bio, social links, saved payout details and sign-in records, hide your public page, and mark the account as closed. Photos you sold stay, so buyers keep their downloads. The payout details copied into each past payout stay with that payout record. Seller-check details and photos you gave us are kept as legal records.",
        '**What we keep for every account:** orders, payments and sales records (including the email on each order and the "first name, last initial" shown on sales), and the notes in reports, disputes and removal requests. We keep them for accounting and to handle refunds and disputes (see [How long we keep it](#how-long-we-keep-it)).',
        "**Your face in event photos** is not linked to your account, so deleting the account does not remove it. To remove photos of you, see [Photos of you](#photos-of-you).",
      ],
    },
    {
      id: "photos-of-you",
      heading: "Photos of you",
      blocks: [
        "**With a runner account:** go to your account and choose Request photo deletion, pick the event and a reason, and take the live check. You can follow the request and withdraw it in your account. Photos uploaded after you ask are not included; you can ask again once the request closes.",
        `**By email, for anyone:** email ${EMAIL_LINK} with the event and the photos (a link or a description), and why. Use email if you have no runner account, cannot or would rather not use the camera, are asking for a child (a parent or guardian can ask for a child), or the event's face templates are already deleted.`,
        "The photographer gets 72 hours to respond, then QuickPitik decides. If we remove a photo, we delete its files and its face templates; a minimal record of the removal is kept. We cannot recall copies that buyers already downloaded, screenshots, or copies shared elsewhere.",
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
        "**Website.** The website sets no cookies of its own. It keeps these in your browser's storage: your sign-in tokens, cart, a payment in progress, saved events, display choices, tips you have dismissed, and whether you have agreed to the face-matching notice. Fonts are part of the website (no font service is called). The sign-in pages load Google's sign-in button from Google; Google's own policies apply to it. During a live check (a seller check or a deletion request), the browser asks to use your camera; the video goes to AWS and is not saved on your device.",
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
