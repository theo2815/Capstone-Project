# QuickPitik — brag plan (rev 2)

**What:** A race-photo marketplace for Cebu. Photographers plug their camera into their phone and every shot uploads as they shoot. Runners finish the race, find their photos by selfie or bib number, pay by QR, and download.
**For:** Runners buried under thousands of event photos, and the photographers who shoot them.
**Stands apart:** One pipeline for both sides: camera → phone → website → runner.
**Visual hook:** A wall of photo tiles too big to scroll through, then it narrows down to *you*.
**Tone:** `default`: punchy, playful, clean. Uses the site's "Finish Line" identity (bone paper, forest green, Anton, mono kickers, race stripe).
**Share caption:** Photographers shoot, and the photos are live before the runner crosses the finish.

**Out of scope (rev 2):** the desktop app (BatchMyPhotos), blur culling, and the "1–2 hrs → 10s" claim.

## Source-checked facts
- Tether: the camera connects to an Android phone by a **USB-C** cable (not wireless). The app detects it and asks for USB permission; the photographer taps **Start auto-upload**, and each shutter press uploads to the event. Real screen copy: `Camera · USB connected`, `Auto-upload · watching`, `Last capture: R6T_1083.JPG`. (`mobile/…/ui/photographer/DashboardScreen.kt`)
- Photographer copy (`website/src/app/photographers/page.tsx`): "Connect your phone to your camera." / "Every shot lands in the marketplace." / "No SD cards. No transfer. Photos go live as you press the shutter."
- Runner search: "Take a selfie" / "Upload a selfie" / "Enter your bib number" / "Search by bib →". Search time 5–10 sec.
- Buy and download (`website/src/components/cart/checkout-modal.tsx`, `orders/save-photos-button.tsx`): previews are watermarked, then Buy → "Scan to pay." with a QR Ph code (any bank or e-wallet) → "All yours." → "Save photo".

## Storyboard (23s, 120 BPM, cuts on the beat)
| # | Time | Scene | On screen |
|---|---|---|---|
| 1 | 0–3s | Hook | Photo wall scrolls fast. "15,000 RACE PHOTOS." then "WHICH ONES ARE YOU?" *(unchanged)* |
| 2 | 3–6s | Reveal | Logo header, race stripe, "Find your **race photos.** Or sell yours." with fanned photo tiles *(unchanged)* |
| 3 | 6–10.5s | **Photographer: tether** | Left: kicker FOR PHOTOGRAPHERS, then "CONNECT YOUR PHONE TO YOUR CAMERA." and "Every shot lands in the marketplace." Right: a camera wired to an Android phone by a USB-C cable. The phone screen shows `Camera · USB connected`, then `Auto-upload · watching`. Three shutter presses on the beat: each one ticks the capture counter (1 → 3), updates `Last capture: R6T_108x.JPG`, and flies a photo into a website gallery strip "Cebu Marathon 2026 · Open for uploads". |
| 4 | 10.5–14.5s | **Runner: selfie** | Kicker FOR RUNNERS · AFTER THE FINISH, "SNAP A SELFIE.", "We surface every shot of you." The Find-your-photos panel: click "Upload a selfie", shutter, face scan, 6 watermarked matches pop in, and SEARCH TIME 5–10 SEC. *(shortened from rev 1)* |
| 5 | 14.5–17s | **Runner: bib** | "OR TYPE YOUR BIB." The bib field types 1042, "Search by bib →" is pressed, and 4 tagged photos appear. *(shortened)* |
| 6 | 17–20s | **Runner: buy and download** | Cursor taps "Buy →" on a match. The checkout card shows "Scan to pay." with a QR Ph code and "Any bank or e-wallet", which flips to "All yours." ✓. Then "Save photo" is tapped, the watermark lifts off the photo, and it drops into a download tray. Left headline: "ALL YOURS." |
| 7 | 20–23s | Outro | Logo, "Race photos, delivered in minutes.", CEBU · PHILIPPINES, quickpitik.com |

Transitions: old content lifts out, the paper background holds, then new content rises in.

## Sound
Same 120 BPM A-major track, re-timed so the drop is at 3.0 and the final hit is at 20, ringing out through the outro. SFX stay tuned to the key and sit under the music: three camera-shutter clicks locked to the beat in scene 3 (the *pitik*), a soft whoosh as each photo lands in the gallery, pops as matches appear, typing ticks for the bib, and a short rising chime on "All yours."

## Share copy (draft)
The photographer plugs their camera into their phone, and every shot goes live as they shoot. By the time you cross the finish, your photos are waiting: find them with a selfie or your bib number, pay by QR, and download. QuickPitik, built in Cebu. quickpitik.com

## Voiceover (Kokoro `af_heart`)
Lines add to what's on screen rather than reading it out. The music ducks about 7 dB under the voice.
| Start | Line |
|---|---|
| 0.35s | Fifteen thousand photos from one race. |
| 3.3s | Meet QuickPitik, made in Cebu. |
| 6.3s | Plug your camera into your phone, and every shot goes live. |
| 10.75s | Crossed the finish line? One selfie finds every shot of you. |
| 14.7s | No selfie? Your bib number works too. |
| 17.25s | Pay by QR, and the full photo is yours. |
| 20.25s | QuickPitik. Race photos, delivered in minutes. |
