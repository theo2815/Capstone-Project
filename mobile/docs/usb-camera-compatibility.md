# USB camera compatibility (Android, direct USB)

Scope: an Android phone connected straight to the camera with a USB / USB OTG cable.
Wi-Fi, Bluetooth, PTP/IP and desktop software are out of scope. Where the code decides
is `app/src/main/java/com/quickpitik/mobile/data/usb/ptp/` (`CameraDrivers.open()` picks
an adapter from the operations the camera advertises; `CameraQuirks.kt` holds the
demonstrated exceptions). Protocol facts come from libgphoto2's ptp2 camlib (pinned
commit in `usb-tethering-plan.md`), Sony's Camera Control PTP 2/3 Reference, Nikon's
per-model USB/MTP specifications (Z series) and the R6/R10 hardware journals.

## What the tiers mean

| Tier | Meaning |
|---|---|
| **Verified** | Run on the body here (physical shutter, card polling, upload). |
| **Implemented** | The adapter exists and the manufacturer or libgphoto2 documents the behaviour; no hardware pass here yet. The app logs each such assumption at watch start. |
| **Partial** | Card import is implemented; live body-shutter tethering is unknown or limited on that body. |
| **Unavailable** | The camera does not expose it over USB, or does not speak PTP at all. |

Two features are tracked per body. **Import** = browse the card and pull JPEGs after
shooting. **Live** = the photographer shoots with the body's own shutter and each new
JPEG reaches the phone within seconds.

## Canon

Adapters: `CanonEosCamera` for every body that advertises `SetRemoteMode` and
`SetEventMode` (EOS extension, 2006 onwards); `CanonBasicPtpCamera` for bodies without
them (import only, live refused with guidance). Remote mode is sent for every session and
left on at close; the card scan is the detector; EOS events are used where a body reports
them (adaptive, off on R6/R10). Camera-side setting on R-series: `Choose USB connection
app` = *Photo Import/Remote Control*. On DSLRs before 2010 the menu is `Communication`,
which must be *PTP*, not *Normal* (Canon's own protocol, not PTP).

| Model | Protocol | Import | Live | Notes |
|---|---|---|---|---|
| EOS 1D Mark II, 1Ds Mark II | basic PTP | Implemented | Unavailable | 2004 bodies; no EOS extension, libgphoto2 has no capture path either |
| EOS 20D, 350D (Rebel XT), 5D | basic PTP | Implemented | Unavailable | 2004-2005; same as above |
| EOS 30D, 400D (Rebel XTi) | EOS extension | Implemented | Implemented | first bodies with the EOS extension (libgphoto2 capture flag) |
| EOS 1D Mark III, 1Ds Mark III, 40D, 450D, 1000D | EOS extension | Implemented | Implemented | 1Ds Mark III assumed identical to 1D Mark III |
| EOS 50D, 5D Mark II, 500D, 7D, 1D Mark IV | EOS extension | Implemented | Implemented | |
| EOS 550D, 60D, 600D, 1100D, 1D X, 5D Mark III, 650D, 6D, 700D, 100D, 70D | EOS extension | Implemented | Implemented | |
| EOS 1200D (Rebel T5), 7D Mark II, 750D, 760D, 1300D, 77D, 800D, 6D Mark II | EOS extension | Implemented | Implemented | |
| EOS 5D Mark IV, 80D, 5DS, 5DS R, 2000D (Rebel T7) | EOS extension | Implemented | Implemented | CloseSession skipped at the end of a session (libgphoto2: the body rejects every command after it) |
| EOS 1DX Mark II, 1DX Mark III, 850D (Kiss X10i) | EOS extension | Implemented | Implemented | |
| EOS R (Ra), RP | EOS extension | Implemented | Implemented | CloseSession skipped (as above); USB app = Photo Import/Remote Control |
| EOS R6, R10 | EOS extension | **Verified** | **Verified** | R6: 214 live photos (2026-09-02); R10: warm-up session before the first vendor op |
| EOS R5, R5 Mark II, R5 C (photo mode), R3, R1 | EOS extension | Implemented | Implemented | same path as R6; R5 C only while in photo mode |
| EOS R6 Mark II, R6 Mark III, R7, R8, R50, R50 V, R100 | EOS extension | Implemented | Implemented | same path as R6 |
| EOS M50 | EOS extension, remote mode 0x15 | Implemented | Partial | libgphoto2 value for the EOS M family; body-shutter behaviour in that mode is untested |
| PowerShot G7 X Mark II | EOS extension, remote mode 0x15 | Implemented | Partial | libgphoto2 lists capture support; body-shutter behaviour untested |
| PowerShot G1 X Mark III, PowerShot V1, PowerShot V10 | expected EOS-style | Implemented | Partial | not in libgphoto2's table; the PowerShot rule applies only if the body advertises RemoteReleaseOn, otherwise plain remote mode 1 |
| IXUS 132 | basic PTP | Implemented | Unavailable | compact in transfer mode; libgphoto2 has no capture path |
| PowerShot G1 (2000) | no PTP | Unavailable | Unavailable | pre-PTP proprietary Canon protocol; the phone cannot talk to it |
| EOS C50 | cinema line | Unknown | Unavailable | still-photo PTP tethering is not documented for the cinema bodies; expect import of stills at most |
| "EOS 4D Digital", "R6 V" | — | — | — | not Canon model names; the nearest bodies are covered above (R6 Mark III, R50 V) |

## Nikon

Adapter: `NikonPtpCamera`. A connected Nikon is in "PC camera mode": dials, buttons and
the shutter stay active, shots go to the card, and "remote mode" (controls ignored) is
only entered by `ChangeCameraMode` or control commands, which the app never sends
(Nikon USB/MTP specifications, all 15 Z bodies; gphoto2 documents tethered capture on
Nikon DSLRs). Events come from `GetEventEx` or `GetEvent` alone, the interrupt pipe only
as a fallback. Both card slots are enumerated. Playback on the body is disabled while
connected, and a card swapped mid-connection is not seen until a replug.

| Model | Import | Live | Notes |
|---|---|---|---|
| D3, D3S, D3X, D300, D300S, D700, D90, D5000 | Implemented | Implemented | USB menu must be *MTP/PTP*, not *Mass Storage* |
| D4, D4S, D5, D6, D500, D600, D610, D750, D780, D800, D800E, D810, D810A, D850, Df | Implemented | Implemented | PTP only, no USB menu |
| D5100, D5200, D5300, D5500, D5600, D7000, D7100, D7200, D7500 | Implemented | Implemented | |
| Z5, Z5 II, Z50, Z50 II, Z6, Z6 II, Z6 III, Z7, Z7 II, Z8, Z9, Zfc, Zf | Implemented | Implemented | behaviour confirmed by Nikon's per-model USB/MTP specification; `USB data connection` = *MTP/PTP* |

## Sony

Adapter: `SonyPtpCamera` (read-only PC Remote). Live needs the camera set to **PC
Remote / Remote Shoot** USB mode with *Still Img. Save Dest.* = **PC+Camera**, *RAW+J PC
Save Img* = **JPEG Only**, and camera priority (Position Key 0x00, the default). Frames
arrive at the buffer handle 0xFFFFC001 and the card copy stays as backup. The card is
hidden in PC Remote mode, so **import needs the MTP USB mode**. Sony's Content Transfer
Mode (PTP 3, `SDIO_OpenSession` function mode 1 or 2) would allow import without
switching modes on newer bodies; it uses a separate content-list API and is not
implemented.

| Model | Protocol | Import | Live | Notes |
|---|---|---|---|---|
| α7 IV, α7 V, α7R V, α1, α1 II, α7C II, α7S III, α7C, α7CR, α6700, α9 II, α9 III | Camera Control PTP 3 | Implemented (MTP mode) | Implemented | values confirmed against Sony's PTP 3 Reference and example parser |
| α7R IV, α7R IV A | Camera Control PTP 3 (PTP 2 also listed) | Implemented (MTP mode) | Implemented | negotiation asks for 3, accepts 2 |
| α9 (2017) | PTP 2 era SDIO | Implemented (MTP mode) | Partial | absent from Sony's published tables; libgphoto2 drives it with the same commands, so the protocol 2 fallback applies; unverified |

## Common requirements and failure modes

- A data-capable USB cable and, on the phone, USB OTG host support. Charge-only cables
  and unpowered hubs are the first thing to rule out.
- The camera's USB menu decides the mode. Transfer-only modes (Sony MTP/Mass Storage,
  Canon *Normal*, any mass-storage mode) are detected and reported; the app cannot
  switch them.
- The app claims the USB interface exclusively, so Android's own MTP file browser must
  not be holding it (replug if the Files app opened the camera first).
- Two card slots, RAW+JPEG and JPEG-only are handled; RAW-only bodies get guidance and
  no upload. Nothing is ever deleted from the card.
- Every body-specific assumption is written to the capture log at watch start
  ("Assumed, not verified on this body: …"), so a photographer can see what is documented
  versus tested.

## Adding or correcting a model

- A new exception (a value, a skipped command, a warm-up) is one row in
  `CameraQuirks.kt`, matched by VID:PID or a model regex, first match wins.
- A brand-new protocol family is one adapter class over `GenericPtpCamera` with the
  four hooks (`startSession`, `startLive`, `events`, `objectAddedEvents`).
- After a hardware pass, move the body to **Verified** here and drop its assumption text
  in the adapter.
