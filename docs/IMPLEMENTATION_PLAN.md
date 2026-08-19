# QuickPitik — Implementation Plan (phase map)

> Created 2026-08-19. This path had been referenced as the "phased roadmap" from four documents without the file ever existing; it now holds the phase history plus the live roadmap. Day-to-day status lives in the root `CLAUDE.md` module table and the vault's `VAULT-INDEX.md` — this file changes only when a phase opens or closes.

## Phase history (shipped)

| Phase | Scope | Landed |
|---|---|---|
| ai-api Phases 1–6 | blur detection/classification, face recognition, bib OCR, async batch + webhooks | pre-2026-04 |
| Backend Phases A–H (PRs 1–11) | auth → events → photos → cart/saved → orders/payments → profile/selfies → photographer → earnings → admin → polish | 2026-05-05 → 05-10 |
| Website four-role build | Guest · Runner · Photographer · Admin, all locked | 2026-05-01 → 05-19 (audit closes through 08-16) |
| ai-api integration | face + bib search live end-to-end (six bugs fixed in one session) | 2026-05-20 |
| Mobile parity | runner + photographer replicate the website (`/admin/*` out of scope) | 2026-05-25 → 08-14 |
| Hardening arcs | backend audit closes + icebox drain · ai-api prod-deployment hardening · three-module reconciliation | 2026-08-14 → 08-16 |

## Live roadmap (what remains)

1. **Mobile verification** — emulator/device passes (in progress since 2026-08-19), then the final milestone: hardware tether verification on the Canon R6 (shutter-watch wiring done; protocol in vault `_journal/2026-08-14-mobile-live-auto-upload-wiring`).
2. **ai-api public deployment** — prod compose is hardened and builds; deploy to a public host, issue the desktop's restricted `blur:read`/`jobs:read` key, slim the CUDA-laden image (#p2).
3. **Google OAuth** — a cross-module build, not a wiring gap: the backend half does not exist and it is blocked on Google credentials (vault `_project/decisions.md`, 2026-08-15).
4. **Real-event beta** — one Cebu marathon end-to-end, then payments launch (PayMongo checkout + webhook already live in dev).

## Capstone track

SRS active this semester, SDD next — workspace + protocol in the vault (`Documentation for capstone paper/`, governed by the Document Skill).
