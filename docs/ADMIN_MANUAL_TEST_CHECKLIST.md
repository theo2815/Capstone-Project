# QuickPitik Website — Admin Portal Manual Test Checklist

> **Document Type:** Manual Quality Assurance (QA) Checklist  
> **Target Surface:** `website/src/app/admin/*` & `website/src/components/admin/*`  
> **Target Audience:** QA Testers, Capstone Panelists, Developers, System Administrators  
> **Prerequisites:** Running backend (`http://localhost:8080`), Next.js website (`http://localhost:3000`), active Admin account (`admin@quickpitik.com`).

---

## 📋 Table of Contents
1. [Prerequisites & Initial Setup](#1-prerequisites--initial-setup)
2. [Phase 1: Access Control & Role Guards](#phase-1-access-control--role-guards)
3. [Phase 2: Admin Shell, Navigation & Command Palette](#phase-2-admin-shell-navigation--command-palette)
4. [Phase 3: Overview & KPI Dashboard (`/admin/overview`)](#phase-3-overview--kpi-dashboard-adminoverview)
5. [Phase 4: Unified Inbox (`/admin/inbox`)](#phase-4-unified-inbox-admininbox)
6. [Phase 5: Photographer Verifications Queue (`/admin/verifications`)](#phase-5-photographer-verifications-queue-adminverifications)
7. [Phase 6: Disputes Management (`/admin/disputes`)](#phase-6-disputes-management-admindisputes)
8. [Phase 7: Content Moderation & Flags Queue (`/admin/flags`)](#phase-7-content-moderation--flags-queue-adminflags)
9. [Phase 8: Payouts & Financial Ledger (`/admin/payouts`)](#phase-8-payouts--financial-ledger-adminpayouts)
10. [Phase 9: Events Management (`/admin/events`)](#phase-9-events-management-adminevents)
11. [Phase 10: Photographers Roster & Messaging (`/admin/photographers`)](#phase-10-photographers-roster--messaging-adminphotographers)
12. [Phase 11: Sales Analytics & Leaderboards (`/admin/sales`)](#phase-11-sales-analytics--leaderboards-adminsales)
13. [Phase 12: Real-Time WebSocket Updates & Toast Sync](#phase-12-real-time-websocket-updates--toast-sync)

---

## 1. Prerequisites & Initial Setup

- [x] **1.1 Services Running:**
  - Backend Spring Boot API is healthy on port `8080`.
  - Next.js Web App is running on port `3000`.
- [x] **1.2 Database Seeded:**
  - Database contains at least 1 Admin user, 2 Photographers (1 verified, 1 pending verification), 2 Runners, and 1 Event with photos.
- [x] **1.3 Environment Variables:**
  - `website/.env.local` contains valid `NEXT_PUBLIC_API_URL=http://localhost:8080/api/v1` and `NEXT_PUBLIC_WS_URL=ws://localhost:8080/ws`.

---

## Phase 1: Access Control & Role Guards

- [x] **TC-ADM-01: Guest Access Rejection**
  - **Action:** In an incognito / logged-out window, navigate directly to `http://localhost:3000/admin` or `http://localhost:3000/admin/overview`.
  - **Expected Result:** Immediately bounced/redirected to `/login?from=%2Fadmin%2Foverview`. No administrative data is flashed.
- [x] **TC-ADM-02: Runner Role Access Rejection**
  - **Action:** Sign in as a user with role `RUNNER`. Attempt to access `http://localhost:3000/admin/overview`.
  - **Expected Result:** Blocked with a 403 Forbidden alert or bounced back to `/events` or `/account`.
- [x] **TC-ADM-03: Photographer Role Access Rejection**
  - **Action:** Sign in as a user with role `PHOTOGRAPHER`. Attempt to access `http://localhost:3000/admin/overview`.
  - **Expected Result:** Blocked and redirected to `/dashboard`.
- [x] **TC-ADM-04: Admin Authentication Success**
  - **Action:** Sign in with an `ADMIN` account. Navigate to `http://localhost:3000/admin`.
  - **Expected Result:** Automatically routed to `/admin/overview`. Admin sidebar navigation and header indicators display properly.

---

## Phase 2: Admin Shell, Navigation & Command Palette

- [x] **TC-ADM-05: Sidebar Rail Navigation Links**
  - **Action:** Click through each link in the left Admin Rail: Overview, Inbox, Events, Verifications, Disputes, Payouts, Photographers, Flags, Sales.
  - **Expected Result:** Each route mounts smoothly with active indicator dot/pill on the rail. No full page reloads.
- [x] **TC-ADM-06: Mobile Strip Responsive Behavior**
  - **Action:** Shrink browser viewport to mobile width (<768px).
  - **Expected Result:** Desktop sidebar tucks away; bottom/top mobile admin strip appears with horizontally scrollable section tabs.
- [x] **TC-ADM-07: Command Palette Trigger (`Cmd+K` / `Ctrl+K`)**
  - **Action:** Press `Ctrl+K` (Windows) or `Cmd+K` (macOS).
  - **Expected Result:** Centered search modal opens instantly with a frosted backdrop blur. Focus is automatically placed in the search input.
- [x] **TC-ADM-08: Command Palette Navigation & Search**
  - **Action:** Type "payouts" or "verifications" in the palette and press `Enter`.
  - **Expected Result:** Dialog closes and the browser navigates immediately to the selected section.
- [x] **TC-ADM-09: Keyboard Shortcuts Legend (`?` Key)**
  - **Action:** Press `?` on the keyboard while on any admin page.
  - **Expected Result:** The Keyboard Shortcuts legend modal pops up, displaying shortcuts for fast navigation (`G then O` for overview, `G then I` for inbox, `Escape` to close).

---

## Phase 3: Overview & KPI Dashboard (`/admin/overview`)

- [x] **TC-ADM-10: KPI Metric Strip Rendering**
  - **Action:** Inspect the top KPI summary tiles (Total Revenue, Pending Payouts, Open Disputes, Pending Verifications, Active Photographers).
  - **Expected Result:** Numbers match server data (tabular numerals formatted cleanly, non-zero where data exists, no `NaN` or `undefined`).
- [ ] **TC-ADM-11: 30-Day Decisions Trend Chart**
  - **Action:** Verify the trend line / bar chart in the center slab. Hover over data points.
  - **Expected Result:** Tooltips display date and action counts (e.g. "May 12: 4 approvals, 1 refund").
- [ ] **TC-ADM-12: Recent Decisions Timeline**
  - **Action:** Scroll down to the decisions audit trail.
  - **Expected Result:** Chronological list of admin actions (dispute resolved, payout held, verification approved) with relative timestamps (e.g., "2 hours ago") and actor badge.

---

## Phase 4: Unified Inbox (`/admin/inbox`)

- [ ] **TC-ADM-13: Aggregated Queue Rendering**
  - **Action:** Navigate to `/admin/inbox`.
  - **Expected Result:** Combines pending items requiring administrative action across Verifications, Disputes, Flags, and Payouts into a unified feed.
- [ ] **TC-ADM-14: Filter Chips Filtering**
  - **Action:** Click filter chips: "All", "Verifications", "Disputes", "Flags", "Payouts".
  - **Expected Result:** The list instantly filters to show only the selected category without page reload. Counter badges update accurately.
- [ ] **TC-ADM-15: Search Query Bar**
  - **Action:** Type an event name, photographer handle, or runner email in the search field.
  - **Expected Result:** Cards live-filter to match the query. Clearing the input restores the full queue.

---

## Phase 5: Photographer Verifications Queue (`/admin/verifications`)

- [ ] **TC-ADM-16: Verification Queue Status Tabs**
  - **Action:** Switch between "Pending Review", "Approved", and "Rejected/Suspended" tabs.
  - **Expected Result:** Lists update to display photographers in the corresponding status.
- [ ] **TC-ADM-17: Portfolio & Government ID Inspection**
  - **Action:** Click a pending photographer card to open the detail drawer.
  - **Expected Result:** Displays photographer legal name, contact email, equipment list, submitted portfolio sample links, and ID verification document thumbnails.
- [ ] **TC-ADM-18: Approve Photographer Flow**
  - **Action:** Click "Approve Verification". Confirm in the confirmation dialog.
  - **Expected Result:** Photographer status flips to `VERIFIED`. Success toast appears. The photographer disappears from "Pending" and moves to "Approved". A notification is pushed to the photographer's inbox.
- [ ] **TC-ADM-19: Reject Photographer Flow**
  - **Action:** Click "Reject". Provide a mandatory rejection reason (e.g. "ID document unreadable, please re-upload clear photo"). Confirm.
  - **Expected Result:** Photographer moves to "Rejected". Card reflects rejection notes. Notification with the reason is transmitted to the photographer.
- [ ] **TC-ADM-20: Suspend Verified Photographer**
  - **Action:** Select an active verified photographer. Click "Suspend Account". Enter justification note.
  - **Expected Result:** Account is suspended. Photographer's upload capabilities are revoked immediately. Status badge updates to "SUSPENDED" in amber/red.
- [ ] **TC-ADM-21: Reset Verification Flow**
  - **Action:** Click "Reset Verification" on a suspended or rejected photographer.
  - **Expected Result:** Resets status back to `PENDING` or unverified state, allowing the photographer to re-submit documentation.

---

## Phase 6: Disputes Management (`/admin/disputes`)

- [ ] **TC-ADM-22: Dispute Detail Deep-Dive (`/admin/disputes/[id]`)**
  - **Action:** Click on an open dispute in the queue.
  - **Expected Result:** Detail view opens showing:
    - Runner's complaint reason (e.g. "Photo blurry", "Wrong runner identified", "Duplicate charge").
    - Order summary with photo thumbnail, price, event name, and transaction date.
    - Photographer credit and response (if provided).
- [ ] **TC-ADM-23: Photo Inspection Lightbox (`Review` Mode)**
  - **Action:** Click on the contested photo thumbnail in the dispute card.
  - **Expected Result:** Opens `PhotoPreviewCard` in `mode="review"` — displays clean unwatermarked photo with high-res zoom controls, but commerce CTAs ("Buy now", "Add to cart") are suppressed.
- [ ] **TC-ADM-24: Resolve Dispute with Refund**
  - **Action:** Click "Resolve & Refund". Enter resolution summary note. Confirm.
  - **Expected Result:** Triggers backend refund call (`PaymongoRefundService`). Status updates to `RESOLVED`. Runner's order reflects refunded state, and dispute is moved to history.
- [ ] **TC-ADM-25: Deny Dispute Flow**
  - **Action:** Click "Deny Dispute". Enter denial rationale (e.g. "Photo matches bib number and watermark was clear before purchase"). Confirm.
  - **Expected Result:** Dispute status updates to `DENIED`. No refund is issued. Notification is dispatched to runner.
- [ ] **TC-ADM-26: Escalate Dispute Flow**
  - **Action:** Click "Escalate".
  - **Expected Result:** Tags dispute as `ESCALATED` for senior review. Escalation badge highlights on the queue card.

---

## Phase 7: Content Moderation & Flags Queue (`/admin/flags`)

- [ ] **TC-ADM-27: Flags Queue Rendering (`ADMIN_FLAGS_ENABLED = true`)**
  - **Action:** Navigate to `/admin/flags`.
  - **Expected Result:** Page renders flagged photos header ("Flags · X open · Y hidden"), search bar, and queue cards.
- [ ] **TC-ADM-28: Hide Contested Photo**
  - **Action:** Click "Hide from Gallery" on a flagged photo card. Confirm in the hide modal.
  - **Expected Result:** Photo status transitions to `HIDDEN` on the backend. It immediately disappears from public runner search and event galleries. Card shows hidden badge with undo/restore option.
- [ ] **TC-ADM-29: Dismiss Flag (Keep Live)**
  - **Action:** Click "Dismiss Flag" on a flagged photo. Enter explanation note.
  - **Expected Result:** Flag status moves to `DISMISSED`. Photo remains live in the public event gallery.
- [ ] **TC-ADM-30: Escalate Flag to Senior Admin**
  - **Action:** Click "Escalate" on a flag card.
  - **Expected Result:** Flag status flips to `ESCALATED`.

---

## Phase 8: Payouts & Financial Ledger (`/admin/payouts`)

- [ ] **TC-ADM-31: Payout Requests Queue**
  - **Action:** Navigate to `/admin/payouts`.
  - **Expected Result:** Displays table of photographer withdrawal requests with Photographer Name, Payment Method (GCash / Maya / Bank Transfer), Account Number, Requested Amount (₱), Fee Breakdown, and Net Amount.
- [ ] **TC-ADM-32: Single Payout Approval**
  - **Action:** Click "Approve" on an individual pending payout row.
  - **Expected Result:** Status changes to `APPROVED` (Ready to Send). Net amount is locked. Success toast appears.
- [ ] **TC-ADM-33: Hold Payout with Rationale**
  - **Action:** Click "Hold" on a suspicious or contested payout request. Enter reason (e.g. "Active copyright dispute under investigation").
  - **Expected Result:** Row flips to `ON_HOLD`. Photographer receives an inbox notice explaining the hold.
- [ ] **TC-ADM-34: Mark Payout as Paid**
  - **Action:** On an approved payout, click "Mark Paid". Enter payment reference / transaction number.
  - **Expected Result:** Payout transitions to `PAID`. Photographer wallet updates to deduct withdrawn balance.
- [ ] **TC-ADM-35: Bulk Approve Payouts Bar**
  - **Action:** Check multiple checkboxes on pending payout rows.
  - **Expected Result:** Fixed bottom floating action bar appears: `"3 requests selected · Total ₱4,500.00 · [Bulk Approve]"`. Clicking "Bulk Approve" approves all selected rows simultaneously.
- [ ] **TC-ADM-36: Payout Cycle Reports Generation**
  - **Action:** Scroll to the Payout Reports section. Inspect report cycles (e.g. 1st–15th or 16th–End of month).
  - **Expected Result:** Shows cycle summary, total disbursement, and exportable ledger data.

---

## Phase 9: Events Management (`/admin/events`)

- [ ] **TC-ADM-37: Events Roster Rendering**
  - **Action:** Navigate to `/admin/events`.
  - **Expected Result:** Renders all registered running events with Event Cover, Name, Race Date, Location, Total Photos Uploaded, and Current Status (Draft, Active, Archived).
- [ ] **TC-ADM-38: Create New Event Form**
  - **Action:** Click "Create Event" button.
  - **Expected Result:** Opens `admin-event-form-modal.tsx`. Fill out Name, Slug, Date, Location, and upload a Banner cover image. Submit form.
  - **Expected Result:** Event is created and immediately appears in the events list. Public URL `/events/[slug]` becomes accessible.
- [ ] **TC-ADM-39: Edit Existing Event**
  - **Action:** Click "Edit" on an existing event card. Update location or date. Save changes.
  - **Expected Result:** Changes persist to backend and reflect in both admin table and public runner event cockpit.
- [ ] **TC-ADM-40: Event State Override**
  - **Action:** Toggle event state between "Live", "Open", and "Archived".
  - **Expected Result:** Overrides take effect; archiving an event closes upload windows for photographers while keeping gallery readable.

---

## Phase 10: Photographers Roster & Messaging (`/admin/photographers`)

- [ ] **TC-ADM-41: Photographer Roster & Handle Search**
  - **Action:** Navigate to `/admin/photographers`. Search by handle or name.
  - **Expected Result:** Grid/table filters dynamically. Displays total covered events, total photos uploaded, lifetime earnings, and account status.
- [ ] **TC-ADM-42: Photographer Profile Inspector (`/admin/photographers/[handle]`)**
  - **Action:** Click on a photographer row/card.
  - **Expected Result:** Opens detailed overview showing bio, cover image, connected payout accounts, social links, and historical upload batches.
- [ ] **TC-ADM-43: Admin Direct Message (DM)**
  - **Action:** Click "Send Message" on the photographer profile. Enter subject and body (e.g. "Please check your watermark position on race photos"). Send.
  - **Expected Result:** Toast confirms dispatch. Log in as that photographer on another tab or phone: verify the message arrives in their notification bell modal with unread badge counter.

---

## Phase 11: Sales Analytics & Leaderboards (`/admin/sales`)

- [ ] **TC-ADM-44: Platform Sales Trend & Volume**
  - **Action:** Navigate to `/admin/sales`. Inspect the Gross Volume and Platform Take Rate charts.
  - **Expected Result:** Visual charts display transaction volume over 7d, 30d, and 90d intervals.
- [ ] **TC-ADM-45: Event Revenue Leaderboard**
  - **Action:** Check the Event Performance table.
  - **Expected Result:** Ranks events by total revenue, number of photo orders, and conversion rate. Sorting columns (by revenue, photo count) works properly.
- [ ] **TC-ADM-46: Top Photographers Leaderboard**
  - **Action:** Check the Top Photographers leaderboard slab.
  - **Expected Result:** Lists top-earning photographers, their upload volume, and runner purchase counts.

---

## Phase 12: Real-Time WebSocket Updates & Toast Sync

- [ ] **TC-ADM-47: WebSocket Auto-Connect on Admin Mount**
  - **Action:** Open browser DevTools Network tab -> filter `WS`. Navigate to any `/admin/*` page.
  - **Expected Result:** A persistent WebSocket handshake is established to `/ws/admin/notifications` using bearer token in `Sec-WebSocket-Protocol`.
- [ ] **TC-ADM-48: Live Dispute Ingestion (Dual-Tab Test)**
  - **Action:** Keep Admin tab open on `/admin/disputes`. In a separate runner tab, file a dispute on an order.
  - **Expected Result:** The Admin tab receives a `"dispute.created"` WebSocket push frame. A top-right toast alerts: "New dispute received". The dispute queue adds the new dispute card without requiring a manual page refresh.
- [ ] **TC-ADM-49: Live Payout Request Ingestion**
  - **Action:** On a photographer tab, submit a payout request.
  - **Expected Result:** Admin tab immediately updates the Payouts KPI counter and inserts the row into `/admin/payouts` in real time.
- [ ] **TC-ADM-50: Network Disconnect & Graceful Reconnect**
  - **Action:** Simulate network offline via DevTools for 5 seconds, then toggle back to online.
  - **Expected Result:** WS reconnect backoff triggers automatically. Upon reconnect, an automatic REST refetch runs to catch any decisions or updates made during the disconnect gap.
