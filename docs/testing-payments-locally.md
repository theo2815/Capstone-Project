# Testing QR Ph payments locally

How to walk the whole checkout payment lifecycle on a dev machine without moving real money.

## The one thing to know

PayMongo test mode still generates **real, scannable QR Ph codes**. Scanning one with a bank or e-wallet app processes a real transaction, even on an `sk_test_` key. PayMongo's own guidance is to never scan them and to use the `test_url` it returns instead. The website surfaces that link as the **Test mode · PayMongo sandbox** panel on the Scan-to-pay step, which is the only supported way to simulate a payment.

## Setup

| What | Where | Value |
|---|---|---|
| PayMongo secret key | `backend/src/main/resources/application-local.yml` (copy `application-local.example.yml`) | must start with `sk_test_` |
| QR lifetime | env `PAYMONGO_CHECKOUT_TTL` | default `PT30M`; use `PT1M` to test expiry |
| Reconciler sweep | `app.payments.paymongo.reconcile-interval-ms` | default 60000 |
| Website API base | `website/.env.local` `NEXT_PUBLIC_API_URL` | your local backend |

The test panel appears only when the backend returns a `testUrl`, and the backend returns it only when its secret key starts with `sk_test_`. A live key never emits it, so nothing here can reach production.

Webhooks are optional locally. PayMongo cannot call `localhost`, but the checkout's "I've paid" button sends `?verify=true`, which asks PayMongo directly and settles or expires the order on the spot. The reconciler does the same every minute for anything the browser missed.

## Reaching each state

Start any run with photos in the cart, open checkout, and generate a QR.

| State | How to get there | What you should see |
|---|---|---|
| Pending / waiting | Generate the QR and do nothing. | Green "Waiting for your payment" notice, countdown under the QR, **I've paid** and **Cancel payment** actions. |
| Confirming | Tap **I've paid** without simulating anything. | "Confirming your payment…" with an elapsed clock; after 60 s the amber "Taking longer than usual" tier and **I'll wait for the email →**. |
| Successful | Open **PayMongo simulator ↗**, authorize the test payment, return, tap **I've paid**. | "All yours." with the paid time, the reference, and **View receipt & download →**. Cart and pending record are cleared. |
| Failed | In the simulator choose the failure action if your account's simulator offers one, return, tap **I've paid**. | Back on Review & pay with the red "Payment didn't go through" notice and **Generate a new QR**. If the simulator has no failure action, this path is covered by unit tests only (`PaymongoCheckoutReconcilerTest`, "last attachment failed"). |
| Expired | Restart the backend with `PAYMONGO_CHECKOUT_TTL=PT1M`, generate a QR, wait. | The countdown turns amber under 5 minutes and hits 0:00; the automatic check at expiry brings you back to Review & pay with the neutral "Your QR code expired" notice. |
| Cancelled | Tap **Cancel payment**, then confirm **Cancel payment** in the dialog (**Keep waiting** does nothing). | Back on Review & pay with "Payment cancelled"; the cart is unchanged. The backend marks the order EXPIRED exactly like a timeout. |
| Cancel loses the race | Authorize in the simulator first, then tap **Cancel payment** and confirm. | "All yours." instead of "Payment cancelled" — the backend asks PayMongo before it cancels. |
| Leave and come back | Close the drawer during a live QR, choose **Leave**. | Cart pill reads "Payment pending"; tapping it reopens the same QR. Authorizing in the simulator while the drawer is closed produces a "Payment confirmed" toast within about 5 s. |
| Refresh mid-QR | Reload the page with a QR on screen. | The drawer reopens on the same QR with the test panel intact. |

## Where the pieces live

- Backend: `OrderService.createQrPhPayment` (test URL passthrough), `OrderService.cancel*` + `PaymongoCheckoutReconciler` (cancel, verify, failed-intent detection), `PaymongoWebhookService` (webhook path when one is configured).
- Website: `components/cart/checkout-modal.tsx` (state machine and panel), `lib/api-orders.ts` (`cancelPendingPayment`, `classifyExpired`), `store/pending-payment-store.ts` (persisted record).
- Tests: `website/src/components/cart/checkout-modal.test.tsx`, `backend/.../OrderServiceCheckoutTest.kt`, `PaymongoCheckoutReconcilerTest.kt`.
