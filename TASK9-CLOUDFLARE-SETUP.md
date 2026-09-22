# TASK 9 — Cloudflare API origin rule

Railway already has `ORIGIN_SECRET`, and its `/actuator/health` healthcheck is configured. The guarded backend has **not** been deployed yet. Your step is to add the matching Cloudflare rule.

1. Open [`docs/api-keys.md`](docs/api-keys.md) on this computer. Find **“Origin protection (TASK 9, 2026-09-23)”** and copy only the characters after `ORIGIN_SECRET=`. Do not copy the label, add quotes, or paste the value into chat.
2. Sign in to Cloudflare and select the **quickpitik.com** zone. Under **DNS**, confirm the `api` record says **Proxied** (orange cloud).
3. Open **Rules → Overview → Create rule → Request Header Transform Rule**. Name it `QuickPitik API origin secret`.
4. Under **When incoming requests match**, choose **Custom filter expression**. Set **Hostname** **equals** `api.quickpitik.com`. The expression should read `http.host eq "api.quickpitik.com"`. Do not add a path or method condition.
5. Under **Modify request header**, choose **Set static**. Set **Header name** to `X-Origin-Secret` and **Value** to the value from step 1. Choose **Deploy**. Check that no later request-header rule removes or changes `X-Origin-Secret` for this hostname.
6. Tell Codex **“Cloudflare rule deployed”** without sending the secret. Codex will check public traffic, deploy the guarded backend, then test direct Railway access and the public API.

`docs/api-keys.md` is already listed in the repo's `.gitignore`; you do not need to change Git settings. This guide contains no secret and can be committed. Cloudflare's [dashboard instructions](https://developers.cloudflare.com/rules/transform/request-header-modification/create-dashboard/) confirm that **Set static** replaces an existing client-supplied header.
