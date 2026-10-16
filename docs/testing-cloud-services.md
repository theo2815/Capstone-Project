# Testing the cloud services

How to test the parts of QuickPitik that run on, or call, outside services (AWS Rekognition, Cloudflare R2, Railway, Vercel, PayMongo). For agents and teammates. **Read the rules and the environment map before running anything.**

## Rules that always apply

1. **Ask the owner first** for anything that uses real keys, costs money, or touches staging or production. Approval for one run does not cover the next.
2. **Never test against production resources** (`quickpitik-prod`, the production database, live PayMongo) unless the owner names that exact check.
3. **Never print, copy, or commit keys.** Read them in memory. Key values live only in `docs/api-keys.md` and `backend/src/main/resources/application-local.yml` (both gitignored).
4. **A skipped test is not a pass.** Report it as skipped.
5. **Check cleanup yourself**, and say what you did not check.

## Environment map

| | Local | Staging | Production |
|---|---|---|---|
| Backend | your machine, `localhost:8080` | Railway project `quickpitik-backend`, environment `staging`, service `staging-backend` | same project, environment `production` |
| Deploys from | — | **every push to `dev-main` that changes `backend/**`** | pushes to `main` |
| Database | Docker Postgres, or a throwaway test container | its own Railway Postgres | its own Railway Postgres |
| Photo bucket | `quickpitik-dev` (the `application-local.yml` template since 2026-10-16) — a locally run backend stores photos there. **Older copies say `quickpitik-prod`: check yours before any upload.** Tests use `quickpitik-dev` (section 1) | `quickpitik-dev` | `quickpitik-prod` |
| Face / bib | off by default (`AI_API_ENABLED=false`); the smoke tests in section 1 call Rekognition directly with the local keys | **off** (`AI_API_ENABLED=false`) | Rekognition |
| Payments | PayMongo test key | PayMongo test key | PayMongo live |
| Email | Resend sandbox sender | Resend sandbox sender (delivers to the account owner only) | `noreply@quickpitik.com` |
| Website | `localhost:3000` | Vercel Preview of `dev-main` (behind Vercel login) | `quickpitik.com` |

Last read from the live settings on 2026-10-09. The live settings win over this table; section 2 shows how to read them.

**A `dev-main` push that changes `backend/**` is a staging deploy.** It also runs any new database migration on the staging database, and migrations cannot be undone. Say so in your handoff; never write "nothing deployed" after a push that changed `backend/`.

## 1. AWS Rekognition and R2 (run from your machine)

### Which bucket the tests use

`application-local.yml` may name any bucket (older copies say `quickpitik-prod`), so the tests never read the bucket from it. Both R2 tests take the bucket from the `R2_SMOKE_BUCKET` variable: without it they skip, and a name containing "prod" is refused. Do not edit the config file to run tests.

The test bucket is `quickpitik-dev`. It is also the staging bucket: the smoke tests write only under `smoke-test/` and delete what they create. Never delete anything else in it.

### Running the whole integration suite in the main checkout

A plain `gradlew integrationTest` in a checkout that has `application-local.yml` still **calls real AWS Rekognition** with the local keys (checks 1, 3 and 4: a few billed calls, test collections created and deleted). The two R2 checks skip. So the whole suite counts as a cloud run: ask the owner first, and report the R2 checks as skipped unless you set the variables of step 2. In a worktree without the config file, all five skip.

The backend rule "`integrationTest` is never skipped" is about starting Docker. It does not replace the owner's approval for the cloud checks.

### The five checks

| # | Test class | Check | Touches |
|---|---|---|---|
| 1 | `RekognitionCredentialsSmokeTest` | Keys work: create and delete a collection | AWS |
| 2 | `RekognitionR2E2eSmokeTest` | Put, get, signed link, download, delete | R2 |
| 3 | `RekognitionR2E2eSmokeTest` | Enroll a selfie, find it with a different selfie | AWS |
| 4 | `RekognitionR2E2eSmokeTest` | Read text on a race photo | AWS |
| 5 | `R2DirectUploadSmokeTest` | Direct upload rules, including a real Chrome upload | R2 |

All are tagged `integration` and live under `backend/src/test/kotlin/com/quickpitik/service/`. None of them needs Docker or Postgres.

### What you need

- JDK 21, the AWS CLI, and Google Chrome in `Program Files`.
- `backend/.storage/selfies/<id>/` with at least two JPGs of one person, and at least one `backend/.storage/events/*/photos/*/original.jpg`. Without them checks 3 and 4 skip.
- Port 3000 free. Check 5 serves a page on `http://localhost:3000`, the origin the dev bucket's upload rules allow. Stop the website dev server first.
- No proxy variables (`HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`) in the process that runs the tests.

### Steps

**1. Preflight (read-only).**
- Keys reach the AWS CLI only as `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` in the child process's environment, set from a script that reads the config. Never type them on a command line or echo them. Rekognition keys are `app.ai.rekognition.access-key` / `secret-key`; R2 keys are `app.storage.access-key` / `secret-key`.
- `aws sts get-caller-identity --region ap-southeast-1` with the Rekognition keys should show the `quickpitik-backend` user.
- `aws s3api head-bucket --bucket quickpitik-dev --endpoint-url <app.storage.endpoint> --region auto` with the R2 keys should succeed.
- List `smoke-test/` in `quickpitik-dev` and keep the list as your baseline (normally empty).

**2. Set these variables for the test process only** (check 2 needs the first; check 5 needs all of them; without them they skip):

| Variable | Value |
|---|---|
| `R2_SMOKE_BUCKET` | `quickpitik-dev` |
| `R2_SMOKE_ENDPOINT` | `app.storage.endpoint` from the config |
| `R2_SMOKE_ACCESS_KEY` / `R2_SMOKE_SECRET_KEY` | `app.storage` keys from the config |
| `R2_SMOKE_ORIGIN` | `http://localhost:3000` |

**3. Run, from `backend/`:**

```
gradlew.bat integrationTest --rerun --tests "*RekognitionCredentialsSmokeTest" --tests "*RekognitionR2E2eSmokeTest" --tests "*R2DirectUploadSmokeTest" --max-workers=1 --console=plain
```

`--rerun` matters: without it Gradle may print `BUILD SUCCESSFUL` without running anything.

**4. Read the results from the XML, not the exit code.**
Open `backend/build/test-results/integrationTest/TEST-*.xml`. Check the files were written by this run. Expect 6 tests, 0 failures, 0 skipped (the sixth is a local signing check).

**5. Verify cleanup (read-only).**
- `aws rekognition describe-collection` on each `qp-smoke-test-…` and `qp-event-…` name printed in the test output should return `ResourceNotFoundException`.
- `smoke-test/` in `quickpitik-dev` should match your baseline.

### Looking at R2 by hand (`cf` CLI)

For ad-hoc checks (list buckets or objects, read bucket settings) use Cloudflare's `cf` CLI (`npm i -g cf`, open beta, replaces Wrangler). Nothing in this repo needs Wrangler.

- Both buckets live on Cloudflare account `277f145551e9715f57bc8e8f70182d5d`. If you have more than one Cloudflare login, make a profile for this account and bind it to your checkout: `cf auth create quickpitik`, then `cf auth activate quickpitik <checkout path>`.
- **Run `cf auth whoami` from the checkout first** and confirm that account ID. Stop if it shows another one.
- List objects: `cf r2 objects list --bucket-name quickpitik-dev` (page with `--cursor`). Bucket-info object counts and sizes lag a live listing.
- Find other commands with `cf cli search "<task>"`; preview changes with `--dry-run`. Writes and deletes on `quickpitik-prod` follow rule 2.
- A delete run without a terminal and without `--force` prints `Aborted.` and still exits 0. Exit 0 does not mean it was deleted.
- `cf` caches the chosen account in `.cloudflare/` (gitignored).

### Known problems

| Symptom | Cause | What to do |
|---|---|---|
| `BindException: Address already in use` | Something holds port 3000 | Free the port, rerun check 5 only |
| Tests skip in a worktree | No `application-local.yml` or `.storage` there | Run from the main checkout |
| STS or R2 call fails to connect | Proxy variable inherited by the shell | Remove it for the child process only |
| `GetBucketCors` returns AccessDenied | The key cannot read bucket settings | Expected; check 5 proves the upload rules instead |

## 2. Backend staging (Railway)

**Setup.** The Railway link inside `backend/` may be stale. Link an empty scratch folder instead and run the commands there:

```
railway link --project quickpitik-backend --environment staging --service staging-backend
```

**After a push to `dev-main`, check the deploy (read-only):**

- `railway deployment list --limit 3` should show `SUCCESS` for your commit. `SKIPPED` means the push did not change `backend/`.
- `railway logs --deployment <id>` should show Flyway validate the migrations, apply only the ones you expect, then `Started QuickPitikApplicationKt`.
- Never run `railway variables` without filtering its output; it prints every secret. To read the two values the smoke requests need:

```
railway variables --json | python -c "import sys,json; d=json.load(sys.stdin); print(d['RAILWAY_PUBLIC_DOMAIN']); print(d['CORS_ALLOWED_ORIGINS'])"
```

**Smoke requests** (base URL: `https://<RAILWAY_PUBLIC_DOMAIN>`; all passed on 2026-10-09):

| Request | Expect |
|---|---|
| `GET /actuator/health` | 200 |
| `GET /api/v1/events` | 200 |
| `POST /api/v1/payments/webhook/paymongo` with no signature | 401 |
| `GET /v3/api-docs` | 404 |
| `GET /actuator/metrics` without login | 401 |
| CORS preflight from the Vercel Preview origin | 200 |
| CORS preflight from `https://evil.example` | 403 |

CORS preflight: `curl -s -o NUL -w "%{http_code}" -X OPTIONS -H "Origin: <origin>" -H "Access-Control-Request-Method: GET" https://<domain>/api/v1/events`

**Deeper staging tests** (uploads, orders, admin actions) create real rows and files. They need the owner's approval, fresh test accounts, and cleanup through the app's own delete paths.

## 3. Website preview (Vercel)

- The `dev-main` Preview talks to backend staging. Its URL is the first entry of staging's `CORS_ALLOWED_ORIGINS`.
- It is behind Vercel login: a request without login redirects (302). The owner must sign in for browser tests.
- If the preview cannot reach the API, check that its origin is in staging's `CORS_ALLOWED_ORIGINS`.

## 4. Production

Checklist and the origin-protection checks: vault `backend/notes/production-deploy.md`. Do nothing on production without the owner's approval for that exact step. Read `flyway_schema_history` before any deploy that carries a migration.

**Database migrations on staging or production.** Flyway applies them at boot and they cannot be undone, so rehearse first on a throwaway database: a clean install, and an upgrade from the version the target is on now. A worked example with rollout and rollback order: vault `_journal/2026-10-08-admin-actions-and-event-location-plan.md` § Session 11.

## 5. ai-api blur service

A separate Railway project. Release check, smoke script and rollback signs: `ai-api/docs/deployment.md` § Railway operations and release check.

## 6. Payments (PayMongo)

Local: `docs/testing-payments-locally.md`. **Never scan a test QR code with a real bank app; it moves real money.** Staging uses a test key and a test webhook; a full paid-order run on staging has no written routine yet.

## 7. Not covered by any automated test

- **CI does not test AWS or R2.** GitHub Actions has no keys, so the five checks skip there. A green CI run says nothing about the cloud services.
- Resend email delivery, Google sign-in, and live WebSocket delivery have no test routine. Check them by hand when a change touches them, and report them as unchecked otherwise.

## What to report

The commit and environment tested, each check's result, whether a staging deploy or migration happened, cleanup results, and what you did not check.
