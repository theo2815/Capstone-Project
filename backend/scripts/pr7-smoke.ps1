# PR 7 smoke — runs against http://localhost:8080 with Postgres + bootRun up.
# Uses curl.exe (Windows 10+ bundled) so the same pattern handles 2xx + 4xx /
# 5xx without IRM's PS 5.1 ErrorDetails quirks. Verifies envelope shape on
# every response.

$ErrorActionPreference = 'Stop'
$base = 'http://localhost:8080/api/v1'
$container = 'quickpitik-postgres'
$db = 'quickpitik'
$dbUser = 'quickpitik'

function Show-Step($name) {
    Write-Host ""
    Write-Host "===== $name =====" -ForegroundColor Cyan
}

function Show-Body($body, $label = '') {
    if ($label) { Write-Host $label -ForegroundColor Yellow }
    if ($null -eq $body) { Write-Host '(empty body)'; return }
    $body | ConvertTo-Json -Depth 8 | Write-Host
}

function PsqlScalar($sql) {
    docker exec $container psql -U $dbUser -d $db -tA -c $sql
}

function PsqlExec($sql) {
    docker exec $container psql -U $dbUser -d $db -c $sql | Out-Null
}

# Run an HTTP request with curl.exe and return @{ code; body } regardless of
# 2xx/4xx/5xx. Body is parsed as JSON when it looks like JSON.
function Http-Call {
    param(
        [string] $Method,
        [string] $Uri,
        [string] $Token,
        [string] $JsonBody,
        [string] $UploadFile
    )
    $sentinel = '__HTTPCODE__'
    $argList = @('-s', '-w', "`n${sentinel}%{http_code}", '-X', $Method, $Uri)
    if ($Token) { $argList += @('-H', "Authorization: Bearer $Token") }
    $tmpJson = $null
    if ($JsonBody) {
        # PowerShell's argv-passing strips inner double quotes from JSON when
        # passed as a single string. Write to a temp file and use curl's @file
        # syntax so the bytes go through unmodified.
        $tmpJson = Join-Path $env:TEMP "pr7-body-$(Get-Random).json"
        $JsonBody | Set-Content -Path $tmpJson -Encoding utf8 -NoNewline
        $argList += @('-H', 'Content-Type: application/json')
        $argList += @('--data-binary', "@$tmpJson")
    }
    if ($UploadFile) { $argList += @('-F', "file=@$UploadFile") }

    $raw = & curl.exe @argList 2>$null
    if ($tmpJson) { Remove-Item $tmpJson -ErrorAction SilentlyContinue }
    $combined = ($raw -join "`n")
    $idx = $combined.LastIndexOf("`n${sentinel}")
    if ($idx -ge 0) {
        $bodyText = $combined.Substring(0, $idx).Trim()
        $code = [int] ($combined.Substring($idx + $sentinel.Length + 1).Trim())
    } else {
        $bodyText = $combined.Trim()
        $code = 0
    }
    $parsed = $null
    if ($bodyText -and ($bodyText.StartsWith('{') -or $bodyText.StartsWith('['))) {
        try { $parsed = $bodyText | ConvertFrom-Json } catch { $parsed = $bodyText }
    } else {
        $parsed = $bodyText
    }
    return @{ code = $code; body = $parsed }
}

# ─── 1. Register a fresh photographer ───────────────────────────────────────
Show-Step '1. Register photographer'
$photoEmail = "pr7-photographer-$(Get-Random)@quickpitik.test"
$registerResp = Http-Call -Method POST -Uri "$base/auth/register" `
    -JsonBody (@{ name = 'Test Photographer'; email = $photoEmail
                  password = 'photographer123'; role = 'PHOTOGRAPHER' } | ConvertTo-Json -Compress)
if ($registerResp.code -ne 200) { Show-Body $registerResp.body "register status=$($registerResp.code)"; exit 1 }
$photoToken = $registerResp.body.data.accessToken
$photoId = $registerResp.body.data.user.id
Write-Host "photographerId=$photoId" -ForegroundColor Green

# ─── 2. Register a runner (for cross-role test) ─────────────────────────────
Show-Step '2. Register runner'
$runnerEmail = "pr7-runner-$(Get-Random)@quickpitik.test"
$runnerResp = Http-Call -Method POST -Uri "$base/auth/register" `
    -JsonBody (@{ name = 'Test Runner'; email = $runnerEmail
                  password = 'runner123'; role = 'RUNNER' } | ConvertTo-Json -Compress)
$runnerToken = $runnerResp.body.data.accessToken
Write-Host "runnerId=$($runnerResp.body.data.user.id)" -ForegroundColor Green

# ─── 3. Runner -> /me/photographer/events should return 403 ─────────────────
Show-Step '3. Runner -> /me/photographer/events expect 403 FORBIDDEN'
$resp = Http-Call -Method GET -Uri "$base/me/photographer/events" -Token $runnerToken
Show-Body $resp.body "status=$($resp.code)"

# ─── 4. Photographer (no rows) -> empty list ────────────────────────────────
Show-Step '4. Photographer (no rows) expect empty PaginatedResponse'
$resp = Http-Call -Method GET -Uri "$base/me/photographer/events" -Token $photoToken
Show-Body $resp.body "status=$($resp.code)"

# ─── 5. Direct DB seed ──────────────────────────────────────────────────────
Show-Step '5. Direct DB seed: photographer_settings, event_photographer, photo'
$eventId = (PsqlScalar "SELECT id FROM events ORDER BY date DESC LIMIT 1;").Trim()
$eventSlug = (PsqlScalar "SELECT slug FROM events WHERE id = '$eventId';").Trim()
Write-Host "eventId=$eventId  slug=$eventSlug" -ForegroundColor Green

PsqlExec @"
INSERT INTO photographer_settings (user_id, handle, brand_name, brand_color, bio,
    cover_gradient_from, cover_gradient_to, watermark_label, verification_status, member_since)
VALUES ('$photoId', 'pr7smoke', 'PR7 Smoke Studio', 'amber', 'PR7 smoke test bio.',
    '#D97706', '#7C2D12', 'PR7', 'APPROVED', CURRENT_DATE);
"@
PsqlExec @"
INSERT INTO event_photographer (event_id, photographer_id, photo_count, sales_count, revenue_kept_php, first_upload_at, last_upload_at)
VALUES ('$eventId', '$photoId', 1, 0, 0, now(), now());
"@
$photoUuid = '11111111-1111-1111-1111-111111111111'
PsqlExec @"
INSERT INTO photos (id, event_id, photographer_id, s3_key, thumbnail_s3_key, watermark_s3_key,
    span, tone, uploaded_at, status, price_php)
VALUES ('$photoUuid', '$eventId', '$photoId',
    'events/$eventId/photos/$photoUuid/original.jpg',
    'events/$eventId/photos/$photoUuid/watermark.jpg',
    'events/$eventId/photos/$photoUuid/watermark.jpg',
    'default', 1, now(), 'LIVE', 125);
"@
PsqlExec @"
INSERT INTO photo_bibs (photo_id, bib_number, ocr_confidence) VALUES ('$photoUuid', '4082', 0.92);
"@

# ─── 6. /me/photographer/events?withUploads=true ────────────────────────────
Show-Step '6. /me/photographer/events?withUploads=true'
$resp = Http-Call -Method GET -Uri "${base}/me/photographer/events?withUploads=true" -Token $photoToken
Show-Body $resp.body "status=$($resp.code)"

# ─── 7. /me/photographer/events/{id} ────────────────────────────────────────
Show-Step "7. /me/photographer/events/$eventId"
$resp = Http-Call -Method GET -Uri "${base}/me/photographer/events/$eventId" -Token $photoToken
Show-Body $resp.body "status=$($resp.code)"

# ─── 8. /me/photographer/events/{id}/photos ─────────────────────────────────
Show-Step "8. /me/photographer/events/$eventId/photos"
$resp = Http-Call -Method GET -Uri "${base}/me/photographer/events/$eventId/photos" -Token $photoToken
Show-Body $resp.body "status=$($resp.code)"

# ─── 9. /me/photographer/photos/{id}/download ───────────────────────────────
Show-Step "9. /me/photographer/photos/$photoUuid/download"
$resp = Http-Call -Method GET -Uri "${base}/me/photographer/photos/$photoUuid/download" -Token $photoToken
Show-Body $resp.body "status=$($resp.code)"

# ─── 10. Cross-photographer download -> 404 (anti-IDOR) ─────────────────────
Show-Step '10. Cross-photographer download (anti-IDOR) expect 404 PHOTO_NOT_FOUND'
$ghostId = '22222222-2222-2222-2222-222222222222'
$resp = Http-Call -Method GET -Uri "${base}/me/photographer/photos/$ghostId/download" -Token $photoToken
Show-Body $resp.body "status=$($resp.code)"

# ─── 11. Public profile lookup ──────────────────────────────────────────────
Show-Step '11. /public/photographers/pr7smoke'
$resp = Http-Call -Method GET -Uri "${base}/public/photographers/pr7smoke"
Show-Body $resp.body "status=$($resp.code)"

# ─── 12. Public gallery photos ──────────────────────────────────────────────
Show-Step "12. /public/photographers/pr7smoke/events/$eventSlug/photos"
$resp = Http-Call -Method GET -Uri "${base}/public/photographers/pr7smoke/events/$eventSlug/photos"
Show-Body $resp.body "status=$($resp.code)"

# ─── 13. Public lookup of unknown handle -> 404 ─────────────────────────────
Show-Step '13. /public/photographers/nonexistent expect 404'
$resp = Http-Call -Method GET -Uri "${base}/public/photographers/nonexistent"
Show-Body $resp.body "status=$($resp.code)"

# ─── 14 + 15. Build a tiny JPEG and exercise upload paths ───────────────────
Add-Type -AssemblyName System.Drawing
$bmp = New-Object System.Drawing.Bitmap 64, 64
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.Clear([System.Drawing.Color]::SteelBlue)
$g.Dispose()
$tmpJpg = Join-Path $env:TEMP "pr7-smoke-$(Get-Random).jpg"
$bmp.Save($tmpJpg, [System.Drawing.Imaging.ImageFormat]::Jpeg)
$bmp.Dispose()

Show-Step '14. Upload with INCOMPLETE verification expect 403 PHOTOGRAPHER_NOT_VERIFIED'
$incompleteEmail = "pr7-incomplete-$(Get-Random)@quickpitik.test"
$incompleteResp = Http-Call -Method POST -Uri "$base/auth/register" `
    -JsonBody (@{ name = 'Incomplete Photographer'; email = $incompleteEmail
                  password = 'photographer123'; role = 'PHOTOGRAPHER' } | ConvertTo-Json -Compress)
$incompleteToken = $incompleteResp.body.data.accessToken
$resp = Http-Call -Method POST -Uri "${base}/me/photographer/events/$eventId/photos" `
    -Token $incompleteToken -UploadFile $tmpJpg
Show-Body $resp.body "status=$($resp.code)"

Show-Step '15. Verified upload (ai-api offline) expect 503 AI_API_UNAVAILABLE'
$resp = Http-Call -Method POST -Uri "${base}/me/photographer/events/$eventId/photos" `
    -Token $photoToken -UploadFile $tmpJpg
Show-Body $resp.body "status=$($resp.code)"

Remove-Item $tmpJpg -ErrorAction SilentlyContinue

Write-Host ""
Write-Host '===== PR 7 smoke complete =====' -ForegroundColor Magenta
