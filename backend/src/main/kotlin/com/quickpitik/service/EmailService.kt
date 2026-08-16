package com.quickpitik.service

import com.quickpitik.config.ResendProperties
import com.quickpitik.dto.email.ResendSendEmailRequest
import com.quickpitik.service.email.ResendClient
import org.slf4j.LoggerFactory
import org.springframework.beans.factory.annotation.Value
import org.springframework.stereotype.Service
import java.net.URLEncoder
import java.nio.charset.StandardCharsets

@Service
class EmailService(
    private val resendClient: ResendClient,
    private val resendProperties: ResendProperties,
    @Value("\${app.cors.allowed-origins:http://localhost:3000}") private val frontendOriginsCsv: String,
) {
    private val log = LoggerFactory.getLogger(javaClass)
    private val frontendOrigin: String =
        frontendOriginsCsv.split(",").firstOrNull()?.trim().orEmpty().ifBlank { "http://localhost:3000" }

    // Dev-mode detection: when the API key is still the in-repo placeholder
    // we skip the Resend call (which would 401) and just log the reset link
    // so local testing has a usable affordance. Production must set
    // RESEND_API_KEY to a real `re_…` key.
    private val devMode: Boolean = resendProperties.apiKey.startsWith("re_dev-only")

    fun sendPasswordResetEmail(toEmail: String, resetToken: String) {
        val resetUrl = "$frontendOrigin/reset-password?token=" +
            URLEncoder.encode(resetToken, StandardCharsets.UTF_8)

        if (devMode) {
            log.info("[EMAIL STUB] password reset for {} — link: {}", toEmail, resetUrl)
            return
        }

        val sender = "${resendProperties.fromName} <${resendProperties.fromAddress}>"
        try {
            val response = resendClient.send(
                ResendSendEmailRequest(
                    from = sender,
                    to = listOf(toEmail),
                    subject = "Reset your QuickPitik password",
                    html = renderHtml(toEmail, resetUrl),
                ),
            )
            log.info("Password reset email sent · to={} resendId={}", toEmail, response.id)
        } catch (ex: Exception) {
            // PasswordResetService.requestReset is anti-enumeration silent —
            // bubbling this up would roll back the saved token and leak
            // delivery state to the caller via the 5xx. Log loudly, return
            // quietly. Manual reprocessing (or a future resend endpoint) can
            // retry against the existing token row.
            log.error("Password reset email failed · to={} err={}", toEmail, ex.message, ex)
        }
    }

    // Goes to the NEW address, never the old one — receiving it is the proof
    // that the requester actually controls that inbox. Unlike the reset mail,
    // a delivery failure here MUST be visible: EmailChangeService is not
    // anti-enumeration silent (the caller already proved who they are with
    // their password), and silently swallowing this would leave the user
    // waiting for a confirmation that never comes.
    fun sendEmailChangeConfirmation(toEmail: String, changeToken: String) {
        val confirmUrl = "$frontendOrigin/confirm-email-change?token=" +
            URLEncoder.encode(changeToken, StandardCharsets.UTF_8)

        if (devMode) {
            log.info("[EMAIL STUB] email change for {} — link: {}", toEmail, confirmUrl)
            return
        }

        val sender = "${resendProperties.fromName} <${resendProperties.fromAddress}>"
        val response = resendClient.send(
            ResendSendEmailRequest(
                from = sender,
                to = listOf(toEmail),
                subject = "Confirm your new QuickPitik email",
                html = renderEmailChangeHtml(toEmail, confirmUrl),
            ),
        )
        log.info("Email-change confirmation sent · to={} resendId={}", toEmail, response.id)
    }

    // Sent AFTER_COMMIT of a registration (EmailVerificationListener), never
    // inline. Failures are swallowed the way the password-reset mail's are, for
    // a stronger reason: verification is advisory, so a lost mail costs the user
    // a flag, not access. `POST /auth/resend-verification` is the retry.
    fun sendEmailVerification(toEmail: String, verificationToken: String) {
        val verifyUrl = "$frontendOrigin/verify-email?token=" +
            URLEncoder.encode(verificationToken, StandardCharsets.UTF_8)

        if (devMode) {
            log.info("[EMAIL STUB] verification for {} — link: {}", toEmail, verifyUrl)
            return
        }

        val sender = "${resendProperties.fromName} <${resendProperties.fromAddress}>"
        try {
            val response = resendClient.send(
                ResendSendEmailRequest(
                    from = sender,
                    to = listOf(toEmail),
                    subject = "Confirm your QuickPitik email",
                    html = renderVerificationHtml(toEmail, verifyUrl),
                ),
            )
            log.info("Verification email sent · to={} resendId={}", toEmail, response.id)
        } catch (ex: Exception) {
            log.error("Verification email failed · to={} err={}", toEmail, ex.message, ex)
        }
    }

    private fun renderVerificationHtml(toEmail: String, verifyUrl: String): String {
        return """
<!DOCTYPE html>
<html><head><meta charset="utf-8" /><title>Confirm your QuickPitik email</title></head>
<body style="margin:0;padding:0;background:#f7f5ee;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;color:#1a1a1a;">
  <table cellspacing="0" cellpadding="0" border="0" width="100%" style="background:#f7f5ee;padding:32px 16px;">
    <tr><td align="center">
      <table cellspacing="0" cellpadding="0" border="0" width="100%" style="max-width:520px;margin:0 auto;background:#fafaf6;border-radius:16px;padding:36px 32px;border:1px solid #e5e2d8;">
        <tr><td>
          <p style="font-family:ui-monospace,'SFMono-Regular',Menlo,Consolas,monospace;text-transform:uppercase;letter-spacing:0.3em;font-size:11px;color:#7a7a7a;margin:0 0 10px;">
            QuickPitik · Welcome
          </p>
          <h1 style="font-family:Georgia,'Times New Roman',serif;font-size:32px;font-weight:500;margin:0 0 18px;letter-spacing:-0.01em;line-height:1.05;color:#1a1a1a;">
            Confirm your email.
          </h1>
          <p style="font-size:15px;line-height:1.6;color:#3a3a3a;margin:0 0 28px;">
            Thanks for joining QuickPitik. Tap below to confirm <strong>${escapeHtml(toEmail)}</strong> —
            the link works for 24 hours. Your account is already active either way; confirming just
            lets us reach you about your photos and orders.
          </p>

          <table cellspacing="0" cellpadding="0" border="0" style="margin:0 0 28px;">
            <tr><td style="padding:6px 0;">
              <a href="${escapeUrl(verifyUrl)}"
                 style="display:inline-block;background:#3b8c5f;color:#fafaf6;
                        text-decoration:none;padding:16px 32px;border-radius:999px;
                        font-family:ui-monospace,'SFMono-Regular',Menlo,Consolas,monospace;
                        text-transform:uppercase;letter-spacing:0.2em;font-size:13px;
                        font-weight:500;">
                Confirm my email  &rarr;
              </a>
            </td></tr>
          </table>

          <hr style="border:none;border-top:1px solid #e5e2d8;margin:28px 0;" />

          <p style="font-size:13px;line-height:1.65;color:#7a7a7a;margin:0;">
            If you didn't sign up for QuickPitik, ignore this email — no account action is needed.
          </p>
        </td></tr>
      </table>
      <p style="text-align:center;font-family:ui-monospace,'SFMono-Regular',Menlo,Consolas,monospace;text-transform:uppercase;letter-spacing:0.3em;font-size:10px;color:#a8a8a8;margin:24px 0 0;">
        QuickPitik · Cebu, Philippines
      </p>
    </td></tr>
  </table>
</body></html>
        """.trimIndent()
    }

    private fun renderEmailChangeHtml(toEmail: String, confirmUrl: String): String {
        return """
<!DOCTYPE html>
<html><head><meta charset="utf-8" /><title>Confirm your new QuickPitik email</title></head>
<body style="margin:0;padding:0;background:#f7f5ee;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;color:#1a1a1a;">
  <table cellspacing="0" cellpadding="0" border="0" width="100%" style="background:#f7f5ee;padding:32px 16px;">
    <tr><td align="center">
      <table cellspacing="0" cellpadding="0" border="0" width="100%" style="max-width:520px;margin:0 auto;background:#fafaf6;border-radius:16px;padding:36px 32px;border:1px solid #e5e2d8;">
        <tr><td>
          <p style="font-family:ui-monospace,'SFMono-Regular',Menlo,Consolas,monospace;text-transform:uppercase;letter-spacing:0.3em;font-size:11px;color:#7a7a7a;margin:0 0 10px;">
            QuickPitik · Email change
          </p>
          <h1 style="font-family:Georgia,'Times New Roman',serif;font-size:32px;font-weight:500;margin:0 0 18px;letter-spacing:-0.01em;line-height:1.05;color:#1a1a1a;">
            Confirm this address.
          </h1>
          <p style="font-size:15px;line-height:1.6;color:#3a3a3a;margin:0 0 28px;">
            Someone asked to make <strong>${escapeHtml(toEmail)}</strong> the sign-in email for their
            QuickPitik account. Tap below to confirm — the link works for 1 hour. Until you do,
            the account keeps its current address.
          </p>

          <table cellspacing="0" cellpadding="0" border="0" style="margin:0 0 28px;">
            <tr><td style="padding:6px 0;">
              <a href="${escapeUrl(confirmUrl)}"
                 style="display:inline-block;background:#3b8c5f;color:#fafaf6;
                        text-decoration:none;padding:16px 32px;border-radius:999px;
                        font-family:ui-monospace,'SFMono-Regular',Menlo,Consolas,monospace;
                        text-transform:uppercase;letter-spacing:0.2em;font-size:13px;
                        font-weight:500;">
                Confirm this email  →
              </a>
            </td></tr>
          </table>

          <hr style="border:none;border-top:1px solid #e5e2d8;margin:28px 0;" />

          <p style="font-size:13px;line-height:1.65;color:#7a7a7a;margin:0;">
            If you weren't expecting this, ignore it — nothing changes, and whoever asked
            cannot sign in with this address.
          </p>
        </td></tr>
      </table>
      <p style="text-align:center;font-family:ui-monospace,'SFMono-Regular',Menlo,Consolas,monospace;text-transform:uppercase;letter-spacing:0.3em;font-size:10px;color:#a8a8a8;margin:24px 0 0;">
        QuickPitik · Cebu, Philippines
      </p>
    </td></tr>
  </table>
</body></html>
        """.trimIndent()
    }

    private fun renderHtml(toEmail: String, resetUrl: String): String {
        return """
<!DOCTYPE html>
<html><head><meta charset="utf-8" /><title>Reset your QuickPitik password</title></head>
<body style="margin:0;padding:0;background:#f7f5ee;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;color:#1a1a1a;">
  <table cellspacing="0" cellpadding="0" border="0" width="100%" style="background:#f7f5ee;padding:32px 16px;">
    <tr><td align="center">
      <table cellspacing="0" cellpadding="0" border="0" width="100%" style="max-width:520px;margin:0 auto;background:#fafaf6;border-radius:16px;padding:36px 32px;border:1px solid #e5e2d8;">
        <tr><td>
          <p style="font-family:ui-monospace,'SFMono-Regular',Menlo,Consolas,monospace;text-transform:uppercase;letter-spacing:0.3em;font-size:11px;color:#7a7a7a;margin:0 0 10px;">
            QuickPitik · Password reset
          </p>
          <h1 style="font-family:Georgia,'Times New Roman',serif;font-size:32px;font-weight:500;margin:0 0 18px;letter-spacing:-0.01em;line-height:1.05;color:#1a1a1a;">
            Reset your password.
          </h1>
          <p style="font-size:15px;line-height:1.6;color:#3a3a3a;margin:0 0 28px;">
            Someone (hopefully you) asked to reset the QuickPitik password for
            <strong>${escapeHtml(toEmail)}</strong>. Tap below to choose a new one — the link works for 15 minutes.
          </p>

          <table cellspacing="0" cellpadding="0" border="0" style="margin:0 0 28px;">
            <tr><td style="padding:6px 0;">
              <a href="${escapeUrl(resetUrl)}"
                 style="display:inline-block;background:#3b8c5f;color:#fafaf6;
                        text-decoration:none;padding:16px 32px;border-radius:999px;
                        font-family:ui-monospace,'SFMono-Regular',Menlo,Consolas,monospace;
                        text-transform:uppercase;letter-spacing:0.2em;font-size:13px;
                        font-weight:500;">
                Set a new password  →
              </a>
            </td></tr>
          </table>

          <hr style="border:none;border-top:1px solid #e5e2d8;margin:28px 0;" />

          <p style="font-size:13px;line-height:1.65;color:#7a7a7a;margin:0;">
            If you didn't request this, ignore this email — your password stays the same. The link expires automatically.
          </p>
        </td></tr>
      </table>
      <p style="text-align:center;font-family:ui-monospace,'SFMono-Regular',Menlo,Consolas,monospace;text-transform:uppercase;letter-spacing:0.3em;font-size:10px;color:#a8a8a8;margin:24px 0 0;">
        QuickPitik · Cebu, Philippines
      </p>
    </td></tr>
  </table>
</body></html>
        """.trimIndent()
    }

    private fun escapeHtml(s: String): String = s
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\"", "&quot;")

    // Reset URL is already URL-encoded; only `&` needs `&amp;` to be valid
    // inside an href attribute in HTML email.
    private fun escapeUrl(s: String): String = s.replace("&", "&amp;")
}
