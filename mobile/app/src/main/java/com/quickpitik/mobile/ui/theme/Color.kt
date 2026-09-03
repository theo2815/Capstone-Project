package com.quickpitik.mobile.ui.theme

import androidx.compose.ui.graphics.Color

// "Finish Line" palette — Website Alignment (overhaul 2026-08-25, migrated to
// mobile 2026-08-26). The neutrals were already byte-identical to the
// website's paper/paper-deep/ink/etc.; only the greens changed and four depth
// tokens are new. The "Bone" names are kept deliberately: the website itself
// keeps --bone as an alias of --paper, and renaming would touch 40+ files for
// zero functional gain.
val Bone = Color(0xFFF8F5EE)         // Warm cream background (website --paper)
val BoneDeep = Color(0xFFEFEAE0)     // Deep cream surface / active tab (--paper-deep)
val Ink = Color(0xFF111111)          // Primary deep ink text
val InkSoft = Color(0xFF2A2A28)      // Soft dark text / dark surface card
val Fresh = Color(0xFF1B7A46)        // Brand accent — Finish Line deep forest green
val FreshDeep = Color(0xFF14562F)    // Hover / pressed state green
val Slate = Color(0xFF475569)        // Slate gray text
val SlateSoft = Color(0xFF64748B)    // Muted light slate gray
val Line = Color(0xFFE5E2D9)         // Border outlines & dividers
val SurfaceWhite = Color(0xFFFFFFFF)  // Light card surface

// Finish Line depth tokens (new 2026-08-26, mirroring globals.css)
val Pine = Color(0xFF0C3321)         // Deepest forest — depth without black
val PineSoft = Color(0xFF123F28)     // Slightly lifted pine
val FreshTint = Color(0xFFE6F1E9)    // Light green wash on paper (chip/badge bg)
val LineStrong = Color(0xFFD4D2C8)   // Stronger border tier

// Semantic states
val ErrorRed = Color(0xFFDC2626)
val WarningOrange = Color(0xFFD97706)
val SuccessGreen = Fresh              // website: --success: var(--fresh)

// Alpha overlays — use these instead of inline Color.Black.copy(alpha = …).
val Scrim = Color(0x66000000)          // 40% ink — fullscreen loading / success scrims
val WatermarkInk = Color(0x59000000)   // ~35% ink — photo-preview watermark backdrop
val NavChrome = Color(0xE6FFFFFF)      // 90% white — pills floating over a photo (website bg-surface/90)