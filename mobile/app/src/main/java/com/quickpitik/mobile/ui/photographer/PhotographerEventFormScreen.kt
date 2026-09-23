package com.quickpitik.mobile.ui.photographer

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Divider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.RadioButton
import androidx.compose.material3.RadioButtonDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.quickpitik.mobile.data.remote.PhotographerEventDetailDto
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.ui.theme.*

// Photographer-owned event create/edit (V46) — mobile mirror of the website
// my-event-form. Create = submit for admin review; edit = direct changes to
// title/date/location/notes/cover/visibility, while the pricing trio on a LIVE
// event becomes a change the admin approves (the form says so). Diff-only PATCH:
// blank/unchanged = not sent.
@Composable
fun PhotographerEventFormScreen(
    viewModel: PhotographerDashboardViewModel,
    eventId: String?, // null = create
    onBack: () -> Unit,
    onSaved: () -> Unit,
) {
    val formState by viewModel.eventFormState.collectAsState()
    val editTarget by viewModel.eventEditTarget.collectAsState()
    val eventsState by viewModel.eventsState.collectAsState()
    val summary = (eventsState as? EventsState.Success)?.events?.firstOrNull { it.id == eventId }
    val isEdit = eventId != null

    LaunchedEffect(eventId) {
        viewModel.resetEventForm()
        if (eventId != null) viewModel.loadEventDetail(eventId)
    }
    LaunchedEffect(formState) {
        if (formState is EventFormState.Success) onSaved()
    }

    // Prefill: detail (has description/organizer/price) when it lands, else the
    // list summary. Keyed to the detail so fields re-seed once it arrives.
    val seed = editTarget
    val requested = seed?.pendingChange
    val reqPricing = requested?.get("pricingMode") as? String
    val reqPrice = (requested?.get("pricePerPhoto") as? Number)?.toDouble()
    val reqWm = requested?.get("watermarkPolicy") as? String
    val reviewStatus = seed?.reviewStatus ?: summary?.reviewStatus
    val isLive = reviewStatus == "approved" || reviewStatus == "change_pending"

    var title by remember(seed) { mutableStateOf(seed?.name ?: summary?.name ?: "") }
    var location by remember(seed) { mutableStateOf(seed?.location ?: summary?.location ?: "") }
    var date by remember(seed) { mutableStateOf(seed?.date ?: summary?.date ?: "") }
    var organizer by remember(seed) { mutableStateOf(seed?.organizerName ?: "") }
    var description by remember(seed) { mutableStateOf(seed?.description ?: "") }
    var visibility by remember(seed) { mutableStateOf(reqOr(seed?.visibility, summary?.visibility) ?: "public") }
    var pricingMode by remember(seed) { mutableStateOf(reqPricing ?: seed?.pricingMode ?: summary?.pricingMode ?: "paid") }
    var price by remember(seed) {
        mutableStateOf((reqPrice ?: seed?.pricePerPhoto)?.takeIf { it > 0 }?.let { formatPrice(it) } ?: "")
    }
    var watermark by remember(seed) {
        mutableStateOf(if ((reqWm ?: seed?.watermarkPolicy) == "none") "none" else "own")
    }
    var coverUri by remember { mutableStateOf<Uri?>(null) }
    val coverPicker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) coverUri = uri
    }

    val dateValid = isValidEventDate(date)
    val priceValid = isValidEventPrice(pricingMode, price)
    val submitting = formState is EventFormState.Submitting
    val canSubmit = title.isNotBlank() && location.isNotBlank() && dateValid && priceValid && !submitting

    // Trio the form would send, and whether it differs from what it prefilled from.
    val trio = pricingTrio(pricingMode, price, watermark)
    val trioChanged = trioChanged(seed, trio)

    val submitLabel = when {
        submitting -> "Saving…"
        !isEdit -> "Submit for review"
        isLive && trioChanged -> "Save · request pricing change"
        else -> "Save changes"
    }

    fun submit() = viewModel.submitEvent(
        eventId,
        ownedEventFormInput(seed, isEdit, title, date, location, organizer, description, visibility, trio, coverUri?.toString()),
    )

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Bone)
            .navigationBarsPadding(),
    ) {
        // Top bar
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .statusBarsPadding()
                .padding(horizontal = 8.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            IconButton(onClick = onBack, modifier = Modifier.size(48.dp)) {
                Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back", tint = Ink)
            }
            Text(
                text = if (isEdit) "Edit event" else "Create an event",
                style = Typography.titleMedium,
                fontWeight = FontWeight.SemiBold,
                color = Ink,
            )
        }

        Column(
            modifier = Modifier
                .weight(1f)
                .verticalScroll(rememberScrollState())
                .imePadding()
                .padding(horizontal = 16.dp)
                .padding(bottom = 24.dp),
        ) {
            FormField("Title", title, { title = it }, "Paksit Sunrise Run", enabled = !submitting)
            Spacer(Modifier.height(16.dp))
            FormField("Location · Venue, City", location, { location = it }, "SRP Boulevard, Cebu City", enabled = !submitting)
            Spacer(Modifier.height(16.dp))
            // ponytail: plain YYYY-MM-DD text field; a native DatePicker dialog is
            // the polish upgrade (functional-first per the build mandate).
            FormField("Date · YYYY-MM-DD", date, { date = it }, "2026-09-20", enabled = !submitting)
            if (date.isNotBlank() && !dateValid) FieldHint("Use YYYY-MM-DD (e.g. 2026-09-20).")
            Spacer(Modifier.height(16.dp))
            FormField("Organizer · optional", organizer, { organizer = it }, "Your studio or the race organizer", enabled = !submitting)
            Spacer(Modifier.height(16.dp))
            FormField("Race day notes · optional", description, { description = it }, "Where you'll shoot, what to look for.", enabled = !submitting, singleLine = false)

            Spacer(Modifier.height(20.dp))
            Kicker("Cover photo · optional")
            Spacer(Modifier.height(8.dp))
            val coverModel: Any? = coverUri ?: RetrofitClient.resolveImageUrl(seed?.bannerUrl)
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .aspectRatio(16f / 9f)
                    .clip(QpCardShape)
                    .background(BoneDeep)
                    .border(1.dp, Line, QpCardShape)
                    .clickable(enabled = !submitting) { coverPicker.launch("image/*") },
                contentAlignment = Alignment.Center,
            ) {
                if (coverModel != null) {
                    AsyncImage(model = coverModel, contentDescription = "Cover", contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize())
                } else {
                    Text("Tap to choose a cover", style = Typography.bodyMedium, color = SlateSoft)
                }
            }

            Spacer(Modifier.height(24.dp))
            Divider()
            Spacer(Modifier.height(16.dp))
            Kicker(if (isLive) "Pricing · change request" else "Pricing")
            if (isLive) {
                Spacer(Modifier.height(6.dp))
                Text(
                    "Pricing on a live event changes only after an admin approves it.",
                    style = Typography.bodySmall,
                    color = SlateSoft,
                )
            }
            Spacer(Modifier.height(10.dp))
            RadioOption("Paid — runners buy each photo; QuickPitik marks the previews", pricingMode == "paid", !submitting) { pricingMode = "paid" }
            RadioOption("Free — anyone downloads the original; never marked", pricingMode == "free", !submitting) { pricingMode = "free" }

            Spacer(Modifier.height(12.dp))
            if (pricingMode == "paid") {
                FormField("Price per photo · ₱", price, { price = it.filter { c -> c.isDigit() || c == '.' } }, "150", enabled = !submitting)
                if (price.isNotBlank() && !priceValid) FieldHint("Enter a price above zero (e.g. 150).")
            } else {
                Kicker("Watermark on previews")
                Spacer(Modifier.height(8.dp))
                RadioOption("My logo on the previews", watermark == "own", !submitting) { watermark = "own" }
                RadioOption("No watermark at all", watermark == "none", !submitting) { watermark = "none" }
            }

            Spacer(Modifier.height(24.dp))
            Divider()
            Spacer(Modifier.height(16.dp))
            Kicker("Visibility")
            Spacer(Modifier.height(6.dp))
            Text(
                "Your events never appear on the general Events page — runners reach them from your profile or a link you share. This only sets whether it shows on your public profile.",
                style = Typography.bodySmall,
                color = SlateSoft,
            )
            Spacer(Modifier.height(10.dp))
            RadioOption("Public — shown on your photographer profile", visibility == "public", !submitting) { visibility = "public" }
            RadioOption("Private — hidden from your profile, reachable only by your link", visibility == "unlisted", !submitting) { visibility = "unlisted" }

            if (!isEdit) {
                Spacer(Modifier.height(16.dp))
                Text(
                    "An admin reviews the event and its price first. Uploads open the moment it is approved.",
                    style = Typography.bodySmall,
                    color = SlateSoft,
                )
            }

            (formState as? EventFormState.Error)?.let {
                Spacer(Modifier.height(16.dp))
                Text(it.message, style = Typography.bodyMedium, color = ErrorRed)
            }

            Spacer(Modifier.height(24.dp))
            PrimaryCta(
                text = submitLabel,
                onClick = { submit() },
                modifier = Modifier.fillMaxWidth(),
                enabled = canSubmit,
                loading = submitting,
            )
        }
    }
}

private fun reqOr(a: String?, b: String?): String? = a ?: b

private fun formatPrice(v: Double): String =
    if (v % 1.0 == 0.0) v.toInt().toString() else v.toString()

@Composable
private fun FormField(
    label: String,
    value: String,
    onValueChange: (String) -> Unit,
    placeholder: String,
    enabled: Boolean = true,
    singleLine: Boolean = true,
) {
    Kicker(label)
    Spacer(Modifier.height(8.dp))
    TextField(
        value = value,
        onValueChange = onValueChange,
        placeholder = { Text(placeholder, color = SlateSoft) },
        singleLine = singleLine,
        enabled = enabled,
        colors = TextFieldDefaults.colors(
            focusedContainerColor = BoneDeep,
            unfocusedContainerColor = BoneDeep,
            focusedIndicatorColor = Fresh,
            unfocusedIndicatorColor = Color.Transparent,
            focusedTextColor = Ink,
            unfocusedTextColor = InkSoft,
        ),
        shape = FieldShape,
        modifier = Modifier.fillMaxWidth(),
    )
}

@Composable
private fun FieldHint(text: String) {
    Spacer(Modifier.height(6.dp))
    Text(text, style = Typography.bodySmall, color = ErrorRed)
}

@Composable
private fun RadioOption(text: String, selected: Boolean, enabled: Boolean, onClick: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .heightIn(min = 48.dp)
            .clip(QpCardShape)
            .clickable(enabled = enabled, onClick = onClick)
            .padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        RadioButton(
            selected = selected,
            onClick = onClick,
            enabled = enabled,
            colors = RadioButtonDefaults.colors(selectedColor = Fresh),
        )
        Spacer(Modifier.width(4.dp))
        Text(text, style = Typography.bodyMedium, color = Ink)
    }
}

// Pure form logic, lifted out of the composable so the diff-PATCH and the
// pricing-change detection are testable: a false negative silently drops a
// price edit, a false positive sends a live event back into admin review.

internal fun isValidEventDate(date: String): Boolean = Regex("""\d{4}-\d{2}-\d{2}""").matches(date)

internal fun isValidEventPrice(pricingMode: String, price: String): Boolean {
    val value = price.trim().toDoubleOrNull()
    return pricingMode == "free" || (price.trim().isNotEmpty() && value != null && value > 0)
}

/** The pricing trio the form would send for its current fields. */
internal data class PricingTrio(val pricingMode: String, val price: String, val watermarkPolicy: String)

internal fun pricingTrio(pricingMode: String, price: String, watermark: String) = PricingTrio(
    pricingMode = pricingMode,
    price = if (pricingMode == "free") "0" else price.trim(),
    watermarkPolicy = if (pricingMode == "free") watermark else "platform",
)

/** Whether [trio] differs from what the form prefilled from: the pending change when one exists, else the event. */
internal fun trioChanged(seed: PhotographerEventDetailDto?, trio: PricingTrio): Boolean {
    if (seed == null) return true
    val requested = seed.pendingChange
    val basePricing = requested?.get("pricingMode") as? String ?: seed.pricingMode
    val basePrice = (requested?.get("pricePerPhoto") as? Number)?.toDouble() ?: seed.pricePerPhoto
    val baseWm = requested?.get("watermarkPolicy") as? String ?: seed.watermarkPolicy
    return basePricing != trio.pricingMode ||
        baseWm != trio.watermarkPolicy ||
        basePrice != (trio.price.toDoubleOrNull() ?: 0.0)
}

/** Create sends every field; edit diffs against the prefill source and sends only what changed. */
internal fun ownedEventFormInput(
    seed: PhotographerEventDetailDto?,
    isEdit: Boolean,
    title: String,
    date: String,
    location: String,
    organizer: String,
    description: String,
    visibility: String,
    trio: PricingTrio,
    coverUri: String?,
): OwnedEventFormInput {
    if (!isEdit) {
        return OwnedEventFormInput(
            title = title.trim(),
            date = date,
            location = location.trim(),
            organizerName = organizer.trim().ifBlank { null },
            description = description.trim().ifBlank { null },
            visibility = visibility,
            pricingMode = trio.pricingMode,
            pricePerPhoto = trio.price,
            watermarkPolicy = trio.watermarkPolicy,
            coverUri = coverUri,
        )
    }
    val changed = trioChanged(seed, trio)
    return OwnedEventFormInput(
        title = title.trim().takeIf { it != seed?.name },
        date = date.takeIf { it != seed?.date },
        location = location.trim().takeIf { it != seed?.location },
        organizerName = organizer.trim().takeIf { it != (seed?.organizerName ?: "") },
        description = description.trim().takeIf { it != (seed?.description ?: "") },
        visibility = visibility.takeIf { it != seed?.visibility },
        pricingMode = trio.pricingMode.takeIf { changed },
        pricePerPhoto = trio.price.takeIf { changed },
        watermarkPolicy = trio.watermarkPolicy.takeIf { changed },
        coverUri = coverUri,
    )
}
