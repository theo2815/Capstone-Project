package com.quickpitik.mobile.ui.runner

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.ExitToApp
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.Person
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.quickpitik.mobile.BuildConfig
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.ui.theme.ArrowLabel
import com.quickpitik.mobile.ui.theme.Bone
import com.quickpitik.mobile.ui.theme.BoneDeep
import com.quickpitik.mobile.ui.theme.ErrorRed
import com.quickpitik.mobile.ui.theme.Fresh
import com.quickpitik.mobile.ui.theme.GhostCta
import com.quickpitik.mobile.ui.theme.Ink
import com.quickpitik.mobile.ui.theme.Kicker
import com.quickpitik.mobile.ui.theme.Line
import com.quickpitik.mobile.ui.theme.PillShape
import com.quickpitik.mobile.ui.theme.PrimaryCta
import com.quickpitik.mobile.ui.theme.QpCardShape
import com.quickpitik.mobile.ui.theme.Slate
import com.quickpitik.mobile.ui.theme.SlateSoft
import com.quickpitik.mobile.ui.theme.TileShape
import com.quickpitik.mobile.ui.theme.Typography

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AccountSettingsScreen(
    viewModel: ProfileViewModel,
    onLogout: () -> Unit
) {
    val name by viewModel.profileName.collectAsState()
    val email by viewModel.profileEmail.collectAsState()

    val avatarUrl by viewModel.avatarUrl.collectAsState()
    val avatarUploading by viewModel.avatarUploading.collectAsState()
    val avatarError by viewModel.avatarError.collectAsState()

    val avatarPicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri -> uri?.let { viewModel.uploadAvatar(it) } }

    val nameSuccess by viewModel.nameUpdateSuccess.collectAsState()
    val nameError by viewModel.nameUpdateError.collectAsState()

    val pwdSuccess by viewModel.passwordUpdateSuccess.collectAsState()
    val pwdError by viewModel.passwordUpdateError.collectAsState()
    val pwdSessionKept by viewModel.passwordSessionKept.collectAsState()

    val emailChangeSubmitting by viewModel.emailChangeSubmitting.collectAsState()
    val emailChangeMessage by viewModel.emailChangeMessage.collectAsState()
    val emailChangeError by viewModel.emailChangeError.collectAsState()

    var nameInput by remember { mutableStateOf(name) }
    var currentPassword by remember { mutableStateOf("") }
    var newPassword by remember { mutableStateOf("") }
    var confirmPassword by remember { mutableStateOf("") }

    var currentPasswordVisible by remember { mutableStateOf(false) }
    var newPasswordVisible by remember { mutableStateOf(false) }
    var confirmPasswordVisible by remember { mutableStateOf(false) }

    var emailFormOpen by remember { mutableStateOf(false) }
    var newEmail by remember { mutableStateOf("") }
    var emailPassword by remember { mutableStateOf("") }
    var emailPasswordVisible by remember { mutableStateOf(false) }

    var passwordMatchError by remember { mutableStateOf<String?>(null) }
    var showRefundPolicy by remember { mutableStateOf(false) }
    var showLogoutConfirm by remember { mutableStateOf(false) }

    // Synchronize local input state with ViewModel name if updated
    LaunchedEffect(name) {
        nameInput = name
    }

    LaunchedEffect(nameSuccess, nameError) {
        if (nameSuccess) {
            viewModel.resetNameState()
        }
    }

    // Clears the inputs but deliberately does NOT call resetPasswordState():
    // that flipped `pwdSuccess` back to false in the same frame, so the
    // confirmation text below rendered for about one frame and was never
    // readable. The flag is cleared when the user next edits a password field.
    LaunchedEffect(pwdSuccess) {
        if (pwdSuccess) {
            currentPassword = ""
            newPassword = ""
            confirmPassword = ""
        }
    }

    // The email form closes on success; the backend's "check your new inbox"
    // message stays on the card, because that is the instruction the runner
    // still has to act on.
    LaunchedEffect(emailChangeMessage) {
        if (emailChangeMessage != null) {
            emailFormOpen = false
            newEmail = ""
            emailPassword = ""
        }
    }

    Surface(
        modifier = Modifier.fillMaxSize(),
        color = Bone
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
                .navigationBarsPadding()
                .padding(top = 24.dp)
        ) {
            // Top Bar
            RunnerTopBar(
                kicker = "ACCOUNT SETTINGS",
                userName = name,
                avatarUrl = avatarUrl,
                onLogout = { showLogoutConfirm = true }
            )

            Spacer(modifier = Modifier.height(20.dp))

            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(horizontal = 20.dp),
                verticalArrangement = Arrangement.spacedBy(20.dp),
                contentPadding = PaddingValues(bottom = 32.dp)
            ) {
                // Section: Account Identity Hero Card
                item {
                    Card(
                        shape = QpCardShape,
                        colors = CardDefaults.cardColors(containerColor = BoneDeep),
                        border = BorderStroke(1.dp, Line),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Column(
                            modifier = Modifier.padding(18.dp),
                            verticalArrangement = Arrangement.spacedBy(16.dp)
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(16.dp),
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                // Interactive 76dp Avatar with tap-to-change
                                Box(
                                    modifier = Modifier
                                        .size(76.dp)
                                        .clip(CircleShape)
                                        .background(Fresh)
                                        .border(2.dp, Line, CircleShape)
                                        .clickable(enabled = !avatarUploading) { avatarPicker.launch("image/*") },
                                    contentAlignment = Alignment.Center
                                ) {
                                    if (!avatarUrl.isNullOrEmpty()) {
                                        AsyncImage(
                                            model = RetrofitClient.resolveImageUrl(avatarUrl),
                                            contentDescription = "Profile picture",
                                            modifier = Modifier.fillMaxSize(),
                                            contentScale = ContentScale.Crop
                                        )
                                    } else {
                                        Text(
                                            text = name.ifBlank { "Runner" }.take(1).uppercase(),
                                            color = Bone,
                                            fontWeight = FontWeight.Bold,
                                            fontSize = 30.sp
                                        )
                                    }
                                }

                                // User info & account badge
                                Column(modifier = Modifier.weight(1f)) {
                                    Surface(
                                        shape = PillShape,
                                        color = Ink,
                                        modifier = Modifier.padding(bottom = 6.dp)
                                    ) {
                                        Row(
                                            modifier = Modifier.padding(horizontal = 8.dp, vertical = 3.dp),
                                            verticalAlignment = Alignment.CenterVertically,
                                            horizontalArrangement = Arrangement.spacedBy(5.dp)
                                        ) {
                                            Box(modifier = Modifier.size(6.dp).clip(CircleShape).background(Fresh))
                                            Text(
                                                text = "RUNNER ACCOUNT",
                                                style = Typography.labelSmall.copy(fontWeight = FontWeight.Bold, letterSpacing = 0.5.sp),
                                                color = Bone
                                            )
                                        }
                                    }
                                    Text(
                                        text = name.ifBlank { "QuickPitik Runner" },
                                        style = Typography.titleLarge,
                                        fontWeight = FontWeight.Bold,
                                        color = Ink,
                                        maxLines = 1,
                                        overflow = TextOverflow.Ellipsis
                                    )
                                    Text(
                                        text = email,
                                        style = Typography.bodySmall,
                                        color = Slate,
                                        maxLines = 1,
                                        overflow = TextOverflow.Ellipsis
                                    )
                                }
                            }

                            // Avatar Actions
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(10.dp)
                            ) {
                                PrimaryCta(
                                    text = "Change photo",
                                    onClick = { avatarPicker.launch("image/*") },
                                    loading = avatarUploading
                                )
                                if (!avatarUrl.isNullOrEmpty()) {
                                    GhostCta(
                                        text = "Remove photo",
                                        onClick = { viewModel.removeAvatar() },
                                        enabled = !avatarUploading
                                    )
                                }
                            }

                            if (avatarError != null) {
                                Text(
                                    text = avatarError ?: "",
                                    color = ErrorRed,
                                    style = Typography.bodySmall
                                )
                            }
                        }
                    }
                }

                // Section 1: Profile Name
                item {
                    Card(
                        shape = QpCardShape,
                        colors = CardDefaults.cardColors(containerColor = BoneDeep),
                        border = BorderStroke(1.dp, Line),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Column(
                            modifier = Modifier.padding(18.dp),
                            verticalArrangement = Arrangement.spacedBy(14.dp)
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(6.dp)
                            ) {
                                Icon(Icons.Default.Person, contentDescription = null, tint = Fresh, modifier = Modifier.size(16.dp))
                                Kicker("01 · Profile information")
                            }
                            Text(
                                text = "Update your full name as it appears on race results, leaderboards, and order receipts.",
                                style = Typography.bodySmall,
                                color = Slate
                            )

                            OutlinedTextField(
                                value = nameInput,
                                onValueChange = { nameInput = it },
                                label = { Text("Full name", color = Slate) },
                                singleLine = true,
                                colors = OutlinedTextFieldDefaults.colors(
                                    focusedBorderColor = Fresh,
                                    unfocusedBorderColor = Line,
                                    focusedLabelColor = Fresh,
                                    focusedTextColor = Ink,
                                    unfocusedTextColor = Ink
                                ),
                                modifier = Modifier.fillMaxWidth()
                            )

                            if (nameError != null) {
                                Text(
                                    text = nameError ?: "",
                                    color = ErrorRed,
                                    style = Typography.bodySmall
                                )
                            }

                            if (nameSuccess) {
                                Row(
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.spacedBy(4.dp)
                                ) {
                                    Icon(Icons.Default.CheckCircle, contentDescription = null, tint = Fresh, modifier = Modifier.size(16.dp))
                                    Text(
                                        text = "Profile name updated successfully!",
                                        color = Fresh,
                                        style = Typography.bodySmall,
                                        fontWeight = FontWeight.Bold
                                    )
                                }
                            }

                            PrimaryCta(
                                text = "Save name",
                                onClick = { viewModel.updateName(nameInput) },
                                enabled = nameInput.trim() != name && nameInput.trim().isNotEmpty(),
                                modifier = Modifier.align(Alignment.End)
                            )
                        }
                    }
                }

                // Section 2: Sign-in Email
                item {
                    Card(
                        shape = QpCardShape,
                        colors = CardDefaults.cardColors(containerColor = BoneDeep),
                        border = BorderStroke(1.dp, Line),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Column(
                            modifier = Modifier.padding(18.dp),
                            verticalArrangement = Arrangement.spacedBy(12.dp)
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(6.dp)
                            ) {
                                Icon(Icons.Default.Info, contentDescription = null, tint = Fresh, modifier = Modifier.size(16.dp))
                                Kicker("02 · Sign-in email")
                            }
                            Text(
                                text = "We'll email a verification link to the new address. Your sign-in email remains unchanged until you confirm it.",
                                style = Typography.bodySmall,
                                color = Slate
                            )

                            Surface(
                                shape = TileShape,
                                color = Bone,
                                border = BorderStroke(1.dp, Line),
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Row(
                                    modifier = Modifier.padding(horizontal = 14.dp, vertical = 12.dp),
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.SpaceBetween
                                ) {
                                    Column {
                                        Kicker("CURRENT EMAIL", color = SlateSoft)
                                        Spacer(modifier = Modifier.height(2.dp))
                                        Text(
                                            text = email,
                                            style = Typography.bodyMedium,
                                            fontWeight = FontWeight.Bold,
                                            color = Ink
                                        )
                                    }
                                    if (!emailFormOpen) {
                                        GhostCta(
                                            text = "Change",
                                            onClick = { emailFormOpen = true }
                                        )
                                    }
                                }
                            }

                            if (emailFormOpen) {
                                OutlinedTextField(
                                    value = newEmail,
                                    onValueChange = {
                                        newEmail = it
                                        viewModel.resetEmailChangeState()
                                    },
                                    label = { Text("New email address", color = Slate) },
                                    singleLine = true,
                                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Email),
                                    colors = OutlinedTextFieldDefaults.colors(
                                        focusedBorderColor = Fresh,
                                        unfocusedBorderColor = Line,
                                        focusedTextColor = Ink,
                                        unfocusedTextColor = Ink
                                    ),
                                    modifier = Modifier.fillMaxWidth()
                                )

                                OutlinedTextField(
                                    value = emailPassword,
                                    onValueChange = {
                                        emailPassword = it
                                        viewModel.resetEmailChangeState()
                                    },
                                    label = { Text("Current password to confirm", color = Slate) },
                                    singleLine = true,
                                    visualTransformation = if (emailPasswordVisible) VisualTransformation.None else PasswordVisualTransformation(),
                                    trailingIcon = {
                                        Text(
                                            text = if (emailPasswordVisible) "HIDE" else "SHOW",
                                            color = Slate,
                                            style = Typography.labelMedium,
                                            fontWeight = FontWeight.Bold,
                                            modifier = Modifier
                                                .clickable { emailPasswordVisible = !emailPasswordVisible }
                                                .padding(end = 12.dp)
                                        )
                                    },
                                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
                                    colors = OutlinedTextFieldDefaults.colors(
                                        focusedBorderColor = Fresh,
                                        unfocusedBorderColor = Line,
                                        focusedTextColor = Ink,
                                        unfocusedTextColor = Ink
                                    ),
                                    modifier = Modifier.fillMaxWidth()
                                )

                                if (emailChangeError != null) {
                                    Text(
                                        text = emailChangeError!!,
                                        color = ErrorRed,
                                        style = Typography.bodySmall
                                    )
                                }

                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.End,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    TextButton(
                                        onClick = {
                                            emailFormOpen = false
                                            newEmail = ""
                                            emailPassword = ""
                                            viewModel.resetEmailChangeState()
                                        }
                                    ) {
                                        Text("CANCEL", color = Slate, fontWeight = FontWeight.Bold)
                                    }
                                    Spacer(modifier = Modifier.width(8.dp))
                                    PrimaryCta(
                                        text = if (emailChangeSubmitting) "Sending…" else "Send link",
                                        onClick = {
                                            viewModel.requestEmailChange(newEmail, emailPassword)
                                        },
                                        enabled = !emailChangeSubmitting &&
                                            newEmail.isNotEmpty() && emailPassword.isNotEmpty()
                                    )
                                }
                            }

                            if (emailChangeMessage != null) {
                                Row(
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                                ) {
                                    Icon(Icons.Default.CheckCircle, contentDescription = null, tint = Fresh, modifier = Modifier.size(16.dp))
                                    Text(
                                        text = emailChangeMessage!!,
                                        color = Ink,
                                        style = Typography.bodySmall,
                                        fontWeight = FontWeight.Bold
                                    )
                                }
                            }
                        }
                    }
                }

                // Section 3: Password Update with SHOW/HIDE Toggles
                item {
                    Card(
                        shape = QpCardShape,
                        colors = CardDefaults.cardColors(containerColor = BoneDeep),
                        border = BorderStroke(1.dp, Line),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Column(
                            modifier = Modifier.padding(18.dp),
                            verticalArrangement = Arrangement.spacedBy(14.dp)
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(6.dp)
                            ) {
                                Icon(Icons.Default.Lock, contentDescription = null, tint = Fresh, modifier = Modifier.size(16.dp))
                                Kicker("03 · Update password")
                            }
                            Text(
                                text = "Use at least 8 characters.",
                                style = Typography.bodySmall,
                                color = Slate
                            )

                            OutlinedTextField(
                                value = currentPassword,
                                onValueChange = {
                                    currentPassword = it
                                    viewModel.resetPasswordState()
                                },
                                label = { Text("Current password", color = Slate) },
                                singleLine = true,
                                visualTransformation = if (currentPasswordVisible) VisualTransformation.None else PasswordVisualTransformation(),
                                trailingIcon = {
                                    Text(
                                        text = if (currentPasswordVisible) "HIDE" else "SHOW",
                                        color = Slate,
                                        style = Typography.labelMedium,
                                        fontWeight = FontWeight.Bold,
                                        modifier = Modifier
                                            .clickable { currentPasswordVisible = !currentPasswordVisible }
                                            .padding(end = 12.dp)
                                    )
                                },
                                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
                                colors = OutlinedTextFieldDefaults.colors(
                                    focusedBorderColor = Fresh,
                                    unfocusedBorderColor = Line,
                                    focusedTextColor = Ink,
                                    unfocusedTextColor = Ink
                                ),
                                modifier = Modifier.fillMaxWidth()
                            )

                            OutlinedTextField(
                                value = newPassword,
                                onValueChange = {
                                    newPassword = it
                                    passwordMatchError = null
                                    viewModel.resetPasswordState()
                                },
                                label = { Text("New password", color = Slate) },
                                singleLine = true,
                                visualTransformation = if (newPasswordVisible) VisualTransformation.None else PasswordVisualTransformation(),
                                trailingIcon = {
                                    Text(
                                        text = if (newPasswordVisible) "HIDE" else "SHOW",
                                        color = Slate,
                                        style = Typography.labelMedium,
                                        fontWeight = FontWeight.Bold,
                                        modifier = Modifier
                                            .clickable { newPasswordVisible = !newPasswordVisible }
                                            .padding(end = 12.dp)
                                    )
                                },
                                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
                                colors = OutlinedTextFieldDefaults.colors(
                                    focusedBorderColor = Fresh,
                                    unfocusedBorderColor = Line,
                                    focusedTextColor = Ink,
                                    unfocusedTextColor = Ink
                                ),
                                modifier = Modifier.fillMaxWidth()
                            )

                            OutlinedTextField(
                                value = confirmPassword,
                                onValueChange = {
                                    confirmPassword = it
                                    passwordMatchError = null
                                    viewModel.resetPasswordState()
                                },
                                label = { Text("Confirm new password", color = Slate) },
                                singleLine = true,
                                visualTransformation = if (confirmPasswordVisible) VisualTransformation.None else PasswordVisualTransformation(),
                                trailingIcon = {
                                    Text(
                                        text = if (confirmPasswordVisible) "HIDE" else "SHOW",
                                        color = Slate,
                                        style = Typography.labelMedium,
                                        fontWeight = FontWeight.Bold,
                                        modifier = Modifier
                                            .clickable { confirmPasswordVisible = !confirmPasswordVisible }
                                            .padding(end = 12.dp)
                                    )
                                },
                                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
                                colors = OutlinedTextFieldDefaults.colors(
                                    focusedBorderColor = Fresh,
                                    unfocusedBorderColor = Line,
                                    focusedTextColor = Ink,
                                    unfocusedTextColor = Ink
                                ),
                                modifier = Modifier.fillMaxWidth()
                            )

                            val errorToShow = passwordMatchError ?: pwdError
                            if (errorToShow != null) {
                                Text(
                                    text = errorToShow,
                                    color = ErrorRed,
                                    style = Typography.bodySmall
                                )
                            }

                            if (pwdSuccess) {
                                Row(
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                                ) {
                                    Icon(Icons.Default.CheckCircle, contentDescription = null, tint = Fresh, modifier = Modifier.size(16.dp))
                                    Text(
                                        text = if (pwdSessionKept) {
                                            "Password changed. Other devices were signed out."
                                        } else {
                                            "Password changed. You'll be signed out on this device shortly."
                                        },
                                        color = Fresh,
                                        style = Typography.bodySmall,
                                        fontWeight = FontWeight.Bold
                                    )
                                }
                            }

                            PrimaryCta(
                                text = "Update password",
                                onClick = {
                                    if (newPassword != confirmPassword) {
                                        passwordMatchError = "New passwords do not match"
                                    } else {
                                        viewModel.changePassword(currentPassword, newPassword)
                                    }
                                },
                                enabled = currentPassword.isNotEmpty() && newPassword.isNotEmpty() && confirmPassword.isNotEmpty(),
                                modifier = Modifier.align(Alignment.End)
                            )
                        }
                    }
                }

                // Section 4: About & Legal
                item {
                    Card(
                        shape = QpCardShape,
                        colors = CardDefaults.cardColors(containerColor = BoneDeep),
                        border = BorderStroke(1.dp, Line),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Column(
                            modifier = Modifier.padding(18.dp),
                            verticalArrangement = Arrangement.spacedBy(10.dp)
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(6.dp)
                            ) {
                                Icon(Icons.Default.Info, contentDescription = null, tint = Fresh, modifier = Modifier.size(16.dp))
                                Kicker("04 · About & Policies")
                            }

                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .clickable { showRefundPolicy = true }
                                    .padding(vertical = 6.dp),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text("Refund & Dispute Policy", style = Typography.bodyMedium, color = Ink, fontWeight = FontWeight.Medium)
                                ArrowLabel("View →", color = Slate, style = Typography.labelMedium)
                            }

                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(vertical = 6.dp),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text("Support Email", style = Typography.bodyMedium, color = Ink, fontWeight = FontWeight.Medium)
                                Text("support@quickpitik.com", style = Typography.bodyMedium, color = Slate)
                            }

                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(vertical = 6.dp),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text("App Version", style = Typography.bodyMedium, color = Ink, fontWeight = FontWeight.Medium)
                                Text("QuickPitik v${BuildConfig.VERSION_NAME}", style = Typography.bodySmall, color = SlateSoft)
                            }
                        }
                    }
                }

                // Section 5: Account Session & Sign Out
                item {
                    Card(
                        shape = QpCardShape,
                        colors = CardDefaults.cardColors(containerColor = BoneDeep),
                        border = BorderStroke(1.dp, Line),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Column(
                            modifier = Modifier.padding(18.dp),
                            verticalArrangement = Arrangement.spacedBy(14.dp)
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.spacedBy(6.dp)
                            ) {
                                Icon(Icons.Default.ExitToApp, contentDescription = null, tint = ErrorRed, modifier = Modifier.size(16.dp))
                                Kicker("05 · Account session", color = ErrorRed)
                            }
                            Text(
                                text = "Signing out will clear your local session on this phone. You can sign back in anytime.",
                                style = Typography.bodySmall,
                                color = Slate
                            )

                            GhostCta(
                                text = "Sign out",
                                onClick = { showLogoutConfirm = true },
                                modifier = Modifier.align(Alignment.End)
                            )

                            Spacer(modifier = Modifier.height(4.dp))
                            Text(
                                text = "Need to permanently delete your account? Contact support@quickpitik.com and our team will process your request within 7 business days.",
                                style = Typography.bodySmall,
                                color = SlateSoft
                            )
                        }
                    }
                }
            }
        }
    }

    if (showRefundPolicy) {
        RefundPolicyDialog(onDismiss = { showRefundPolicy = false })
    }

    if (showLogoutConfirm) {
        AlertDialog(
            onDismissRequest = { showLogoutConfirm = false },
            containerColor = Bone,
            title = {
                Text(
                    text = "Sign out of QuickPitik?",
                    style = Typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = Ink
                )
            },
            text = {
                Text(
                    text = "You will need to sign in again to access your race log, selfies, and purchased photos.",
                    style = Typography.bodyMedium,
                    color = Slate
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        showLogoutConfirm = false
                        onLogout()
                    }
                ) {
                    Text("SIGN OUT", color = ErrorRed, fontWeight = FontWeight.Bold)
                }
            },
            dismissButton = {
                TextButton(onClick = { showLogoutConfirm = false }) {
                    Text("CANCEL", color = Slate, fontWeight = FontWeight.Bold)
                }
            }
        )
    }
}
