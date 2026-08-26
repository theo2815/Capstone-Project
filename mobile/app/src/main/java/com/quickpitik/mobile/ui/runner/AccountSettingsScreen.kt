package com.quickpitik.mobile.ui.runner

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage
import com.quickpitik.mobile.data.remote.RetrofitClient
import com.quickpitik.mobile.ui.theme.*

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

    var emailFormOpen by remember { mutableStateOf(false) }
    var newEmail by remember { mutableStateOf("") }
    var emailPassword by remember { mutableStateOf("") }

    var passwordMatchError by remember { mutableStateOf<String?>(null) }

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
                onLogout = onLogout
            )

            Spacer(modifier = Modifier.height(24.dp))

            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(horizontal = 24.dp),
                verticalArrangement = Arrangement.spacedBy(24.dp),
                contentPadding = PaddingValues(bottom = 24.dp)
            ) {
                // Section 1: Display Name
                item {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(BoneDeep, QpCardShape)
                            .padding(20.dp),
                        verticalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        Kicker("01 · Profile name")

                        OutlinedTextField(
                            value = nameInput,
                            onValueChange = { nameInput = it },
                            label = { Text("Full Name", color = Slate) },
                            singleLine = true,
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = Fresh,
                                unfocusedBorderColor = SlateSoft,
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
                            Text(
                                text = "Name updated successfully!",
                                color = Fresh,
                                style = Typography.bodySmall,
                                fontWeight = FontWeight.Bold
                            )
                        }

                        PrimaryCta(
                            text = "Save name",
                            onClick = { viewModel.updateName(nameInput) },
                            enabled = nameInput.trim() != name && nameInput.trim().isNotEmpty(),
                            modifier = Modifier.align(Alignment.End)
                        )
                    }
                }

                // Section 2: Profile Picture (avatar)
                item {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(BoneDeep, QpCardShape)
                            .padding(20.dp),
                        verticalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        Kicker("02 · Profile picture")
                        Text(
                            text = "Shown next to your name across QuickPitik.",
                            style = Typography.bodySmall,
                            color = Slate
                        )
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(16.dp),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Box(
                                modifier = Modifier
                                    .size(72.dp)
                                    .clip(CircleShape)
                                    .background(Fresh)
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
                                        fontSize = 28.sp
                                    )
                                }
                            }
                            Column(modifier = Modifier.weight(1f)) {
                                PrimaryCta(
                                    text = "Change photo",
                                    onClick = { avatarPicker.launch("image/*") },
                                    loading = avatarUploading
                                )
                                // Only offered when there is something to
                                // remove. Ghost, not Fresh — the Change button
                                // above already owns the one accent here.
                                if (!avatarUrl.isNullOrEmpty()) {
                                    Spacer(modifier = Modifier.height(8.dp))
                                    TextButton(
                                        onClick = { viewModel.removeAvatar() },
                                        enabled = !avatarUploading
                                    ) {
                                        Text(
                                            text = "REMOVE PHOTO",
                                            color = Slate,
                                            fontWeight = FontWeight.Bold
                                        )
                                    }
                                }
                                if (avatarError != null) {
                                    Spacer(modifier = Modifier.height(8.dp))
                                    Text(
                                        text = avatarError ?: "",
                                        color = ErrorRed,
                                        style = Typography.bodySmall
                                    )
                                }
                            }
                        }
                    }
                }

                // Section 3: Sign-in email — request a change (step 1 of 2).
                // The address shown NEVER updates here: the backend only mails a
                // confirmation link, and the swap happens when that link is
                // opened from the new inbox (web-only route). Copy has to keep
                // that promise or a runner will think they're already switched.
                item {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(BoneDeep, QpCardShape)
                            .padding(20.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Kicker("03 · Sign-in email")
                        Text(
                            text = "We'll email a confirmation link to the new address. " +
                                "Your sign-in email stays the same until you open it.",
                            style = Typography.bodySmall,
                            color = Slate
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .background(Bone, TileShape)
                                .padding(12.dp)
                        ) {
                            Text(
                                text = email,
                                style = Typography.bodyMedium,
                                fontWeight = FontWeight.Bold,
                                color = Slate
                            )
                        }

                        if (!emailFormOpen) {
                            GhostCta(
                                text = "Change email",
                                onClick = { emailFormOpen = true },
                                modifier = Modifier.align(Alignment.End)
                            )
                        } else {
                            OutlinedTextField(
                                value = newEmail,
                                onValueChange = {
                                    newEmail = it
                                    viewModel.resetEmailChangeState()
                                },
                                label = { Text("New Email", color = Slate) },
                                singleLine = true,
                                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Email),
                                colors = OutlinedTextFieldDefaults.colors(
                                    focusedBorderColor = Fresh,
                                    unfocusedBorderColor = SlateSoft,
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
                                label = { Text("Current Password", color = Slate) },
                                singleLine = true,
                                visualTransformation = PasswordVisualTransformation(),
                                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
                                colors = OutlinedTextFieldDefaults.colors(
                                    focusedBorderColor = Fresh,
                                    unfocusedBorderColor = SlateSoft,
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
                                GhostCta(
                                    text = if (emailChangeSubmitting) "Sending…" else "Send link",
                                    onClick = {
                                        viewModel.requestEmailChange(newEmail, emailPassword)
                                    },
                                    enabled = !emailChangeSubmitting &&
                                        newEmail.isNotEmpty() && emailPassword.isNotEmpty()
                                )
                            }
                        }

                        // Deliberately not phrased as success — the address has
                        // not moved yet. Kept visible after the form closes so
                        // the runner still sees where to look.
                        if (emailChangeMessage != null) {
                            Text(
                                text = emailChangeMessage!!,
                                color = Ink,
                                style = Typography.bodySmall,
                                fontWeight = FontWeight.Bold
                            )
                        }
                    }
                }

                // Section 3: Password Update
                item {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(BoneDeep, QpCardShape)
                            .padding(20.dp),
                        verticalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        Kicker("04 · Update password")

                        OutlinedTextField(
                            value = currentPassword,
                            onValueChange = {
                                currentPassword = it
                                viewModel.resetPasswordState()
                            },
                            label = { Text("Current Password", color = Slate) },
                            singleLine = true,
                            visualTransformation = PasswordVisualTransformation(),
                            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = Fresh,
                                unfocusedBorderColor = SlateSoft,
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
                            label = { Text("New Password", color = Slate) },
                            singleLine = true,
                            visualTransformation = PasswordVisualTransformation(),
                            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = Fresh,
                                unfocusedBorderColor = SlateSoft,
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
                            label = { Text("Confirm New Password", color = Slate) },
                            singleLine = true,
                            visualTransformation = PasswordVisualTransformation(),
                            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
                            colors = OutlinedTextFieldDefaults.colors(
                                focusedBorderColor = Fresh,
                                unfocusedBorderColor = SlateSoft,
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

                // Section 5: Sign out + account help
                item {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(BoneDeep, QpCardShape)
                            .padding(20.dp),
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Kicker("05 · Sign out")
                        Text(
                            text = "You'll need to sign in again on this device to access your profile, selfies, and orders.",
                            style = Typography.bodySmall,
                            color = Slate
                        )
                        GhostCta(
                            text = "Sign out",
                            onClick = onLogout,
                            modifier = Modifier.align(Alignment.End)
                        )
                        Text(
                            text = "Need to delete your account? Contact support@quickpitik.com and we'll handle it within 7 days.",
                            style = Typography.bodySmall,
                            color = Slate
                        )
                    }
                }

                item {
                    Spacer(modifier = Modifier.height(24.dp))
                }
            }
        }
    }
}
