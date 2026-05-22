package com.quickpitik.mobile.ui.runner

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.quickpitik.mobile.ui.theme.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AccountSettingsScreen(
    viewModel: ProfileViewModel,
    onNavigateBack: () -> Unit
) {
    val name by viewModel.profileName.collectAsState()
    val email by viewModel.profileEmail.collectAsState()

    val nameSuccess by viewModel.nameUpdateSuccess.collectAsState()
    val nameError by viewModel.nameUpdateError.collectAsState()

    val pwdSuccess by viewModel.passwordUpdateSuccess.collectAsState()
    val pwdError by viewModel.passwordUpdateError.collectAsState()

    var nameInput by remember { mutableStateOf(name) }
    var currentPassword by remember { mutableStateOf("") }
    var newPassword by remember { mutableStateOf("") }
    var confirmPassword by remember { mutableStateOf("") }

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

    LaunchedEffect(pwdSuccess) {
        if (pwdSuccess) {
            currentPassword = ""
            newPassword = ""
            confirmPassword = ""
            viewModel.resetPasswordState()
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
        ) {
            // Top Bar
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(horizontal = 16.dp, vertical = 12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                IconButton(
                    onClick = onNavigateBack,
                    colors = IconButtonDefaults.iconButtonColors(containerColor = BoneDeep)
                ) {
                    Icon(Icons.Default.ArrowBack, contentDescription = "Back", tint = Ink)
                }
                Spacer(modifier = Modifier.width(16.dp))
                Text(
                    text = "ACCOUNT SETTINGS",
                    style = Typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = Ink
                )
            }

            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(horizontal = 24.dp),
                verticalArrangement = Arrangement.spacedBy(24.dp)
            ) {
                // Section 1: Display Name
                item {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(BoneDeep, RoundedCornerShape(16.dp))
                            .padding(20.dp),
                        verticalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        Text(
                            text = "01. Profile Name",
                            style = Typography.titleSmall,
                            fontWeight = FontWeight.Bold,
                            color = Ink
                        )

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

                        Button(
                            onClick = { viewModel.updateName(nameInput) },
                            enabled = nameInput.trim() != name && nameInput.trim().isNotEmpty(),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = Fresh,
                                contentColor = Bone,
                                disabledContainerColor = SlateSoft,
                                disabledContentColor = Slate
                            ),
                            shape = RoundedCornerShape(8.dp),
                            modifier = Modifier.align(Alignment.End)
                        ) {
                            Text("SAVE NAME", fontWeight = FontWeight.Bold)
                        }
                    }
                }

                // Section 2: Email Card (Read-only)
                item {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(BoneDeep, RoundedCornerShape(16.dp))
                            .padding(20.dp),
                        verticalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Text(
                            text = "02. Sign-In Email",
                            style = Typography.titleSmall,
                            fontWeight = FontWeight.Bold,
                            color = Ink
                        )
                        Text(
                            text = "Your account email address is locked for security.",
                            style = Typography.bodySmall,
                            color = Slate
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .background(Bone, RoundedCornerShape(8.dp))
                                .padding(12.dp)
                        ) {
                            Text(
                                text = email,
                                style = Typography.bodyMedium,
                                fontWeight = FontWeight.Bold,
                                color = Slate
                            )
                        }
                    }
                }

                // Section 3: Password Update
                item {
                    Column(
                        modifier = Modifier
                            .fillMaxWidth()
                            .background(BoneDeep, RoundedCornerShape(16.dp))
                            .padding(20.dp),
                        verticalArrangement = Arrangement.spacedBy(16.dp)
                    ) {
                        Text(
                            text = "03. Update Password",
                            style = Typography.titleSmall,
                            fontWeight = FontWeight.Bold,
                            color = Ink
                        )

                        OutlinedTextField(
                            value = currentPassword,
                            onValueChange = { currentPassword = it },
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
                                text = "Password changed successfully!",
                                color = Fresh,
                                style = Typography.bodySmall,
                                fontWeight = FontWeight.Bold
                            )
                        }

                        Button(
                            onClick = {
                                if (newPassword != confirmPassword) {
                                    passwordMatchError = "New passwords do not match"
                                } else {
                                    viewModel.changePassword(currentPassword, newPassword)
                                }
                            },
                            enabled = currentPassword.isNotEmpty() && newPassword.isNotEmpty() && confirmPassword.isNotEmpty(),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = Fresh,
                                contentColor = Bone,
                                disabledContainerColor = SlateSoft,
                                disabledContentColor = Slate
                            ),
                            shape = RoundedCornerShape(8.dp),
                            modifier = Modifier.align(Alignment.End)
                        ) {
                            Text("UPDATE PASSWORD", fontWeight = FontWeight.Bold)
                        }
                    }
                }

                item {
                    Spacer(modifier = Modifier.height(24.dp))
                }
            }
        }
    }
}
