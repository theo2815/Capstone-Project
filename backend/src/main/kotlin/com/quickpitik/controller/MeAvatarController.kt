package com.quickpitik.controller

import com.quickpitik.dto.auth.UserDto
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.profile.AvatarService
import org.springframework.http.MediaType
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.annotation.DeleteMapping
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestPart
import org.springframework.web.bind.annotation.RestController
import org.springframework.web.multipart.MultipartFile

@RestController
@RequestMapping("/api/v1/me/avatar")
class MeAvatarController(
    private val avatarService: AvatarService,
) {
    @PostMapping(consumes = [MediaType.MULTIPART_FORM_DATA_VALUE])
    fun upload(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @RequestPart("file") file: MultipartFile,
    ): UserDto = avatarService.upload(
        userId = principal.userId,
        file = file.bytes,
        contentType = file.contentType,
    )

    @DeleteMapping
    fun delete(@AuthenticationPrincipal principal: AuthPrincipal): UserDto =
        avatarService.delete(principal.userId)
}
