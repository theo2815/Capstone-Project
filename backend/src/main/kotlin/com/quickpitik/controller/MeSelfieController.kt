package com.quickpitik.controller

import com.quickpitik.dto.profile.SelfieRefDto
import com.quickpitik.security.AuthPrincipal
import com.quickpitik.service.profile.SelfieService
import org.springframework.http.MediaType
import org.springframework.security.core.annotation.AuthenticationPrincipal
import org.springframework.web.bind.annotation.DeleteMapping
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PathVariable
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestPart
import org.springframework.web.bind.annotation.RestController
import org.springframework.web.multipart.MultipartFile
import java.util.UUID

@RestController
@RequestMapping("/api/v1/me/selfies")
class MeSelfieController(
    private val selfieService: SelfieService,
) {
    @GetMapping
    fun list(@AuthenticationPrincipal principal: AuthPrincipal): List<SelfieRefDto> =
        selfieService.list(principal.userId)

    @PostMapping(consumes = [MediaType.MULTIPART_FORM_DATA_VALUE])
    fun upload(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @RequestPart("file") file: MultipartFile,
    ): SelfieRefDto = selfieService.upload(
        userId = principal.userId,
        file = file.bytes,
        contentType = file.contentType,
        filename = file.originalFilename,
    )

    @DeleteMapping("/{id}")
    fun delete(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable id: UUID,
    ): Map<String, Boolean> =
        mapOf("removed" to selfieService.delete(principal.userId, id))

    @PostMapping("/{id}/set-primary")
    fun setPrimary(
        @AuthenticationPrincipal principal: AuthPrincipal,
        @PathVariable id: UUID,
    ): List<SelfieRefDto> = selfieService.setPrimary(principal.userId, id)
}
