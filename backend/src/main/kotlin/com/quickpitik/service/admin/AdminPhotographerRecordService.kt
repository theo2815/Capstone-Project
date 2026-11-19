package com.quickpitik.service.admin

import com.quickpitik.common.ErrorCodes
import com.quickpitik.dto.admin.AdminPhotographerRecordDto
import com.quickpitik.dto.admin.AdminRecordEventDto
import com.quickpitik.exception.NotFoundException
import com.quickpitik.repository.EventPhotographerRepository
import com.quickpitik.repository.SellerDetailsRepository
import com.quickpitik.repository.UserRepository
import org.springframework.data.domain.PageRequest
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.util.UUID

// AX-13: the public profile 404s for photographers who are not approved or are suspended,
// so admins read a photographer's events and record facts here.
@Service
@Transactional(readOnly = true)
class AdminPhotographerRecordService(
    private val userRepository: UserRepository,
    private val eventPhotographerRepository: EventPhotographerRepository,
    private val sellerDetailsRepository: SellerDetailsRepository,
) {
    fun record(userId: UUID): AdminPhotographerRecordDto {
        if (!userRepository.existsById(userId)) {
            throw NotFoundException(code = ErrorCodes.USER_NOT_FOUND, message = "User not found")
        }
        val totals = eventPhotographerRepository.adminRecordTotals(userId)
        val events = eventPhotographerRepository.findAdminRecordEvents(userId, PageRequest.of(0, MAX_EVENTS)).map {
            AdminRecordEventDto(it.eventId, it.name, it.date.toString(), it.visibility.wire, it.photoCount)
        }
        return AdminPhotographerRecordDto(
            events = events,
            eventsTotal = totals.eventsTotal,
            sellerDetailsStatus = sellerDetailsRepository.findById(userId).orElse(null)?.status?.reviewState(),
            salesCount = totals.salesCount,
            photoCount = totals.photoCount,
        )
    }

    private companion object {
        const val MAX_EVENTS = 200
    }
}
