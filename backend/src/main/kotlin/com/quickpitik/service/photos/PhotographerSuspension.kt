package com.quickpitik.service.photos

import com.quickpitik.entity.Photo
import com.quickpitik.repository.UserRepository
import org.springframework.stereotype.Service
import java.util.UUID

// The Kotlin side of "a suspended photographer's photos are not for sale". The SQL copies
// are PhotoRepository.OWNER_NOT_SUSPENDED_JPQL/SQL; AA-I12b uses this one in cart and checkout.
@Service
class PhotographerSuspension(private val users: UserRepository) {
    fun suspendedOwnerIds(photos: Collection<Photo>): Set<UUID> {
        val owners = photos.mapNotNullTo(mutableSetOf()) { it.photographerId }
        if (owners.isEmpty()) return emptySet()
        return users.findSuspendedIdsIn(owners).toSet()
    }
}
