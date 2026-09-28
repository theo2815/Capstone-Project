package com.quickpitik.repository

import com.quickpitik.entity.PhotoMetadata
import org.springframework.data.jpa.repository.JpaRepository
import java.util.UUID

interface PhotoMetadataRepository : JpaRepository<PhotoMetadata, UUID>
