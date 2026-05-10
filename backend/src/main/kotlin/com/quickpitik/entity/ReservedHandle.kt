package com.quickpitik.entity

import jakarta.persistence.Column
import jakarta.persistence.Entity
import jakarta.persistence.Id
import jakarta.persistence.Table

@Entity
@Table(name = "reserved_handles")
class ReservedHandle(
    @Id
    @Column(name = "handle", nullable = false, length = 40, updatable = false)
    val handle: String,
)
