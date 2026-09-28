package com.quickpitik.dto.photographer

import com.fasterxml.jackson.annotation.JsonProperty
import com.fasterxml.jackson.annotation.JsonSetter
import com.fasterxml.jackson.annotation.Nulls

data class MetadataSettingsDto(
    @param:JsonProperty(required = true)
    @param:JsonSetter(nulls = Nulls.FAIL)
    val metadataPublicByDefault: Boolean,
)

data class EventMetadataVisibilityDto(
    @param:JsonProperty(required = true)
    @param:JsonSetter(nulls = Nulls.FAIL)
    val metadataPublic: Boolean,
)
