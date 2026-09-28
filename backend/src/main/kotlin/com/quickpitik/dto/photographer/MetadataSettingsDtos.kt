package com.quickpitik.dto.photographer

import com.fasterxml.jackson.annotation.JsonProperty
import com.fasterxml.jackson.annotation.JsonSetter
import com.fasterxml.jackson.annotation.Nulls
import com.fasterxml.jackson.core.JsonParser
import com.fasterxml.jackson.core.JsonToken
import com.fasterxml.jackson.databind.DeserializationContext
import com.fasterxml.jackson.databind.JsonDeserializer
import com.fasterxml.jackson.databind.JsonMappingException
import com.fasterxml.jackson.databind.annotation.JsonDeserialize

data class MetadataSettingsDto(
    @param:JsonProperty(required = true)
    @param:JsonSetter(nulls = Nulls.FAIL)
    @param:JsonDeserialize(using = StrictBooleanDeserializer::class)
    val metadataPublicByDefault: Boolean,
)

data class EventMetadataVisibilityDto(
    @param:JsonProperty(required = true)
    @param:JsonSetter(nulls = Nulls.FAIL)
    @param:JsonDeserialize(using = StrictBooleanDeserializer::class)
    val metadataPublic: Boolean,
)

class StrictBooleanDeserializer : JsonDeserializer<Boolean>() {
    override fun deserialize(parser: JsonParser, context: DeserializationContext): Boolean = when (parser.currentToken()) {
        JsonToken.VALUE_TRUE -> true
        JsonToken.VALUE_FALSE -> false
        else -> throw JsonMappingException.from(parser, "Expected a JSON boolean")
    }
}
