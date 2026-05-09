package com.quickpitik.service.reference

import com.fasterxml.jackson.core.type.TypeReference
import com.fasterxml.jackson.databind.ObjectMapper
import com.quickpitik.dto.reference.RegionDto
import jakarta.annotation.PostConstruct
import org.slf4j.LoggerFactory
import org.springframework.core.io.ClassPathResource
import org.springframework.stereotype.Service
import java.util.concurrent.atomic.AtomicReference

@Service
class RegionsService(
    private val objectMapper: ObjectMapper,
) {
    private val log = LoggerFactory.getLogger(javaClass)
    private val regions = AtomicReference<List<RegionDto>>(emptyList())

    @PostConstruct
    fun load() {
        val resource = ClassPathResource("data/ph-regions.json")
        require(resource.exists()) { "ph-regions.json missing from classpath" }
        val parsed: List<RegionDto> = resource.inputStream.use { stream ->
            objectMapper.readValue(stream, object : TypeReference<List<RegionDto>>() {})
        }
        regions.set(parsed)
        log.info("Loaded {} PH regions", parsed.size)
    }

    fun all(): List<RegionDto> = regions.get()

    fun isValidPair(regionCode: String, provinceCode: String): Boolean =
        regions.get().firstOrNull { it.code == regionCode }
            ?.provinces?.any { it.code == provinceCode } ?: false
}
