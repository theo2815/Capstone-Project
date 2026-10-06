package com.quickpitik.service.earnings

import com.quickpitik.common.PaginationParams
import com.quickpitik.entity.Event
import com.quickpitik.entity.EventStatus
import com.quickpitik.entity.Photo
import com.quickpitik.entity.Transaction
import com.quickpitik.repository.EventRepository
import com.quickpitik.repository.PhotoRepository
import com.quickpitik.repository.TransactionRepository
import com.quickpitik.repository.UserRepository
import com.quickpitik.service.photographer.PhotographerPhotoDtoMapper
import org.junit.jupiter.api.Test
import org.mockito.Mockito
import org.springframework.data.domain.PageImpl
import org.springframework.data.domain.PageRequest
import java.math.BigDecimal
import java.time.LocalDate
import java.time.OffsetDateTime
import java.util.UUID
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class TransactionsLedgerServiceTest {
    private val transactions = Mockito.mock(TransactionRepository::class.java)
    private val users = Mockito.mock(UserRepository::class.java)
    private val events = Mockito.mock(EventRepository::class.java)
    private val photos = Mockito.mock(PhotoRepository::class.java)
    private val photoMapper = Mockito.mock(PhotographerPhotoDtoMapper::class.java)
    private val service = TransactionsLedgerService(transactions, users, events, photos, photoMapper)
    private val seller = UUID.randomUUID()

    @Test
    fun `ledger enrichment preserves totals`() {
        val event = event()
        val photo = Photo(eventId = event.id, photographerId = seller, s3Key = "photo.jpg", pricePhp = BigDecimal("100.00"))
        val orderId = UUID.randomUUID()
        val originalPaidAt = OffsetDateTime.parse("2026-08-31T16:30:00Z")
        val refundCreatedAt = OffsetDateTime.parse("2026-09-02T03:00:00Z")
        val sale = transaction(event.id, photo.id, orderId, originalPaidAt, originalPaidAt, "75.00")
        val refund = transaction(event.id, photo.id, orderId, originalPaidAt, refundCreatedAt, "-25.00", true, sale.id)
        Mockito.`when`(transactions.pageForPhotographer(anyArg(), anyArg()))
            .thenReturn(PageImpl(listOf(refund, sale), PageRequest.of(0, 20), 2))
        Mockito.`when`(events.findAllById(setOf(event.id))).thenReturn(listOf(event))
        Mockito.`when`(photos.findAllById(setOf(photo.id))).thenReturn(listOf(photo))
        Mockito.`when`(photoMapper.thumbnailUrl(photo)).thenReturn("https://signed.example/photo")
        Mockito.`when`(transactions.pickAmountsForBucketing(anyArg(), anyArg(), anyArg())).thenReturn(
            listOf(
                arrayOf<Any>(originalPaidAt, BigDecimal("75.00")),
                arrayOf<Any>(originalPaidAt, BigDecimal("-25.00")),
            ),
        )

        val ledger = service.list(seller, PaginationParams(0, 20))

        assertEquals(mapOf("2026-09" to BigDecimal("50.00")), ledger.monthTotals)
        assertEquals(2L, ledger.total)
        val refundRow = ledger.items.single { it.id == refund.id.toString() }
        assertEquals(orderId.toString(), refundRow.orderId)
        assertTrue(refundRow.isRefund)
        assertEquals(sale.id.toString(), refundRow.refundOf)
        assertEquals(originalPaidAt, refundRow.paidAt)
        assertEquals(refundCreatedAt, refundRow.recordedAt)
        assertEquals("https://signed.example/photo", refundRow.thumbnailUrl)
    }

    @Test
    fun `absent and foreign photos never sign another seller image`() {
        val event = event()
        val foreignPhoto = Photo(eventId = event.id, photographerId = UUID.randomUUID(), s3Key = "foreign.jpg", pricePhp = BigDecimal.TEN)
        val missingPhotoId = UUID.randomUUID()
        val paidAt = OffsetDateTime.parse("2026-09-01T10:00:00Z")
        val rows = listOf(
            transaction(event.id, foreignPhoto.id, UUID.randomUUID(), paidAt, paidAt, "10.00"),
            transaction(event.id, missingPhotoId, UUID.randomUUID(), paidAt, paidAt, "10.00"),
        )
        Mockito.`when`(transactions.pageForPhotographer(anyArg(), anyArg())).thenReturn(PageImpl(rows))
        Mockito.`when`(events.findAllById(setOf(event.id))).thenReturn(listOf(event))
        Mockito.`when`(photos.findAllById(setOf(foreignPhoto.id, missingPhotoId))).thenReturn(listOf(foreignPhoto))
        Mockito.`when`(transactions.pickAmountsForBucketing(anyArg(), anyArg(), anyArg())).thenReturn(emptyList())

        val ledger = service.list(seller, PaginationParams(0, 20))

        assertTrue(ledger.items.all { it.thumbnailUrl == null })
        Mockito.verifyNoInteractions(photoMapper)
    }

    private fun event() = Event(
        slug = "ledger-${UUID.randomUUID()}",
        name = "Ledger event",
        date = LocalDate.of(2026, 9, 1),
        location = "Cebu",
        status = EventStatus.ACTIVE,
    )

    private fun transaction(
        eventId: UUID,
        photoId: UUID,
        orderId: UUID,
        paidAt: OffsetDateTime,
        createdAt: OffsetDateTime,
        amount: String,
        isRefund: Boolean = false,
        refundOf: UUID? = null,
    ) = Transaction(
        paidAt = paidAt,
        photographerId = seller,
        eventId = eventId,
        photoId = photoId,
        orderId = orderId,
        buyerDisplayName = "Aira S.",
        amountKeptPhp = BigDecimal(amount),
        isRefund = isRefund,
        refundOf = refundOf,
        createdAt = createdAt,
    )

    private fun <T> anyArg(): T = Mockito.any()
}
