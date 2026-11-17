package com.quickpitik.websocket

import org.junit.jupiter.api.Test
import org.mockito.Mockito
import java.util.UUID

class PhotoHiddenBroadcasterTest {
    @Test
    fun `the broadcaster sends one photo_hidden frame per photo after commit`() {
        val registry = Mockito.mock(EventPhotoSessionRegistry::class.java)
        val eventId = UUID.randomUUID()
        val first = UUID.randomUUID()
        val second = UUID.randomUUID()

        PhotoHiddenBroadcaster(registry).onHidden(PhotosHiddenEvent(eventId, listOf(first, second)))

        Mockito.verify(registry).broadcast(eventId, mapOf("type" to "photo.hidden", "photoId" to first))
        Mockito.verify(registry).broadcast(eventId, mapOf("type" to "photo.hidden", "photoId" to second))
        Mockito.verifyNoMoreInteractions(registry)
    }
}
