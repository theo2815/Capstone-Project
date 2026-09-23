package com.quickpitik.mobile.ui.photographer

import com.quickpitik.mobile.data.local.StatusCount
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The upload bar's batch-relative arithmetic. COMPLETED rows persist forever as
 * the card-import dedupe ledger, so a plain synced/total would show a 500-photo
 * race followed by one new photo at 99% instantly — the bug the baseline exists
 * to prevent. Nothing else guards this: the VM only stores what these numbers say.
 */
class QueueProgressTest {

    private fun counts(vararg pairs: Pair<String, Int>) = pairs.map { StatusCount(it.first, it.second) }

    @Test fun `an idle queue is done, and an empty one has not started`() {
        val settled = queueProgress(counts("COMPLETED" to 500), previousBaseline = 0, wasActive = false)
        assertEquals(1f, settled.progress, 0f)
        assertFalse(settled.active)
        assertEquals(500, settled.total)
        assertEquals(0f, queueProgress(emptyList(), 0, false).progress, 0f)
    }

    @Test fun `a new batch on top of old history starts at zero, not at 99 percent`() {
        // 500 already synced from an earlier race; one photo just queued.
        val first = queueProgress(counts("COMPLETED" to 500, "QUEUED" to 1), previousBaseline = 0, wasActive = false)
        assertTrue(first.active)
        assertEquals(500, first.baseline) // the idle -> active edge snapshots the history
        assertEquals(0f, first.progress, 0f)

        // That photo lands: the batch is 1 of 1, not 501 of 501.
        val done = queueProgress(counts("COMPLETED" to 501), first.baseline, first.active)
        assertEquals(1f, done.progress, 0f)
        assertFalse(done.active)
    }

    @Test fun `the baseline holds for the life of a batch and only moves on the next idle edge`() {
        var s = queueProgress(counts("COMPLETED" to 10, "QUEUED" to 5), 0, false)
        assertEquals(10, s.baseline)
        assertEquals(0f, s.progress, 0f)
        // Two of the five have landed, one is uploading, two still wait: 2 of 5,
        // not 12 of 15.
        s = queueProgress(counts("COMPLETED" to 12, "QUEUED" to 2, "UPLOADING" to 1), s.baseline, s.active)
        assertEquals(10, s.baseline)
        assertEquals(0.4f, s.progress, 0.001f)
        // The counters the chip renders, not just the bar. Three different
        // numbers on purpose — equal ones hide a field wired to its neighbour.
        assertEquals(Triple(12, 2, 1), Triple(s.synced, s.queued, s.uploading))
        // A later batch re-snapshots from the new settled total.
        val idle = queueProgress(counts("COMPLETED" to 15), s.baseline, s.active)
        assertEquals(1f, idle.progress, 0f)
        assertEquals(15, queueProgress(counts("COMPLETED" to 15, "QUEUED" to 2), idle.baseline, idle.active).baseline)
    }

    @Test fun `failures and duplicates settle a batch so the bar still reaches 100 percent`() {
        var s = queueProgress(counts("COMPLETED" to 1, "QUEUED" to 3), 0, false)
        assertEquals(1, s.baseline)
        // One uploaded, one refused as a duplicate, one failed: nothing left to wait for.
        s = queueProgress(
            counts("COMPLETED" to 2, "DUPLICATE" to 1, "FAILED" to 1),
            s.baseline,
            s.active,
        )
        assertEquals(1f, s.progress, 0f)
        assertEquals(1, s.failed)
        assertEquals(4, s.total)
        // Mid-batch is where it matters: a failure and a duplicate refusal are
        // settled, nothing will ever move them, so a bar that ignored either
        // would stick short of full with an empty queue.
        val mid = queueProgress(counts("COMPLETED" to 1, "FAILED" to 1, "DUPLICATE" to 1, "UPLOADING" to 1), 0, true)
        assertEquals(0.75f, mid.progress, 0.001f)
        // Duplicates are also part of the history the next batch baselines against.
        assertEquals(7, queueProgress(counts("COMPLETED" to 5, "DUPLICATE" to 2, "QUEUED" to 1), 0, false).baseline)
    }

    @Test fun `a baseline above what is settled drops to it, so the bar never reads negative or freezes at zero`() {
        // Rows can leave the settled count mid-batch: Retry sends FAILED rows back to
        // QUEUED, and the worker retires COMPLETED rows whose phone copy was deleted
        // (deleteCompletedBefore). And if the queue is ever emptied, observeQueue returns
        // before calling this, so the old pair carries over.
        val pruned = queueProgress(counts("COMPLETED" to 350, "QUEUED" to 250), previousBaseline = 500, wasActive = true)
        assertEquals(0f, pruned.progress, 0f)
        assertEquals(350, pruned.baseline)
        // After an emptied queue the next batch measures from what is there, not from 500.
        val wiped = queueProgress(counts("QUEUED" to 10), previousBaseline = 500, wasActive = true)
        assertEquals(0, wiped.baseline)
        val later = queueProgress(counts("COMPLETED" to 3, "QUEUED" to 7), wiped.baseline, wiped.active)
        assertEquals(0.3f, later.progress, 0.001f)
    }
}
