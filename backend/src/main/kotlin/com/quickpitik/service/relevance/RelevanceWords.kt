package com.quickpitik.service.relevance

private const val MAX_WORDS_PER_PHOTO = 40

// RV-1.2 (plan 5.5, P12): the words of one photo worth counting. Upper-cased first; only letters, 3 to 40 of
// them, so bib numbers and most dates never pass; distinct; at most 40, in the order read.
internal fun cleanWords(raw: List<String>): List<String> = raw.asSequence()
    .map { it.uppercase() }
    .filter { it.length in 3..40 && it.all(Char::isLetter) }
    .distinct()
    .take(MAX_WORDS_PER_PHOTO)
    .toList()
