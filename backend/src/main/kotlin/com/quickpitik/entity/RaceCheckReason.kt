package com.quickpitik.entity

// AA-R1: why an automatic check sent a race to an admin (events.check_reasons, V96).
enum class RaceCheckReason {
    BAD_WORDS,
    NO_RACE_WORD,
    NON_RACE_WORD,
    DATE_OUT_OF_WINDOW,
    DAILY_CAP,
    COVER_FLAGGED,
    COVER_UNCHECKED,
    POSSIBLE_DUPLICATE,
}
