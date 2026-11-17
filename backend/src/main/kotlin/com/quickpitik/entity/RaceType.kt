package com.quickpitik.entity

// AA-R1: a shared race event's type (events.event_type, V96). NULL = legacy or
// "Other" (the personal photographer flow); OTHER exists only in GET /race-types.
enum class RaceType(val label: String, val hint: String) {
    ROAD_RACE("Road race", "5K to 42K on roads, marathons included"),
    FUN_RUN("Fun run", "A short, casual run on foot"),
    TRAIL_RUN("Trail run", "A run off-road, on trails or mountains"),
    ULTRA("Ultramarathon", "A run longer than 42K"),
    OBSTACLE_RACE("Obstacle race", "A run with obstacles along the course"),
    CYCLING("Cycling race", "A road, mountain bike or gran fondo ride"),
}
