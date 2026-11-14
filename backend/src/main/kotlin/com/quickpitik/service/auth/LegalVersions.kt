package com.quickpitik.service.auth

// RV-L4 (P5): is the accepted legal version at least the minimum? Dot-separated numbers compared one by one, so
// 1.10 is above 1.9; a missing part counts as 0. A null, blank or non-numeric accepted version ("1.7-draft") is never
// enough. RV-2.2a reuses it for the consent check.
internal fun atLeast(accepted: String?, min: String): Boolean {
    val a = versionParts(accepted ?: return false) ?: return false
    val m = versionParts(min) ?: return false
    for (i in 0 until maxOf(a.size, m.size)) {
        val x = a.getOrElse(i) { 0 }
        val y = m.getOrElse(i) { 0 }
        if (x != y) return x > y
    }
    return true
}

private fun versionParts(v: String): List<Int>? = v.split('.').map { it.toIntOrNull() ?: return null }
