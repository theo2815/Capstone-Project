package com.quickpitik.common

// AX-44: a search term for `LIKE ... ESCAPE '\'`, so %, _ and \ match only themselves.
fun escapeLike(term: String): String =
    term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
