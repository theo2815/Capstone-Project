package com.quickpitik.config

import org.springframework.boot.context.properties.ConfigurationProperties

// `app.admin.flags-enabled` controls whether the /admin/flags surface is
// reachable. Default false — Q-A5 leaves the flagging queue out of v1
// scope; the endpoints exist behind this gate so the schema + plumbing are
// in place when we flip the switch.
@ConfigurationProperties(prefix = "app.admin")
data class AdminProperties(
    val flagsEnabled: Boolean = true,
)
