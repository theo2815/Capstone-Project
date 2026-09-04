// UNWIRED — zero call sites as of 2026-08-14. Retained deliberately, not dead
// by accident: this is the Wi-Fi FTP fallback for bodies whose shutter locks
// while a USB PTP session is open. It stopped being a prerequisite once the R6
// was shown to shoot happily in EOS event mode (see vault
// mobile/notes/camera-tethering), but nothing has been runtime-verified on
// hardware yet — so it stays until USB live auto-upload passes on the R6.
// Never runtime-tested. Do not treat as a working path.

package com.quickpitik.mobile.data.ftp

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.InputStream
import java.net.Inet4Address
import java.net.InetSocketAddress
import java.net.NetworkInterface
import java.net.ServerSocket
import java.net.Socket
import java.net.SocketTimeoutException
import java.util.Collections

/**
 * Minimal FTP server that RECEIVES files — just enough for a camera (Canon EOS,
 * Nikon, Sony, Fuji) configured to FTP each shot to the phone over Wi-Fi to land
 * in the app. We implement only the receive subset of RFC 959 plus the common
 * bits cameras use: binary TYPE, passive (PASV/EPSV) and active (PORT/EPRT) data
 * channels, MKD/CWD as no-ops, and STOR to capture the bytes.
 *
 * Why a Wi-Fi FTP path: a Canon body locks its physical shutter while a USB PTP
 * session is open, so USB card-poll can't run while the photographer shoots. The
 * camera's built-in FTP has NO host session — the body shoots normally and pushes
 * each frame out over Wi-Fi. Each received file is handed to [onFile], which the
 * ViewModel drops into the existing Room upload queue → PhotoUploadWorker.
 *
 * Runs on a high port (Android can't bind <1024 without root) — default 2121; set
 * the same port in the camera's FTP settings. Accepts any username/password to
 * keep field setup foolproof, and serves plain FTP (set the camera to "FTP", not
 * FTPS) on the local Wi-Fi.
 */
class FtpReceiverServer(
    private val port: Int = DEFAULT_PORT,
    private val onFile: suspend (filename: String, bytes: ByteArray) -> Unit,
    private val onLog: (String) -> Unit,
) {
    @Volatile private var serverSocket: ServerSocket? = null
    private val clients = Collections.synchronizedSet(HashSet<Socket>())

    /** Best-effort site-local IPv4 of this device — what to type into the camera. */
    fun localIpAddress(): String? = bestLocalIpv4()?.hostAddress

    suspend fun run() = coroutineScope {
        val server = ServerSocket()
        server.reuseAddress = true
        server.bind(InetSocketAddress(port))
        server.soTimeout = ACCEPT_TIMEOUT_MS // makes accept() cancellable
        serverSocket = server
        onLog(
            "FTP ready on ${bestLocalIpv4()?.hostAddress ?: "this phone"} : $port. " +
                "In the camera's FTP settings, set Server = this phone's IP, Port = $port, " +
                "mode = FTP (passive), any username/password, and turn ON auto-transfer after shooting."
        )
        try {
            while (currentCoroutineContext().isActive) {
                val client = try {
                    server.accept()
                } catch (e: SocketTimeoutException) {
                    continue
                } catch (e: Exception) {
                    if (!currentCoroutineContext().isActive) break
                    continue
                }
                clients.add(client)
                launch(Dispatchers.IO) {
                    try {
                        handleClient(client)
                    } finally {
                        clients.remove(client)
                    }
                }
            }
        } finally {
            runCatching { server.close() }
            synchronized(clients) { clients.toList() }.forEach { runCatching { it.close() } }
            serverSocket = null
            onLog("FTP server stopped.")
        }
    }

    /** Stop listening and drop any live connections (unblocks the accept/read loops). */
    fun stop() {
        runCatching { serverSocket?.close() }
        synchronized(clients) { clients.toList() }.forEach { runCatching { it.close() } }
    }

    private suspend fun handleClient(control: Socket) {
        control.soTimeout = CONTROL_TIMEOUT_MS
        val input = control.getInputStream()
        val output = control.getOutputStream()
        var dataServer: ServerSocket? = null       // passive
        var activeTarget: InetSocketAddress? = null // active (PORT/EPRT)

        fun reply(line: String) {
            output.write((line + "\r\n").toByteArray(Charsets.UTF_8))
            output.flush()
        }

        try {
            reply("220 QuickPitik FTP ready")
            var cwd = "/"
            while (currentCoroutineContext().isActive && !control.isClosed) {
                val line = readLine(input) ?: break
                if (line.isEmpty()) continue
                val sp = line.indexOf(' ')
                val cmd = (if (sp == -1) line else line.substring(0, sp)).uppercase()
                val arg = if (sp == -1) "" else line.substring(sp + 1).trim()

                when (cmd) {
                    "USER" -> reply("331 username ok, need password")
                    "PASS" -> reply("230 logged in")
                    "ACCT" -> reply("230 ok")
                    "SYST" -> reply("215 UNIX Type: L8")
                    "FEAT" -> { reply("211-Features:"); reply(" UTF8"); reply(" PASV"); reply(" EPSV"); reply("211 End") }
                    "OPTS" -> reply("200 ok")
                    "TYPE" -> reply("200 type set")
                    "MODE" -> reply("200 ok")
                    "STRU" -> reply("200 ok")
                    "NOOP" -> reply("200 noop")
                    "PWD", "XPWD" -> reply("257 \"$cwd\"")
                    "CWD", "XCWD" -> { cwd = normalizeDir(cwd, arg); reply("250 cwd ok") }
                    "CDUP" -> { cwd = parentDir(cwd); reply("250 cdup ok") }
                    "MKD", "XMKD" -> reply("257 \"${normalizeDir(cwd, arg)}\" created")
                    "RMD", "DELE" -> reply("250 ok")
                    "RNFR" -> reply("350 ready")
                    "RNTO" -> reply("250 ok")
                    "SIZE" -> reply("213 0")
                    "MDTM" -> reply("213 20260101000000")
                    "LIST", "NLST", "MLSD" -> {
                        reply("150 listing")
                        runCatching { openData(dataServer, activeTarget)?.close() }
                        dataServer?.let { runCatching { it.close() } }
                        dataServer = null; activeTarget = null
                        reply("226 listing done")
                    }
                    "PASV" -> {
                        dataServer?.let { runCatching { it.close() } }
                        val ds = ServerSocket(0).apply { soTimeout = DATA_TIMEOUT_MS }
                        dataServer = ds
                        activeTarget = null
                        val ip = (control.localAddress as? Inet4Address)?.address ?: bestLocalIpv4()?.address
                        val p = ds.localPort
                        if (ip != null && ip.size == 4) {
                            reply(
                                "227 Entering Passive Mode (%d,%d,%d,%d,%d,%d)".format(
                                    ip[0].toInt() and 0xFF, ip[1].toInt() and 0xFF,
                                    ip[2].toInt() and 0xFF, ip[3].toInt() and 0xFF,
                                    p shr 8, p and 0xFF,
                                )
                            )
                        } else {
                            reply("425 cannot enter passive mode")
                        }
                    }
                    "EPSV" -> {
                        dataServer?.let { runCatching { it.close() } }
                        val ds = ServerSocket(0).apply { soTimeout = DATA_TIMEOUT_MS }
                        dataServer = ds
                        activeTarget = null
                        reply("229 Entering Extended Passive Mode (|||${ds.localPort}|)")
                    }
                    "PORT" -> {
                        dataServer?.let { runCatching { it.close() } }; dataServer = null
                        activeTarget = parsePort(arg)
                        reply(if (activeTarget != null) "200 port ok" else "501 bad PORT")
                    }
                    "EPRT" -> {
                        dataServer?.let { runCatching { it.close() } }; dataServer = null
                        activeTarget = parseEprt(arg)
                        reply(if (activeTarget != null) "200 eprt ok" else "501 bad EPRT")
                    }
                    "STOR", "STOU", "APPE" -> {
                        val filename = sanitizeName(arg)
                        reply("150 ok to send data")
                        val data = openData(dataServer, activeTarget)
                        dataServer?.let { runCatching { it.close() } }
                        dataServer = null; activeTarget = null
                        if (data == null) { reply("425 can't open data connection"); continue }
                        val bytes = try {
                            data.getInputStream().readBytes()
                        } catch (e: Exception) {
                            runCatching { data.close() }
                            reply("426 transfer failed: ${e.message}")
                            continue
                        }
                        runCatching { data.close() }
                        if (bytes.isEmpty()) { reply("226 transfer complete (empty)"); continue }
                        onLog("received $filename (${bytes.size / 1024} KB) → queued")
                        runCatching { onFile(filename, bytes) }
                            .onFailure { onLog("  queue failed: ${it.message}") }
                        reply("226 transfer complete")
                    }
                    "QUIT" -> { reply("221 bye"); break }
                    "ABOR" -> reply("226 abort ok")
                    else -> reply("502 not implemented")
                }
            }
        } catch (e: Exception) {
            // client dropped — normal between shots; the camera reconnects
        } finally {
            runCatching { dataServer?.close() }
            runCatching { control.close() }
        }
    }

    /** Open the data connection — accept the passive socket, or dial the active target. */
    private fun openData(passive: ServerSocket?, active: InetSocketAddress?): Socket? = try {
        when {
            passive != null -> passive.accept()
            active != null -> Socket().apply { connect(active, DATA_TIMEOUT_MS) }
            else -> null
        }
    } catch (e: Exception) {
        onLog("data connection failed: ${e.message}")
        null
    }

    /** Read one CRLF-terminated control line; tolerant of idle (soTimeout) waits. */
    private fun readLine(input: InputStream): String? {
        val sb = StringBuilder()
        while (true) {
            val b = try {
                input.read()
            } catch (e: SocketTimeoutException) {
                continue // idle between commands — keep the control connection open
            } catch (e: Exception) {
                return null // socket closed / reset
            }
            if (b == -1) return if (sb.isEmpty()) null else sb.toString().trimEnd('\r')
            if (b == '\n'.code) return sb.toString().trimEnd('\r')
            sb.append(b.toChar())
        }
    }

    private fun bestLocalIpv4(): Inet4Address? = try {
        NetworkInterface.getNetworkInterfaces().toList()
            .filter { it.isUp && !it.isLoopback }
            .flatMap { it.inetAddresses.toList() }
            .filterIsInstance<Inet4Address>()
            .firstOrNull { it.isSiteLocalAddress }
    } catch (e: Exception) {
        null
    }

    private fun parsePort(arg: String): InetSocketAddress? = try {
        val n = arg.split(',').map { it.trim().toInt() }
        if (n.size != 6) null
        else InetSocketAddress("${n[0]}.${n[1]}.${n[2]}.${n[3]}", n[4] * 256 + n[5])
    } catch (e: Exception) {
        null
    }

    private fun parseEprt(arg: String): InetSocketAddress? = try {
        // |proto|addr|port|
        val parts = arg.split('|')
        if (parts.size < 4) null else InetSocketAddress(parts[2], parts[3].toInt())
    } catch (e: Exception) {
        null
    }

    private fun normalizeDir(cwd: String, arg: String): String = when {
        arg.isBlank() -> cwd
        arg.startsWith("/") -> arg
        cwd.endsWith("/") -> cwd + arg
        else -> "$cwd/$arg"
    }

    private fun parentDir(cwd: String): String =
        cwd.trimEnd('/').substringBeforeLast('/', "").ifEmpty { "/" }

    /** Keep only the file's base name — cameras STOR with a path like /DCIM/.../IMG.JPG. */
    private fun sanitizeName(arg: String): String =
        arg.substringAfterLast('/').substringAfterLast('\\')
            .ifBlank { "frame_${System.currentTimeMillis()}.jpg" }

    companion object {
        const val DEFAULT_PORT = 2121
        private const val ACCEPT_TIMEOUT_MS = 1000
        private const val CONTROL_TIMEOUT_MS = 1000
        private const val DATA_TIMEOUT_MS = 15000
    }
}
