from __future__ import annotations

from cryptography.fernet import Fernet, InvalidToken

from src.config import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

_fernet: Fernet | None = None


def _get_fernet() -> Fernet | None:
    """Return a Fernet instance using WEBHOOK_SECRET_KEY, or None if unconfigured.

    NOTE: The Fernet instance is cached on first call. Rotating
    WEBHOOK_SECRET_KEY requires a full process restart — existing secrets
    encrypted with the old key will fail to decrypt until re-encrypted.
    """
    global _fernet
    if _fernet is not None:
        return _fernet

    key = get_settings().WEBHOOK_SECRET_KEY
    if not key:
        return None

    try:
        _fernet = Fernet(key.encode())
        return _fernet
    except Exception:
        logger.error("Invalid WEBHOOK_SECRET_KEY — webhook secrets stored in plaintext")
        return None


def encrypt_secret(plaintext: str) -> str:
    """Encrypt a webhook secret. Returns plaintext if no key is configured."""
    f = _get_fernet()
    if f is None:
        return plaintext
    return f.encrypt(plaintext.encode()).decode()


def decrypt_secret(ciphertext: str) -> str:
    """Decrypt a webhook secret. Returns ciphertext as-is if no key or decryption fails."""
    f = _get_fernet()
    if f is None:
        return ciphertext
    try:
        return f.decrypt(ciphertext.encode()).decode()
    except InvalidToken:
        # Likely a plaintext secret from before encryption was enabled
        return ciphertext
