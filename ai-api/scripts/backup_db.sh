#!/usr/bin/env bash
# INFRA-11: Database backup script for EventAI.
#
# Usage:
#   ./scripts/backup_db.sh                      # backup to ./backups/
#   ./scripts/backup_db.sh /mnt/nfs/backups      # backup to custom dir
#   PGHOST=db PGUSER=postgres ./scripts/backup_db.sh  # override connection
#
# Requires: pg_dump (from postgresql-client package)
#
# Retention: keeps the most recent N backups (default 7). Older files are deleted.

set -euo pipefail

BACKUP_DIR="${1:-./backups}"
RETENTION="${BACKUP_RETENTION:-7}"

# Connection defaults (override via env vars)
PGHOST="${PGHOST:-localhost}"
PGPORT="${PGPORT:-5432}"
PGUSER="${PGUSER:-postgres}"
PGDATABASE="${PGDATABASE:-eventai}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FILENAME="eventai_${TIMESTAMP}.sql.gz"

mkdir -p "$BACKUP_DIR"

echo "[backup] Starting pg_dump of ${PGDATABASE}@${PGHOST}:${PGPORT} ..."

pg_dump \
    -h "$PGHOST" \
    -p "$PGPORT" \
    -U "$PGUSER" \
    -d "$PGDATABASE" \
    --no-owner \
    --no-acl \
    --format=plain \
    | gzip > "${BACKUP_DIR}/${FILENAME}"

SIZE=$(du -h "${BACKUP_DIR}/${FILENAME}" | cut -f1)
echo "[backup] Saved ${BACKUP_DIR}/${FILENAME} (${SIZE})"

# Rotate old backups — keep only the N most recent
BACKUP_COUNT=$(ls -1 "${BACKUP_DIR}"/eventai_*.sql.gz 2>/dev/null | wc -l)
if [ "$BACKUP_COUNT" -gt "$RETENTION" ]; then
    DELETE_COUNT=$((BACKUP_COUNT - RETENTION))
    echo "[backup] Rotating: removing ${DELETE_COUNT} old backup(s) (keeping ${RETENTION})"
    ls -1t "${BACKUP_DIR}"/eventai_*.sql.gz | tail -n "$DELETE_COUNT" | xargs rm -f
fi

echo "[backup] Done. ${BACKUP_COUNT} backup(s) in ${BACKUP_DIR}"
