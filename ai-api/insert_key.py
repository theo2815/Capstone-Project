import subprocess

sql = """INSERT INTO api_keys (id, key_hash, name, scopes, rate_tier, active) VALUES ('1625cf37-2c9d-402c-adb0-ebf343975a37', 'ac0bfe8b04cb025f60e552263f006337ae5c45c5124847b1194a07b2f25ae643', 'Desktop App', '["*"]', 'pro', true);"""

subprocess.run(["docker", "compose", "exec", "db", "psql", "-U", "postgres", "-d", "eventai", "-c", sql])
