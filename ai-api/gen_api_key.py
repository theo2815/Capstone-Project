import hashlib, uuid, secrets

raw_key = "sk_test_" + secrets.token_hex(16)
key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

print(f"Your API key: {raw_key}")
print()
print("Run this SQL:")
print(f"""INSERT INTO api_keys (id, key_hash, name, scopes, rate_tier, active)
VALUES ('{uuid.uuid4()}', '{key_hash}', 'Desktop App', '["*"]', 'pro', true);""")
