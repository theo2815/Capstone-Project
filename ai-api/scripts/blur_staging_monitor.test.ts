import assert from 'node:assert/strict';
import { test } from 'node:test';
import { run } from './blur_staging_monitor.ts';

const env = {
  STAGING_BLUR_URL: 'https://blur.example.test',
  BLUR_MONITOR_API_KEY: 'probe-secret',
  MONITOR_MAX_LATENCY_MS: '1000',
  RESEND_API_KEY: 'email-secret',
  RESEND_FROM_ADDRESS: 'alerts@example.test',
  ALERT_TO: 'theo@example.test',
};

test('healthy readiness and valid fast classification stay quiet', async () => {
  const calls: string[] = [];
  const fetcher: typeof fetch = async (input, init) => {
    const url = String(input);
    calls.push(url);
    if (url.endsWith('/readyz')) return Response.json({ status: 'ready' });
    assert.equal(init?.method, 'POST');
    assert.equal((init?.headers as Record<string, string>)['X-API-Key'], 'probe-secret');
    assert.ok((init?.body as FormData).get('file') instanceof File);
    return Response.json({ success: true, data: { predicted_class: 'sharp' } });
  };
  await run(env, fetcher, () => 0);
  assert.deepEqual(calls, [
    'https://blur.example.test/api/v1/health/readyz',
    'https://blur.example.test/api/v1/blur/classify',
  ]);
});

test('slow classification sends a bounded failure alert', async () => {
  const calls: string[] = [];
  const fetcher: typeof fetch = async (input, init) => {
    const url = String(input);
    calls.push(url);
    if (url.endsWith('/readyz')) return Response.json({ status: 'ready' });
    if (url.endsWith('/classify')) {
      return Response.json({ success: true, data: { predicted_class: 'sharp' } });
    }
    assert.equal((init?.headers as Record<string, string>).Authorization, 'Bearer email-secret');
    const email = JSON.parse(String(init?.body));
    assert.equal(email.from, 'alerts@example.test');
    assert.deepEqual(email.to, ['theo@example.test']);
    assert.match(email.text, /latency/);
    assert.doesNotMatch(email.text, /probe-secret|email-secret|blur\.example\.test/);
    return Response.json({ id: 'queued' });
  };
  let tick = 0;
  await run(env, fetcher, () => (tick++ === 0 ? 0 : 1001));
  assert.equal(calls.at(-1), 'https://api.resend.com/emails');
});

test('failed readiness alerts and failed email rejects', async () => {
  const fetcher: typeof fetch = async (input) => {
    if (String(input).endsWith('/readyz')) return new Response('', { status: 503 });
    assert.equal(String(input), 'https://api.resend.com/emails');
    return new Response('', { status: 403 });
  };
  await assert.rejects(run(env, fetcher, () => 0), /alert send failed/);
});

test('invalid configuration rejects before any request', async () => {
  const fetcher: typeof fetch = async () => { throw new Error('unexpected request'); };
  await assert.rejects(run({ ...env, MONITOR_MAX_LATENCY_MS: '0' }, fetcher), /MONITOR_MAX_LATENCY_MS/);
});
