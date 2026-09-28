// Deploy this file as a Railway Function with cron */5 * * * *.
// Synthetic 64x64 JPEG; never use a runner's photo for monitoring.
const jpeg = '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAA0JCgsKCA0LCgsODg0PEyAVExISEyccHhcgLikxMC4pLSwzOko+MzZGNywtQFdBRkxOUlNSMj5aYVpQYEpRUk//2wBDAQ4ODhMREyYVFSZPNS01T09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0//wAARCABAAEADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDnaKKK7ziCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAP/9k=';

export async function run(
  env: Record<string, string | undefined> = process.env,
  request: typeof fetch = fetch,
  now: () => number = () => performance.now(),
): Promise<void> {
  const required = ['STAGING_BLUR_URL', 'BLUR_MONITOR_API_KEY', 'MONITOR_MAX_LATENCY_MS',
    'RESEND_API_KEY', 'RESEND_FROM_ADDRESS', 'ALERT_TO'];
  for (const name of required) {
    if (!env[name]?.trim()) throw new Error(`${name} is required`);
  }
  const maxLatency = Number(env.MONITOR_MAX_LATENCY_MS);
  if (!Number.isSafeInteger(maxLatency) || maxLatency <= 0) {
    throw new Error('MONITOR_MAX_LATENCY_MS must be a positive integer');
  }
  let base: URL;
  try {
    base = new URL(env.STAGING_BLUR_URL!);
    if (base.protocol !== 'https:' || base.username || base.password || base.search || base.hash || base.pathname !== '/') {
      throw new Error('invalid URL');
    }
  } catch {
    throw new Error('STAGING_BLUR_URL must be an HTTPS origin');
  }

  let failure = '';
  try {
    const ready = await request(new URL('/api/v1/health/readyz', base), {
      signal: AbortSignal.timeout(maxLatency),
    });
    if (ready.status !== 200) {
      failure = `readiness HTTP ${ready.status}`;
    } else {
      const form = new FormData();
      form.set('file', new File([Uint8Array.from(atob(jpeg), c => c.charCodeAt(0))],
        'probe.jpg', { type: 'image/jpeg' }));
      const start = now();
      const result = await request(new URL('/api/v1/blur/classify', base), {
        method: 'POST',
        headers: { 'X-API-Key': env.BLUR_MONITOR_API_KEY! },
        body: form,
        signal: AbortSignal.timeout(maxLatency),
      });
      const body = result.status === 200 ? await result.json() : null;
      const latency = now() - start;
      if (result.status !== 200) failure = `classification HTTP ${result.status}`;
      else if (body?.success !== true || !['sharp', 'defocused_object_portrait',
        'defocused_blurred', 'motion_blurred'].includes(body?.data?.predicted_class)) {
        failure = 'classification response invalid';
      } else if (latency > maxLatency) failure = `classification latency ${Math.ceil(latency)} ms exceeded ${maxLatency} ms`;
    }
  } catch {
    failure = 'probe request failed';
  }

  if (!failure) return;
  const alert = await request('https://api.resend.com/emails', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${env.RESEND_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      from: env.RESEND_FROM_ADDRESS,
      to: [env.ALERT_TO],
      subject: 'QuickPitik blur staging monitor failed',
      text: `The blur staging probe failed: ${failure}.`,
    }),
    signal: AbortSignal.timeout(10000),
  });
  if (!alert.ok) throw new Error('alert send failed');
}

if (import.meta.main) {
  run().catch(() => {
    console.error('Blur monitor failed');
    process.exitCode = 1;
  });
}
