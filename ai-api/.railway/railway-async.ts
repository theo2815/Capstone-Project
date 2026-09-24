import { bucket, defineRailway, preserve, project, ref, service } from "railway/iac";

// Optional Task 10B topology. Apply only after the split-service qualification.
export const partial = "ai-api-blur";

export default defineRailway(() => {
  const inputs = bucket("ai-api-blur-inputs", { region: "sin" });
  const storage = {
    AWS_S3_BUCKET_NAME: ref(inputs, "BUCKET"),
    AWS_ENDPOINT_URL: ref(inputs, "ENDPOINT"),
    AWS_ACCESS_KEY_ID: ref(inputs, "ACCESS_KEY_ID"),
    AWS_SECRET_ACCESS_KEY: ref(inputs, "SECRET_ACCESS_KEY"),
    AWS_DEFAULT_REGION: ref(inputs, "REGION"),
    AWS_S3_URL_STYLE: "virtual",
  };

  const api = service("ai-api-blur", {
    healthcheck: "/api/v1/health/readyz",
    healthcheckTimeout: 300,
    preDeploy: "python -m alembic upgrade head",
    env: {
      ALLOWED_ORIGINS: preserve(),
      BLUR_ASYNC_ENABLED: "true",
      DATABASE_URL: preserve(),
      DEBUG: preserve(),
      ENVIRONMENT: preserve(),
      MODEL_PROFILE: "blur",
      REDIS_URL: preserve(),
      WEBHOOK_SECRET_KEY: preserve(),
      WORKERS: "1",
      ...storage,
    },
  });

  const worker = service("ai-api-blur-worker", {
    start: "celery -A src.workers.celery_app worker -Q blur,default -c 1 --loglevel=info",
    env: {
      DATABASE_URL: preserve(),
      REDIS_URL: preserve(),
      WEBHOOK_SECRET_KEY: preserve(),
      MODEL_PROFILE: "blur",
      WORKER_QUEUES: "blur,default",
      ...storage,
    },
  });

  const beat = service("ai-api-blur-beat", {
    start: "celery -A src.workers.celery_app beat --loglevel=info",
    env: {
      DATABASE_URL: preserve(),
      REDIS_URL: preserve(),
      MODEL_PROFILE: "blur",
      ...storage,
    },
  });

  return project("quickpitik-ai-api-blur", { resources: [api, worker, beat, inputs] });
});
