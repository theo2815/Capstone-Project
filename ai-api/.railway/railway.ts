import { defineRailway, preserve, project, service } from "railway/iac";

export const partial = "ai-api-blur";

export default defineRailway(() => {
  const api = service("ai-api-blur", {
    healthcheck: "/api/v1/health/readyz",
    healthcheckTimeout: 300,
    preDeploy: "python -m alembic upgrade head",
    env: {
      ALLOWED_ORIGINS: preserve(),
      DATABASE_URL: preserve(),
      DEBUG: preserve(),
      ENVIRONMENT: preserve(),
      MODEL_PROFILE: preserve(),
      REDIS_URL: preserve(),
      WEBHOOK_SECRET_KEY: preserve(),
      WORKERS: preserve(),
    },
  });

  return project("quickpitik-ai-api-blur", { resources: [api] });
});
