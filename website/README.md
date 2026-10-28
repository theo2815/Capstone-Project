# QuickPitik website

The Next.js site behind [quickpitik.com](https://quickpitik.com): the public site, runner face and bib search, and the admin pages. It is hosted on Vercel and talks only to the Spring Boot backend (`../backend`), never to `ai-api`. Project overview: the [root README](../README.md).

## Run it locally

Needs Node.js 22 and the backend running on `http://localhost:8080`.

```bash
cp .env.local.example .env.local   # backend API and WebSocket URLs, Google sign-in client ID
npm install
npm run dev                        # http://localhost:3000
```

| Script | What it does |
|---|---|
| `npm run dev` | Development server on http://localhost:3000 |
| `npm run build` | Lint, then a production build |
| `npm start` | Serve the production build |
| `npm run lint` | ESLint |
| `npm test` | Vitest tests |

## Deploys

Vercel deploys production from `main`; every other branch gets a Preview behind a Vercel login.
