# QuickPitik - AI API Documentation

This folder contains all documentation for the ai-api microservice.

## Table of Contents

### Core

| Document | Description |
|---|---|
| [CLAUDE.md](../CLAUDE.md) | Entry point for AI agents and new team members (lives at `ai-api/CLAUDE.md`) |
| [architecture.md](architecture.md) | Internal 4-layer architecture, model registry, patterns |
| [api-reference.md](api-reference.md) | Every API endpoint with request/response examples |
| [folder-structure.md](folder-structure.md) | Every file and folder explained |
| [tech-stack.md](tech-stack.md) | Libraries, models, and why each was chosen |

### Integration (ai-api + backends)

| Document | Description |
|---|---|
| [integration-architecture.md](integration-architecture.md) | Responsibility boundary: ai-api vs backends vs desktop |
| [integration-contracts.md](integration-contracts.md) | API contracts — how each backend calls ai-api, with code examples |

### Operations

| Document | Description |
|---|---|
| [setup-guide.md](setup-guide.md) | How to install, configure, and run the project |
| [deployment.md](deployment.md) | Docker, GPU, scaling, and production deployment |
| [maintenance-guide.md](maintenance-guide.md) | Operator runbook — monitoring, scaling, recovery, data handling |
| [security.md](security.md) | Auth, rate limiting, input validation, privacy |
| [cpp-integration.md](cpp-integration.md) | How Python and C++ work together |

### AI System

| Document | Description |
|---|---|
| [ai-system-overview.md](ai-system-overview.md) | Current state of the three ML pipelines — models, accuracy, artifacts, endpoints |
