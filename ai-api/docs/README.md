# EventAI - AI API Documentation

This folder contains all documentation for the ai-api microservice.

## Table of Contents

### Core

| Document | Description |
|---|---|
| [CLAUDE.md](CLAUDE.md) | Entry point for AI agents and new team members |
| [architecture.md](architecture.md) | Internal 4-layer architecture, model registry, patterns |
| [api-reference.md](api-reference.md) | Every API endpoint with request/response examples |
| [folder-structure.md](folder-structure.md) | Every file and folder explained |
| [tech-stack.md](tech-stack.md) | Libraries, models, and why each was chosen |

### Integration (ai-api + backends)

| Document | Description |
|---|---|
| [integration-architecture.md](integration-architecture.md) | Responsibility boundary: ai-api vs backends vs desktop |
| [integration-contracts.md](integration-contracts.md) | API contracts — how each backend calls ai-api, with code examples |
| [feature-analysis-report.md](feature-analysis-report.md) | Production readiness audit — issues to fix before launch |
| [rescan-audit-report.md](rescan-audit-report.md) | Post-fix rescan — 13 remaining findings (RS-1 through RS-13) |

### Operations

| Document | Description |
|---|---|
| [setup-guide.md](setup-guide.md) | How to install, configure, and run the project |
| [deployment.md](deployment.md) | Docker, GPU, scaling, and production deployment |
| [security.md](security.md) | Auth, rate limiting, input validation, privacy |
| [cpp-integration.md](cpp-integration.md) | How Python and C++ work together |

### Training & Phases

| Document | Description |
|---|---|
| [phase-plan.md](phase-plan.md) | Implementation phases and what's done vs pending |
| [phase-plan-for-blur-detection-training.md](phase-plan-for-blur-detection-training.md) | Blur classifier training — 98.68% accuracy, ONNX exported |
| [phase-plan-face-bibnumber-training.md](phase-plan-face-bibnumber-training.md) | Face + bib number combined detection training plan |
| [training-guide-face-bib.md](training-guide-face-bib.md) | Step-by-step training guide for face+bib detection model |
| [training-guide-bib.md](training-guide-bib.md) | Step-by-step training guide for bib-only detection model |
