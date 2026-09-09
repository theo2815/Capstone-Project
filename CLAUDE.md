# CLAUDE.md — QuickPitik (Root)

The canonical, agent-agnostic rulebook for this project is **`AGENTS.md`** (same folder). It is imported below so Claude loads it in full — **read `AGENTS.md`; do not duplicate its content here.**

@AGENTS.md

---

## Claude-only notes

Everything above (via `AGENTS.md`) applies to Claude too. These are the Claude-specific *mechanics* — other agents can ignore them:

- **Auto-loading rules:** `.claude/rules/*.md` load automatically on matching paths — `ai-api-boundary.md` (touching `backend/`/`website/`/`mobile/`), `frontend-design.md` (web/desktop UI), `mobile-design.md` (mobile UI). Other agents must open these by path; the rule text says which.
- **Design/writing guides** (`AGENTS.md` § Design & Writing Guides) are surfaced to Claude via those rule files and the Skill tool. The guide *content* in the vault's `Claude Skills/` is plain Markdown any agent can read.
- **Session mechanics:** the `UserPromptSubmit` hook in `.claude/settings.json` re-injects `.claude/preflight-reminder.txt` each turn; `.claude/settings.local.json` holds this machine's permission allowlist (gitignored). None of this is required by the rules — it's convenience wiring for Claude Code.
