# Team Workspaces — per-member local paths

Where each team member keeps the three shared repos on their own machine. These
values are machine-local and differ per person; the *rule* that agents read them
lives in `AGENTS.md` § External Working Directories.

**This is the only home for these paths.** Never copy a value into another doc —
link here. If a path moves, change it here and nowhere else.

## How an agent picks the right member

1. Run `git config user.email` in whichever repo the session started in.
2. Match it against a **Git identity** line below; use that member's paths.
3. **Confirm the directory exists before using it.** This file is a hypothesis;
   the filesystem is the truth.
4. If the email matches nothing, the path reads `TBD`, or the directory is
   missing → **ask the user**, then record the answer here in the same session.

Never guess a path, derive one from another member's layout, or silently fall
back to a different member's directory.

> `git config user.email` is per-repo, not per-person — a shared or lab machine
> reports the wrong identity (this repo's history already carries one such
> commit). Step 3 is what catches it.

---

## Chan, Theo Cedric — Lead Developer

- **Git identity:** `theocedricchan28@gmail.com`
- **Second Brain (vault):** `C:\Users\Theo Cedric Chan\Documents\Obsidian Vault\QuickPitik Vault`
- **BatchMyPhotos:** `C:\Users\Theo Cedric Chan\Documents\Start Up project\BatchMyPhotos`
- **Monorepo:** `C:\Users\Theo Cedric Chan\Documents\Start Up project\Capstone-Project`

## Ycoy, Dillan Marquin — Developer

- **Git identity:** `dillan@quickpitik.com` · `dillan.dillam@gmail.com` · `dillanycoy@Adubus-MacBook-Air.local`
- **Second Brain (vault):** `/Users/dillanycoy/Capstone-Obsidian-Vault`
- **BatchMyPhotos:** `TBD — ask Dillan, then record it here`
- **Monorepo:** `TBD — ask Dillan, then record it here`

## Tapales, Christian Kyle — Developer

- **Git identity:** `tapaleskyle629@gmail.com` · `63660581+christian-tapales@users.noreply.github.com`
- **Second Brain (vault):** `TBD — ask Christian, then record it here`
- **BatchMyPhotos:** `TBD — ask Christian, then record it here`
- **Monorepo:** `TBD — ask Christian, then record it here`

---

## Notes

- **Folder names differ per machine.** Theo's vault clone is `QuickPitik Vault`
  under `Obsidian Vault\`; Dillan's is the repo's default name
  `Capstone-Obsidian-Vault` in his home directory. There is **no** reliable
  relative path between the monorepo and the vault — use the absolutes above.
- **One entry per member assumes one machine each.** If someone starts working
  from a second machine, add its identity line and its own path block.
- **Kristine Purez and Brye Sy** (Documents leads) have no entry — they work in
  the vault's `Documentation for capstone paper/` workspace and need neither the
  monorepo nor BatchMyPhotos. Add them if that changes.
- Roster, roles, and name-format conventions: vault `_project/team.md`.
