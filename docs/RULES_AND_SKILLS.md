# Rules And Skills

## Challenge Rules From Local Project Docs

From `resources/docs/challenge_rules_faq.md` and project notes:

- Teams can have up to 3 members.
- Participants keep their IP.
- Free, publicly available external data is allowed if reproducible.
- Submission format must match `submission_template.csv`.
- The score is the average R2 across the three targets.
- Important date in the local docs: evaluation starts on `2026-03-14`.

## Repo Working Rules

These are the practical rules currently shaping work in this repository:

- Default execution environment is `ds-base`.
- Prefer reproducible scripts under `src/` over notebook-only experiments.
- Do not trust local CV alone when making submission choices.
- Keep a saved output directory for every meaningful experiment.
- Treat `submission_advanced_stacking` as the canonical recovered baseline.
- Treat online-confirmed anchors as higher-priority evidence than local estimates.
- Avoid deleting or overwriting historical output directories unless there is a clear reason.

## Current Evidence Hierarchy

When deciding what to trust, use this order:

1. Confirmed online score
2. Submission-level similarity to confirmed online anchors
3. Strict OOF / spatial CV behavior
4. Pure local estimator output

## Available Codex Skills In This Workspace

### `skill-creator`

Use when creating or updating a Codex skill.

Main guidance:

- keep skills concise
- store only essential instructions in `SKILL.md`
- move detailed material into `references/`, `scripts/`, or `assets/`
- validate skills after edits

### `skill-installer`

Use when listing installable skills or installing curated or GitHub-hosted skills.

Main guidance:

- use helper scripts instead of ad-hoc installs
- request escalation because installer scripts use network
- after install, remind the user to restart Codex

### `slides`

Use when creating or editing PowerPoint decks with the artifacts tool.

Main guidance:

- write raw JavaScript for the artifacts runtime
- save output to a user-visible path
- QA with rendered previews before handoff

### `spreadsheets`

Use when creating or editing workbooks with the artifacts tool.

Main guidance:

- write raw JavaScript for the artifacts runtime
- model workbook structure first
- recalculate before export
- verify charts and images after export

## What These Skills Mean For This Repo

Right now, these skills are support tools rather than core modeling tools.

- `slides` is useful for challenge presentation decks or final reports.
- `spreadsheets` is useful for experiment tracking workbooks.
- `skill-creator` and `skill-installer` matter only if we decide to formalize our EY workflow as a reusable Codex skill.

They do not replace the main repo workflows under `src/data`, `src/models`, and `src/evaluation`.
